import torch
from einops import rearrange
from torch import nn


class GCIDM(nn.Module):
    """LeWM plus a goal-conditioned inverse-dynamics *plan* head.

    Architecturally identical to :class:`~stable_worldmodel.wm.smwm.SMWM` —
    the same encoder / conditional-transformer predictor / action embedder —
    with SMWM's single-step ``inverse_model`` (``(z_t, z_{t+1}) -> a_t``)
    replaced by a ``policy_head`` that maps the predictor's own observation
    context plus a *goal* latent to the whole action-block plan joining them
    (``(z_{t-C+1..t}, z_goal) -> a_t..a_{t+H-1}``).

    Why the change: a one-step inverse model only needs whatever differs
    between two adjacent frames, which on the cube task is almost entirely
    the gripper. Over ``H`` blocks the plan has to route around and grasp an
    object, so the head cannot emit it without encoding where that object
    is — which is the task-relevant information the DR experiments found
    missing.

    The head lives *inside* the world model so it is covered by
    ``save_pretrained``/``load_pretrained`` (which serialize the model state
    dict and ``cfg.model``) and therefore survives into planning, where
    :class:`~stable_worldmodel.planning.solver.PolicyCEMSolver` uses it to
    replace CEM's zero-mean Gaussian initial proposal.

    Because ``get_action`` is exposed, the model also satisfies
    :class:`~stable_worldmodel.protocols.Actionable`, so the ordinary
    ``prepare_init_action`` warm-start tail-fill works with it too.

    The training objective lives in
    ``scripts/train/gcidm.py::gcidm_forward``::

        L = pred_loss
            + lambda_policy * policy_loss      # MSE on mu, shapes the encoder
            + lambda_std    * policy_std_loss  # NLL on a detached mu
            + lambda_sigreg * sigreg_loss      # kept, weight 0 by default

    At ``lambda_policy = 0`` this reduces to LeWM exactly.
    """

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        policy_head=None,
        **kwargs,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.policy_head = policy_head

        # `Actionable` is a runtime-checkable Protocol, so `isinstance` only
        # tests that the attribute *exists*. Bind `get_action` solely when
        # there is a head behind it: both `prepare_init_action` and
        # `ShootingCostEvaluator` branch on that isinstance check, and a
        # head-less GCIDM must fall back to zero-padding rather than be
        # treated as an actor and raise mid-solve.
        if policy_head is not None:
            self.get_action = self._get_action

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """
        pixels = info['pixels'].to(next(self.encoder.parameters()).dtype)
        b = pixels.size(0)
        pixels = rearrange(
            pixels, 'b t ... -> (b t) ...'
        )  # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info['emb'] = rearrange(emb, '(b t) d -> b t d', b=b)

        if 'action' in info:
            info['act_emb'] = self.action_encoder(info['action'])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, 'b t d -> (b t) d'))
        preds = rearrange(preds, '(b t) d -> b t d', b=emb.size(0))
        return preds

    def predict_plan(self, z_ctx, z_goal):
        """Predict the action-block plan joining a context to a goal latent.
        z_ctx:  (B, C, D) context latents, oldest first
        z_goal: (B, D)
        Returns ``(mu, sigma)``, each (B, horizon, action_dim); ``sigma`` is
        None when the head was built with ``predict_std=False``.
        """
        assert self.policy_head is not None, 'No policy head configured'
        return self.policy_head(z_ctx, z_goal)

    ####################
    ## Inference only ##
    ####################

    def _policy_context(self, info):
        """Stack the planning context and goal frames into one batch.

        Reads the *unexpanded* planning ``info_dict`` — what a solver holds
        before it broadcasts over action samples:

        - ``info['pixels']``: (B, F, C, h, w) strided context frames. ``F``
          grows from 1 to ``history_size`` over the opening steps of an
          episode, so a short context is left-padded by repeating its oldest
          frame — the same "as if the env had been stationary" convention
          ``HistoryBuffer`` uses when it has to stack envs at different fill
          levels.
        - ``info['goal']``: (B, 1, C, h, w), or (B, C, h, w).

        Returns the (B, context_frames + 1, C, h, w) stack, goal last.
        """
        assert 'pixels' in info, 'pixels not in info_dict'
        assert 'goal' in info, 'goal not in info_dict'

        pixels = info['pixels']
        n_ctx = self.policy_head.context_frames

        if pixels.size(1) < n_ctx:
            pad = pixels[:, :1].expand(
                pixels.size(0), n_ctx - pixels.size(1), *pixels.shape[2:]
            )
            pixels = torch.cat([pad, pixels], dim=1)
        pixels = pixels[:, -n_ctx:]

        goal = info['goal']
        goal = goal[:, :1] if goal.dim() == pixels.dim() else goal[:, None]

        return torch.cat([pixels, goal.to(pixels.dtype)], dim=1)

    @torch.no_grad()
    def get_action_distribution(self, info, horizon=None):
        """``(mu, sigma)`` for the plan from the current state to the goal.

        The solver-facing entry point: one encoder call over the context
        frames plus the goal frame, then the head. Returns tensors of shape
        (B, horizon, action_dim) in the *normalized* action space the solver
        operates in.
        """
        assert self.policy_head is not None, 'No policy head configured'

        head_horizon = self.policy_head.horizon
        if horizon is not None and horizon != head_horizon:
            raise ValueError(
                f'solver asked for horizon {horizon} but the policy head '
                f'was trained for {head_horizon}: plan_config.horizon must '
                'equal the wm.goal_horizon used at training time'
            )

        frames = self._policy_context(info)
        n_ctx = self.policy_head.context_frames
        emb = self.encode({'pixels': frames})['emb']
        return self.predict_plan(emb[:, :n_ctx], emb[:, n_ctx])

    def _get_action(self, info, horizon=1, prefix_actions=None):
        """``Actionable``: the plan's mean, truncated to ``horizon`` blocks.

        Exposed as ``get_action`` (bound in ``__init__`` only when a policy
        head is configured) so the model satisfies
        :class:`~stable_worldmodel.protocols.Actionable` and the generic
        ``prepare_init_action`` tail-fill works. ``prefix_actions`` is
        accepted for protocol compatibility but ignored: the head is
        conditioned on observations and a goal, not on a partial plan, so
        there is no latent to advance. The returned tail is therefore the
        *last* ``horizon`` blocks of the plan, which is the segment those
        prefix actions would have led into.
        """
        mu, _ = self.get_action_distribution(info)
        if horizon > mu.size(1):
            raise ValueError(
                f'policy head predicts {mu.size(1)} blocks, '
                f'cannot fill {horizon}'
            )
        return mu[:, mu.size(1) - horizon :]

    def rollout(self, info, action_sequence, history_size: int = None):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, H, C, h, w) — H context frames (block timesteps)
        action_sequence: (B, S, T, action_dim) — strictly-future candidates
        info['action_history']: (B, S, H - 1, action_dim) — executed action
            blocks between the context frames (required when H > 1)
         - S is the number of action plan samples
         - T is the planning horizon
        Returns ``info`` with ``predicted_emb`` of shape (B, S, H + T, D);
        the first H entries are the encoded context frames.
        """
        if history_size is None:
            history_size = getattr(self.predictor, 'num_frames', 3)

        assert 'pixels' in info, 'pixels not in info_dict'
        H = info['pixels'].size(2)
        B, S, T = action_sequence.shape[:3]
        act_past = info.get('action_history')
        if act_past is None:
            act_past = action_sequence.new_zeros(
                B, S, 0, action_sequence.size(-1)
            )
        assert act_past.size(2) == H - 1, (
            f'action_history must hold H-1={H - 1} executed blocks, '
            f'got {act_past.size(2)}'
        )
        # action paired with context frame k is the block leaving it; the
        # current frame (k = H-1) pairs with the first candidate
        info['action'] = torch.cat(
            [act_past, action_sequence[:, :, :1]], dim=2
        )

        # encode initial state, or reuse cached embedding from a prior rollout.
        # detach: to avoid backprop in encoder
        if 'emb' not in info:
            _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
            _init = self.encode(_init)
            info['emb'] = (
                _init['emb'].detach().unsqueeze(1).expand(B, S, -1, -1)
            )

        # flatten batch and sample dimensions for rollout
        emb_init = rearrange(info['emb'], 'b s ... -> (b s) ...')
        act_past_flat = rearrange(act_past, 'b s ... -> (b s) ...')
        act_cand_flat = rearrange(action_sequence, 'b s ... -> (b s) ...')
        all_act_emb = self.action_encoder(
            torch.cat([act_past_flat, act_cand_flat], dim=1)
        )  # (BS, H - 1 + T, A_emb); index k = block leaving frame k

        # rollout predictor autoregressively, one step per candidate
        # emb_list holds individual (BS, D) frames, each with its own grad_fn
        HS = history_size
        emb_list = list(emb_init.unbind(dim=1))  # H tensors of shape (BS, D)
        for t in range(T):
            lo = max(0, H + t - HS)
            emb_trunc = torch.stack(emb_list[lo:], dim=1)  # (BS, HS, D)
            act_trunc = all_act_emb[:, lo : H + t]  # (BS, HS, A_emb)
            emb_list.append(self.predict(emb_trunc, act_trunc)[:, -1])

        emb = torch.stack(emb_list, dim=1)  # (BS, H + T, D)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, '(b s) ... -> b s ...', b=B, s=S)
        info['predicted_emb'] = pred_rollout

        return info


__all__ = ['GCIDM']
