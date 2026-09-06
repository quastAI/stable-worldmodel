"""Arm P: plan from ground-truth state, with a learned predictor.

Arm P answers "how well would the planner do if the encoder had been
*perfect*?" -- so it is the ceiling that arms A, R and C are measured against
on the ``SR / SR(P)`` axis. It shares the predictor family, capacity and
training budget with those arms; the *only* thing that differs is where the
latent comes from.

The one thing that makes this arm subtle
----------------------------------------
:meth:`StateWM.encode` **never looks at pixels**. It reads the ground-truth
content vector out of ``info`` -- the ``privileged/*`` and ``proprio/*``
columns are already in every recorded row and are forwarded by
``WorldModelPolicy._prepare_info`` -- whitens it, and writes it to
``info['emb']``.

That makes the arm's correctness depend on a *config* rather than on code: the
eval config has to list those columns in ``keys_to_cache``, or ``encode``
quietly sees nothing. A missing column does not raise; it produces a constant
embedding, a flat cost surface, and a success rate that looks exactly like a
catastrophic identifiability failure. :meth:`StateWM.encode` therefore refuses
a state vector it cannot find, and
:meth:`StateWM.assert_state_varies` is the phase-8 gate for the subtler case
where the columns are present but constant.
"""

import numpy as np
import torch
from einops import rearrange
from torch import nn


class StateWM(nn.Module):
    """Ground-truth state as the latent, with a learned transition model.

    Args:
        predictor: Same predictor family and capacity as arm A's, so the arm
            comparison isolates the representation rather than the dynamics
            model.
        action_encoder: Embeds actions for the predictor's conditioning.
        state_keys: Info columns concatenated, in order, to form the state.
            Order matters and is fixed here rather than inferred from dict
            iteration, so a checkpoint always means the same thing.
        mean: ``(d,)`` whitening mean. Stored as a buffer so it travels with
            the checkpoint.
        std: ``(d,)`` whitening scale.
        latent_dim: Width the state is projected to. ``None`` uses the raw
            whitened state, which is the default and the honest choice: arm P
            is meant to be the ground truth, not a learned re-encoding of it.
    """

    def __init__(
        self,
        predictor,
        action_encoder,
        state_keys,
        mean=None,
        std=None,
        latent_dim=None,
        **kwargs,
    ):
        super().__init__()
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.state_keys = list(state_keys)

        dim = len(mean) if mean is not None else 0
        self.register_buffer(
            'mean',
            torch.zeros(dim) if mean is None else torch.as_tensor(
                mean, dtype=torch.float32
            ),
        )
        self.register_buffer(
            'std',
            torch.ones(dim) if std is None else torch.as_tensor(
                std, dtype=torch.float32
            ),
        )
        self.projection = (
            nn.Linear(dim, latent_dim) if latent_dim else nn.Identity()
        )

    # ------------------------------------------------------------------
    # state assembly
    # ------------------------------------------------------------------

    def gather_state(self, info):
        """Concatenate the declared state columns, in declared order.

        Args:
            info: Dict of tensors, each ``(..., d_k)`` or ``(...,)``.

        Returns:
            torch.Tensor: ``(..., sum d_k)``.

        Raises:
            KeyError: If a declared column is absent. Arm P silently reading
                zeros is the failure this exists to prevent -- it is
                indistinguishable, downstream, from an encoder that learned
                nothing.
        """
        missing = [key for key in self.state_keys if key not in info]
        if missing:
            raise KeyError(
                f'StateWM is missing state columns {missing}. Add them to the '
                "eval config's `keys_to_cache`; without them this arm plans "
                'on a constant embedding and reports a failure that looks '
                'like an identifiability result.'
            )

        parts = []
        for key in self.state_keys:
            value = info[key]
            if not torch.is_tensor(value):
                value = torch.as_tensor(np.asarray(value))
            if value.ndim == 0:
                value = value.reshape(1)
            parts.append(value.float())

        target_ndim = max(part.ndim for part in parts)
        parts = [
            part.unsqueeze(-1) if part.ndim < target_ndim else part
            for part in parts
        ]
        return torch.cat(parts, dim=-1)

    def whiten(self, state):
        """Standardise with the buffers recorded at training time."""
        return (state - self.mean) / self.std.clamp_min(1e-6)

    @staticmethod
    def assert_state_varies(states, tol=1e-6):
        """The phase-8 gate: the state reaching ``encode`` must not be constant.

        The plumbing can be *present* and still wrong -- a column that is
        cached but never written arrives as a constant. This is checked across
        eval episodes rather than within one, because a single episode's state
        legitimately barely moves.

        Args:
            states: ``(N, d)`` states gathered across episodes.
            tol: Minimum per-dimension standard deviation to accept.

        Raises:
            ValueError: If every dimension is constant.
        """
        states = np.asarray(states, dtype=np.float64).reshape(len(states), -1)
        spread = states.std(axis=0)
        if spread.max() <= tol:
            raise ValueError(
                'the state vector reaching StateWM.encode is constant across '
                f'eval episodes (max std {spread.max():.2e}). The '
                '`privileged/*` columns are not being cached.'
            )
        return {
            'state_std_min': float(spread.min()),
            'state_std_max': float(spread.max()),
            'constant_dims': int((spread <= tol).sum()),
        }

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def encode(self, info):
        """Read the ground-truth state; do not touch pixels.

        Returns:
            dict: ``info`` with ``emb`` and, when actions are present,
            ``act_emb``.
        """
        state = self.whiten(self.gather_state(info))
        info['emb'] = self.projection(state)

        if 'action' in info:
            info['act_emb'] = self.action_encoder(info['action'])
        return info

    def predict(self, emb, act_emb):
        return self.predictor(emb, act_emb)

    def rollout(self, info, action_sequence, history_size: int = None):
        """Roll candidates forward in whitened state space.

        Identical in structure to
        :meth:`~stable_worldmodel.wm.lejepa.module.FrozenEncoderWM.rollout` --
        the planner, the solvers and ``GoalMSE`` are held fixed across arms, so
        the rollout semantics must be too.
        """
        if history_size is None:
            history_size = getattr(self.predictor, 'num_frames', 3)

        b, s, t = action_sequence.shape[:3]
        act_past = info.get('action_history')
        if act_past is None:
            act_past = action_sequence.new_zeros(
                b, s, 0, action_sequence.size(-1)
            )
        h_frames = act_past.size(2) + 1

        info['action'] = torch.cat(
            [act_past, action_sequence[:, :, :1]], dim=2
        )

        if 'emb' not in info:
            init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
            init = self.encode(init)
            emb = init['emb']
            if emb.ndim == 2:
                emb = emb.unsqueeze(1)
            info['emb'] = emb.unsqueeze(1).expand(b, s, h_frames, -1)

        emb_init = rearrange(info['emb'], 'b s ... -> (b s) ...')
        all_act_emb = self.action_encoder(
            torch.cat(
                [
                    rearrange(act_past, 'b s ... -> (b s) ...'),
                    rearrange(action_sequence, 'b s ... -> (b s) ...'),
                ],
                dim=1,
            )
        )

        emb_list = list(emb_init.unbind(dim=1))
        for step in range(t):
            lo = max(0, h_frames + step - history_size)
            emb_trunc = torch.stack(emb_list[lo:], dim=1)
            act_trunc = all_act_emb[:, lo : h_frames + step]
            emb_list.append(self.predict(emb_trunc, act_trunc)[:, -1])

        info['predicted_emb'] = rearrange(
            torch.stack(emb_list, dim=1), '(b s) ... -> b s ...', b=b, s=s
        )
        return info


__all__ = ['StateWM']
