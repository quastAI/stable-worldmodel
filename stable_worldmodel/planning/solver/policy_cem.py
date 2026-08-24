"""CEM warm-started from a model's goal-conditioned policy head."""

from typing import Any

import torch

from .cem import CEMSolver


class PolicyCEMSolver(CEMSolver):
    """CEM whose *initial* proposal distribution comes from a policy head.

    Plain :class:`~stable_worldmodel.planning.solver.CEMSolver` opens every
    solve from ``N(0, var_scale)`` — a zero-mean plan of the right shape and
    nothing more. This subclass asks the world model for
    ``get_action_distribution(info_dict, horizon)`` and uses the returned
    ``(mu, sigma)`` as iteration 0's mean and spread instead. Iterations
    ``1..n_steps`` are ordinary CEM: elite mean and elite std, unchanged. The
    policy owns the initialization only.

    Two modes:

    ``'replace'`` (default)
        The head proposes the full horizon and any caller-supplied
        ``init_action`` (the previous solve's leftover plan, under
        ``PlanConfig.warm_start``) is discarded. This is the mode that
        isolates the head's contribution, and it is what
        ``cube_quadruple_dr_gcidm.yaml`` ships — paired with
        ``plan_config.warm_start: false`` so no leftover plan is built at all.

    ``'tail'``
        The previous plan keeps its prefix and the head fills only the
        remaining steps, via the generic ``prepare_init_action`` path. Needs
        the wrapped cost object to forward ``get_action`` to the model
        (``ShootingCostEvaluator`` does).

    Note on ``sigma`` vs ``var``: despite the name, ``CEMSolver``'s ``var`` is
    a **standard deviation** — ``candidates = randn * var + mean``, and it is
    refitted as ``topk_candidates.std(...)``. So the head's ``sigma`` is used
    as-is, scaled by ``var_scale`` and floored by ``var_floor``; it is never
    squared.

    Args:
        warm_start_mode: ``'replace'`` or ``'tail'`` (see above).
        var_floor: Lower clamp on the per-dim proposal std. Guards against a
            confidently-wrong head collapsing the search to a point; the
            floor is what lets CEM still explore away from the proposal.
        **kwargs: Forwarded to :class:`CEMSolver`.
    """

    def __init__(
        self,
        *args: Any,
        warm_start_mode: str = 'replace',
        var_floor: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        if warm_start_mode not in ('replace', 'tail'):
            raise ValueError(
                f"warm_start_mode must be 'replace' or 'tail', "
                f'got {warm_start_mode!r}'
            )
        self.warm_start_mode = warm_start_mode
        self.var_floor = var_floor
        self._policy_std: torch.Tensor | None = None

    @property
    def actor(self):
        """The model exposing ``get_action_distribution``.

        Unwraps the ``ShootingCostEvaluator`` seam: solvers are handed a cost
        object, and for the composed evaluator the model is ``cost.model``.
        Models that expose ``get_cost`` natively are their own actor.
        """
        model = getattr(self.cost, 'model', self.cost)
        if not hasattr(model, 'get_action_distribution'):
            raise TypeError(
                f'{type(model).__name__} has no get_action_distribution; '
                'PolicyCEMSolver needs a model with a policy head (e.g. '
                'stable_worldmodel.wm.gcidm.GCIDM). Use CEMSolver instead.'
            )
        return model

    @torch.inference_mode()
    def solve(
        self, info_dict: dict, init_action: torch.Tensor | None = None
    ) -> dict:
        """Query the policy head, then run the inherited CEM loop."""
        mu, std = self.actor.get_action_distribution(
            info_dict, horizon=self.horizon
        )

        n_envs = mu.shape[0]
        mu = mu.reshape(n_envs, self.horizon, self.action_dim).to(
            device=self.device, dtype=self.dtype
        )

        # Stashed for init_action_distrib(), which the inherited solve()
        # calls but which is not passed the info_dict.
        self._policy_std = None
        if std is not None:
            self._policy_std = (
                (self.var_scale * std.reshape_as(mu))
                .to(device=self.device, dtype=self.dtype)
                .clamp_min(self.var_floor)
            )

        if self.warm_start_mode == 'replace':
            # A full-horizon mean makes prepare_init_action a pass-through,
            # so the head's plan reaches the sampler untouched.
            #
            # Cloned because CEMSolver writes its optimized plan *back into*
            # the tensor it was handed here (`mean[start:end] = batch_mean`,
            # where `mean` is this very tensor when it already covers the
            # horizon). The head's output is not ours to overwrite.
            init_action = mu.clone()

        try:
            return super().solve(info_dict, init_action=init_action)
        finally:
            self._policy_std = None

    def init_action_distrib(
        self, n_envs: int, actions: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Substitute the head's spread for CEM's flat ``var_scale``."""
        mean, var = super().init_action_distrib(n_envs, actions)
        if self._policy_std is not None:
            var = self._policy_std[:n_envs].clone()
        return mean, var
