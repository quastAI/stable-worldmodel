"""The LeJEPA encoder's LR schedule: linear warmup, a hold at peak, then cosine.

One scheduler covers every shape this pipeline needs, selected by a single
fraction rather than an enum:

==================  ==================================================
``constant_frac``   shape
==================  ==================================================
``0.0``             warmup, then cosine to ``eta_min`` -- a pure cosine.
``0.5``             the paper's recipe (App. H.4): "constant learning
                    rate for the first half of training, followed by
                    cosine decay to zero".
``1.0``             warmup, then flat forever -- the exploration shape.
==================  ==================================================

Why a hold phase at all, rather than the pure cosine this repo used to run: a
constant LR does not converge, it orbits in a noise ball whose radius scales
with the LR, and ``align_loss`` is a difference of the two views' embeddings --
exactly the quantity a jittering encoder inflates. So the *hold* is what buys
representation learning and the *anneal* is what makes the final number mean
something; a schedule with only one of the two is measuring the wrong thing.
Measured: a 25-epoch constant run plateaued at 2.29x its alignment floor, where
the paper's annealed runs sit at 0.976x of theirs.

Why this replaced ``LinearWarmupCosineAnnealingLR``:

* that class has no hold phase, so the paper's recipe could not be expressed;
* ``constant`` had to be encoded as ``eta_min == base_lr``, which cancels the
  cosine term arithmetically. A wrapper that dropped one kwarg in transit would
  silently hand back a real cosine annealing to zero and the run would look
  healthy for 100 epochs before its trend turned out to be an artefact of the
  LR. Here ``constant`` is ``hold_steps == max_steps``, which cannot degrade
  into an anneal;
* its cosine argument is unclamped past ``max_steps``, so a run that oversteps
  (an extra validation-driven step, a resumed counter) walks back *up* the far
  side of the cosine. This clamps.
"""

import math

from torch.optim.lr_scheduler import LRScheduler


__all__ = ['WarmupHoldCosineLR', 'schedule_kwargs']


class WarmupHoldCosineLR(LRScheduler):
    """Linear warmup to the peak LR, a hold there, then cosine to ``eta_min``.

    Stepped **per optimizer step**, not per epoch: ``max_steps`` is counted in
    optimizer steps, so on an epoch interval the counter would only ever reach
    ``max_epochs`` and the anneal would never happen (LEJEPA_RUN.md §8 defect 6).

    Args:
        optimizer: The optimizer whose groups are scheduled.
        warmup_steps: Steps of linear ramp from ``warmup_start_lr`` to the
            group's base LR. ``0`` starts at the peak.
        hold_steps: Step at which the cosine begins, counted from step 0 (so it
            includes the warmup). Clamped into ``[warmup_steps, max_steps]``.
            Equal to ``max_steps`` for a constant schedule.
        max_steps: Total optimizer steps; where the cosine reaches ``eta_min``.
        warmup_start_lr: LR at step 0. Defaults to 0.0.
        eta_min: LR at and after ``max_steps``. Defaults to 0.0.
        last_epoch: Standard torch resume counter. Defaults to -1.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps,
        hold_steps,
        max_steps,
        warmup_start_lr=0.0,
        eta_min=0.0,
        last_epoch=-1,
    ):
        if max_steps < 1:
            raise ValueError(f'max_steps must be >= 1, got {max_steps}')
        self.max_steps = int(max_steps)
        self.warmup_steps = max(0, min(int(warmup_steps), self.max_steps))
        self.hold_steps = max(
            self.warmup_steps, min(int(hold_steps), self.max_steps)
        )
        self.warmup_start_lr = float(warmup_start_lr)
        self.eta_min = float(eta_min)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """The LR for the current step, one entry per parameter group."""
        step = self.last_epoch

        if step < self.warmup_steps:
            frac = step / self.warmup_steps
            return [
                self.warmup_start_lr + (base - self.warmup_start_lr) * frac
                for base in self.base_lrs
            ]

        # `hold_steps == max_steps` means the anneal window is empty, i.e. the
        # constant shape. Checked here rather than left to the arithmetic below,
        # where an empty window would make every overstep land at `eta_min` --
        # a schedule asked to be flat forever must not fall off a cliff at
        # `max_steps`.
        if step < self.hold_steps or self.hold_steps >= self.max_steps:
            return list(self.base_lrs)

        # Clamped, so overstepping max_steps holds at eta_min instead of
        # walking back up the far side of the cosine.
        span = max(self.max_steps - self.hold_steps, 1)
        progress = min(max((step - self.hold_steps) / span, 0.0), 1.0)
        decay = (1.0 + math.cos(math.pi * progress)) / 2.0
        return [
            self.eta_min + (base - self.eta_min) * decay
            for base in self.base_lrs
        ]


def schedule_kwargs(warmup_steps, total_steps, constant_frac):
    """Resolve the config's two knobs into :class:`WarmupHoldCosineLR` kwargs.

    Kept separate from the scheduler so the shape mapping is asserted in one
    place and can be checked against the realised scheduler at train start.

    ``constant_frac`` is a fraction of the **whole** run, matching the paper's
    "first half of training" rather than the first half after warmup -- at 700
    warmup steps out of 70k the distinction is 1%, and a fraction of the total
    is what the sentence being replicated says.

    Args:
        warmup_steps: Absolute step count, clamped to ``total_steps - 1``.
        total_steps: ``max_epochs * len(train_loader)``.
        constant_frac: Fraction of the run held at the peak LR, in ``[0, 1]``.

    Returns:
        A dict with ``warmup_steps``, ``hold_steps``, ``max_steps`` and
        ``eta_min``, ready to splat into the scheduler.

    Raises:
        ValueError: If ``constant_frac`` is outside ``[0, 1]``.
    """
    frac = float(constant_frac)
    if not 0.0 <= frac <= 1.0:
        raise ValueError(
            f'schedule.constant_frac must be in [0, 1], got {frac}'
        )

    total = int(total_steps)
    warmup = max(1, min(int(warmup_steps), total - 1))
    hold = max(warmup, min(int(round(frac * total)), total))
    return {
        'warmup_steps': warmup,
        'hold_steps': hold,
        'max_steps': total,
        'eta_min': 0.0,
    }
