"""The LeJEPA LR schedule: warmup, hold, cosine, and the shape mapping.

The schedule is the one component in this pipeline that has already failed
silently once (LEJEPA_RUN.md section 8 defect 6: a step-counted ``max_steps``
advanced on an epoch interval, so the anneal never happened and nothing logged
it). These tests pin the realised LR at every phase boundary rather than
trusting the formula by inspection.
"""

import math

import pytest
import torch

from stable_worldmodel.wm.lejepa.schedule import (
    WarmupHoldCosineLR,
    schedule_kwargs,
)


BASE_LR = 3.0e-3


def _sched(**kwargs):
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=BASE_LR)
    return WarmupHoldCosineLR(opt, **kwargs)


def _trace(scheduler, steps):
    """The realised LR at steps ``0..steps-1``, stepping as training would."""
    out = []
    for _ in range(steps):
        out.append(scheduler.optimizer.param_groups[0]['lr'])
        scheduler.step()
    return out


# --------------------------------------------------------------------------
# schedule_kwargs: the shape mapping
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('frac', 'want_hold'),
    [(0.0, 700), (0.5, 35150), (1.0, 70300)],
)
def test_shape_mapping(frac, want_hold):
    got = schedule_kwargs(700, 70300, frac)
    assert got['hold_steps'] == want_hold
    assert got['warmup_steps'] == 700
    assert got['max_steps'] == 70300
    assert got['eta_min'] == 0.0


def test_hold_never_precedes_warmup():
    # frac * total lands inside the warmup, which would otherwise give a
    # hold_steps < warmup_steps and an inverted schedule.
    got = schedule_kwargs(700, 70300, 0.001)
    assert got['hold_steps'] == got['warmup_steps'] == 700


def test_warmup_clamped_to_budget():
    # A 10-epoch run at 703 steps/epoch cannot honour a 7030-step warmup.
    got = schedule_kwargs(70_000, 7030, 0.5)
    assert got['warmup_steps'] == 7029


@pytest.mark.parametrize('frac', [-0.01, 1.01, 2.0])
def test_frac_out_of_range_raises(frac):
    with pytest.raises(ValueError, match='constant_frac'):
        schedule_kwargs(700, 70300, frac)


# --------------------------------------------------------------------------
# WarmupHoldCosineLR: the three shapes
# --------------------------------------------------------------------------


def test_paper_shape_phases():
    """frac=0.5: ramp to peak, hold at peak, anneal to zero."""
    s = _sched(**schedule_kwargs(10, 100, 0.5))
    lrs = _trace(s, 101)

    assert lrs[0] == pytest.approx(0.0)  # warmup starts at zero
    assert lrs[5] == pytest.approx(BASE_LR * 0.5)  # linear ramp
    assert lrs[10] == pytest.approx(BASE_LR)  # peak reached
    assert lrs[49] == pytest.approx(BASE_LR)  # still held
    assert lrs[50] == pytest.approx(BASE_LR)  # anneal begins here
    assert lrs[75] == pytest.approx(BASE_LR * 0.5)  # cosine midpoint
    assert lrs[100] == pytest.approx(0.0)  # reaches eta_min

    hold = lrs[10:50]
    assert all(x == pytest.approx(BASE_LR) for x in hold)
    anneal = lrs[50:101]
    assert all(a >= b for a, b in zip(anneal, anneal[1:]))


def test_constant_shape_never_anneals():
    """frac=1.0 is flat forever -- the failure the old encoding risked."""
    s = _sched(**schedule_kwargs(10, 100, 1.0))
    lrs = _trace(s, 101)
    assert all(x == pytest.approx(BASE_LR) for x in lrs[10:])


def test_pure_cosine_shape_has_no_hold():
    """frac=0.0 starts annealing the step after warmup ends."""
    s = _sched(**schedule_kwargs(10, 100, 0.0))
    lrs = _trace(s, 101)
    assert lrs[10] == pytest.approx(BASE_LR)
    assert lrs[11] < BASE_LR
    assert lrs[55] == pytest.approx(BASE_LR * 0.5)
    assert lrs[100] == pytest.approx(0.0)


def test_pure_cosine_is_half_peak_at_midpoint():
    """Why frac=0.0 wastes a large budget: half the LR gone by halfway.

    The cosine spans ``[warmup_steps, max_steps]``, so its exact midpoint is
    step 35500 rather than the run's own halfway point at 35150 -- which is
    itself still within 0.2% of half the peak, the number that matters.
    """
    s = _sched(**schedule_kwargs(700, 70300, 0.0))
    lrs = _trace(s, 35_501)
    assert lrs[35_500] == pytest.approx(BASE_LR * 0.5)
    assert lrs[35_150] == pytest.approx(BASE_LR * 0.5, rel=2e-2)


def test_paper_shape_is_at_peak_at_midpoint():
    """The contrast: frac=0.5 still has the full LR at the halfway point."""
    s = _sched(**schedule_kwargs(700, 70300, 0.5))
    lrs = _trace(s, 35_151)
    assert lrs[35_149] == pytest.approx(BASE_LR)


# --------------------------------------------------------------------------
# Edge cases the previous scheduler got wrong
# --------------------------------------------------------------------------


def test_oversteps_clamp_at_eta_min():
    """Past max_steps the LR must hold, not walk back up the cosine.

    spt's LinearWarmupCosineAnnealingLR leaves the cosine argument unclamped,
    so a run that oversteps (an extra step from a resumed counter) climbs the
    far side of the curve back toward the peak.
    """
    s = _sched(**schedule_kwargs(10, 100, 0.5))
    lrs = _trace(s, 140)
    assert lrs[100] == pytest.approx(0.0)
    assert all(x == pytest.approx(0.0) for x in lrs[100:])


def test_zero_warmup_starts_at_peak():
    s = _sched(warmup_steps=0, hold_steps=50, max_steps=100)
    assert _trace(s, 1)[0] == pytest.approx(BASE_LR)


def test_hold_equal_to_max_steps_does_not_divide_by_zero():
    s = _sched(warmup_steps=10, hold_steps=100, max_steps=100)
    lrs = _trace(s, 120)
    assert all(x == pytest.approx(BASE_LR) for x in lrs[10:])


def test_hold_clamped_above_max_steps():
    s = _sched(warmup_steps=10, hold_steps=10_000, max_steps=100)
    assert s.hold_steps == 100


def test_eta_min_is_the_floor():
    s = _sched(warmup_steps=0, hold_steps=0, max_steps=100, eta_min=1e-5)
    lrs = _trace(s, 101)
    assert lrs[100] == pytest.approx(1e-5)
    assert min(lrs) == pytest.approx(1e-5)


def test_max_steps_below_one_raises():
    with pytest.raises(ValueError, match='max_steps'):
        _sched(warmup_steps=0, hold_steps=0, max_steps=0)


def test_all_param_groups_scheduled():
    a = torch.nn.Parameter(torch.zeros(1))
    b = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD(
        [{'params': [a], 'lr': 1e-2}, {'params': [b], 'lr': 1e-3}]
    )
    s = WarmupHoldCosineLR(opt, warmup_steps=0, hold_steps=0, max_steps=100)
    for _ in range(50):
        s.step()
    expected = (1.0 + math.cos(math.pi * 0.5)) / 2.0
    assert opt.param_groups[0]['lr'] == pytest.approx(1e-2 * expected)
    assert opt.param_groups[1]['lr'] == pytest.approx(1e-3 * expected)


def test_works_through_spt_create_scheduler():
    """The kwargs must survive spt's dispatch -- a partial is called directly.

    This is the path VerifyScheduleCallback guards: if spt ever stopped
    honouring partials, hold_steps would vanish and the paper's recipe would
    become a pure cosine.
    """
    from functools import partial

    from stable_pretraining.optim.lr_scheduler import create_scheduler

    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=BASE_LR)
    sched = schedule_kwargs(10, 100, 0.5)
    realised = create_scheduler(opt, partial(WarmupHoldCosineLR, **sched))

    assert isinstance(realised, WarmupHoldCosineLR)
    assert realised.warmup_steps == 10
    assert realised.hold_steps == 50
    assert realised.max_steps == 100
