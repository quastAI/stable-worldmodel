"""What to probe for: the OGBCubeDR label registry.

Every entry names the Lance columns it reads, how those columns are reduced
into a label vector, and which of two groups it belongs to:

``state``
    Physically meaningful and *visible in the frame*, and relevant to
    planning. A representation good enough to plan with should expose these.

``nuisance``
    Domain-randomization axes — visible, but irrelevant to the task. Whether
    a JEPA latent keeps them is a genuine question, not a failure either way.

Only one target per physical quantity is registered: the arm's joint angles
and the per-cube centroid/max-height reductions are redundant with the
targets already kept (effector pose + gripper state, and the full sorted
cube positions/heights) and are not probed. Likewise the digit decal is only
probed for its class value, not its floor position or size.

Cube positions and heights are permutation-invariant (sorted by x) because
cube colours are drawn uniformly at random per episode (``cube.color`` is a
``Box(0, 1, (4, 3))``): the *set* of cube positions is visible in the frame,
but nothing says which physical cube is "block 0", so an index-based label
would not be a well-defined function of the image.

Angles go through ``sincos`` rather than being regressed raw: yaw wraps at
±π, and a linear read-out cannot represent that discontinuity, so a raw-yaw
R² would measure the parameterization rather than the representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


BLOCK_POS_COLUMNS = tuple(f'privileged/block_{i}_pos' for i in range(4))


@dataclass(frozen=True)
class ProbeTarget:
    """One probing target.

    Args:
        name: Short identifier, used as the results key.
        columns: Lance column names this target reads, in order.
        kind: ``'regression'`` or ``'classification'``.
        reduce: Name of the reducer in :data:`REDUCERS` applied to the
            stacked columns.
        group: ``'state'`` or ``'nuisance'``.
        num_classes: Class count for ``kind='classification'``.
        units: Physical unit of the label, for the MAE column of the report.
        episode_constant: True when the label never changes inside an
            episode (every domain-randomization axis is like this). Such a
            target's *effective* sample size is the number of episodes, not
            the number of frames: sampling 20 frames from one episode gives
            20 identical labels. Reported as ``n_train_effective`` so a
            score is read against the right N.
        note: Why this target is here / what to expect from it.
    """

    name: str
    columns: tuple[str, ...]
    kind: str
    reduce: str = 'concat'
    group: str = 'state'
    num_classes: int | None = None
    units: str = ''
    episode_constant: bool = False
    note: str = ''
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in ('regression', 'classification'):
            raise ValueError(f'{self.name}: bad kind {self.kind!r}')
        if self.group not in ('state', 'nuisance'):
            raise ValueError(f'{self.name}: bad group {self.group!r}')
        if self.reduce not in REDUCERS:
            raise ValueError(f'{self.name}: bad reducer {self.reduce!r}')
        if self.kind == 'classification' and not self.num_classes:
            raise ValueError(f'{self.name}: classification needs num_classes')


#######################
##      reducers     ##
#######################
#
# Each reducer takes the per-column arrays of one batch of samples --
# ``{column: (N, dim)}``, already sliced to a single timestep -- and returns
# the label array. Regression reducers return float32 ``(N, D)``; the
# classification reducer returns int64 ``(N,)``.


def _stack_blocks(cols: dict[str, np.ndarray]) -> np.ndarray:
    """``{block_i_pos: (N, 3)}`` -> ``(N, 4, 3)`` in block order."""
    return np.stack([cols[c] for c in BLOCK_POS_COLUMNS], axis=1)


def reduce_concat(cols: dict[str, np.ndarray], order) -> np.ndarray:
    """Concatenate the requested columns along the feature axis."""
    return np.concatenate([cols[c] for c in order], axis=-1).astype(np.float32)


def reduce_sincos(cols: dict[str, np.ndarray], order) -> np.ndarray:
    """Map every scalar angle to ``(sin, cos)``, avoiding the ±π wrap."""
    out = []
    for c in order:
        angle = cols[c]
        out.append(np.sin(angle))
        out.append(np.cos(angle))
    return np.concatenate(out, axis=-1).astype(np.float32)


def reduce_label(cols: dict[str, np.ndarray], order) -> np.ndarray:
    """A single scalar column as an integer class index."""
    (col,) = order
    return np.rint(cols[col].reshape(-1)).astype(np.int64)


def reduce_blocks_z_sorted(cols, order) -> np.ndarray:
    """The four cube heights, sorted — the stack profile, 4 dims."""
    return np.sort(_stack_blocks(cols)[..., 2], axis=1).astype(np.float32)


def reduce_blocks_pos_sorted(cols, order) -> np.ndarray:
    """Cube positions sorted by x, flattened — the position *set*, 12 dims.

    Sorting by a single coordinate makes the label a well-defined function
    of the image while staying agnostic to which cube is which. It is
    discontinuous where two cubes swap x order; with 4 cubes over a 0.25 m
    span that boundary is a measure-zero slice of the state space.
    """
    blocks = _stack_blocks(cols)
    order_idx = np.argsort(blocks[..., 0], axis=1)
    sorted_blocks = np.take_along_axis(blocks, order_idx[..., None], axis=1)
    return sorted_blocks.reshape(len(blocks), -1).astype(np.float32)


REDUCERS = {
    'concat': reduce_concat,
    'sincos': reduce_sincos,
    'label': reduce_label,
    'blocks_z_sorted': reduce_blocks_z_sorted,
    'blocks_pos_sorted': reduce_blocks_pos_sorted,
}


#######################
##      registry     ##
#######################

TARGETS: tuple[ProbeTarget, ...] = (
    # ---- state: arm ----
    ProbeTarget(
        name='effector_pos',
        columns=('proprio/effector_pos',),
        kind='regression',
        units='m',
        note='Gripper tip position. Directly visible; the easiest target.',
    ),
    ProbeTarget(
        name='effector_yaw',
        columns=('proprio/effector_yaw',),
        kind='regression',
        reduce='sincos',
        units='sin/cos',
        note='Wrist yaw as (sin, cos).',
    ),
    ProbeTarget(
        name='gripper_opening',
        columns=('proprio/gripper_opening',),
        kind='regression',
        units='norm',
        note='Continuous in [0, 1]. Fine-grained, small in the frame.',
    ),
    ProbeTarget(
        name='gripper_contact',
        columns=('proprio/gripper_contact',),
        kind='regression',
        units='norm',
        note='Contact signal in [0, 1] — is the gripper holding a cube.',
    ),
    # ---- state: cubes (permutation-invariant, see module docstring) ----
    ProbeTarget(
        name='cube_pos_sorted',
        columns=BLOCK_POS_COLUMNS,
        kind='regression',
        reduce='blocks_pos_sorted',
        units='m',
        note='Cube positions sorted by x — full layout, 12 dims.',
    ),
    ProbeTarget(
        name='cube_z_sorted',
        columns=BLOCK_POS_COLUMNS,
        kind='regression',
        reduce='blocks_z_sorted',
        units='m',
        note='All four cube heights, sorted — the stacking signal.',
    ),
    # ---- state: the digit decal (Run.md section 4) ----
    ProbeTarget(
        name='digit_value',
        columns=('privileged/digit_0_value',),
        kind='classification',
        reduce='label',
        num_classes=10,
        episode_constant=True,
        note=(
            'Which digit is painted on the floor. ~0.3-1.6% of the frame, '
            'and a cube can partly occlude it, so expect some label noise.'
        ),
    ),
    # ---- nuisance: domain-randomization appearance axes ----
    ProbeTarget(
        name='floor_material',
        columns=('privileged/floor_material',),
        kind='classification',
        reduce='label',
        num_classes=8,
        group='nuisance',
        episode_constant=True,
        note='Index into the 8-material floor pool.',
    ),
    ProbeTarget(
        name='wall_material',
        columns=('privileged/wall_material',),
        kind='classification',
        reduce='label',
        num_classes=8,
        group='nuisance',
        episode_constant=True,
        note='Index into the 8-material wall pool.',
    ),
    ProbeTarget(
        name='floor_rgb',
        columns=('privileged/floor_rgb',),
        kind='regression',
        group='nuisance',
        units='rgb',
        episode_constant=True,
        note='Sampled floor tint.',
    ),
    ProbeTarget(
        name='light_pos',
        columns=('privileged/light_pos',),
        kind='regression',
        group='nuisance',
        units='m',
        episode_constant=True,
        note='Light position — inferable only through shading.',
    ),
)

TARGETS_BY_NAME = {t.name: t for t in TARGETS}
GROUPS = ('state', 'nuisance')


def required_columns(targets) -> list[str]:
    """Every Lance column the given targets read, deduplicated, in order."""
    seen: list[str] = []
    for target in targets:
        for col in target.columns:
            if col not in seen:
                seen.append(col)
    return seen


def select_targets(names=None, groups=None) -> list[ProbeTarget]:
    """Resolve a target selection.

    Args:
        names: Explicit target names. Takes precedence over ``groups``.
        groups: Groups to include; ``None`` means all of them.

    Returns:
        The selected targets, in registry order.
    """
    if names:
        missing = [n for n in names if n not in TARGETS_BY_NAME]
        if missing:
            raise KeyError(
                f'Unknown probe targets {missing}. '
                f'Available: {sorted(TARGETS_BY_NAME)}'
            )
        wanted = set(names)
        return [t for t in TARGETS if t.name in wanted]

    keep = set(groups) if groups else set(GROUPS)
    bad = keep - set(GROUPS)
    if bad:
        raise KeyError(f'Unknown groups {sorted(bad)}; expected {GROUPS}')
    return [t for t in TARGETS if t.group in keep]


def build_labels(
    target: ProbeTarget,
    columns: dict[str, np.ndarray],
    step: int,
) -> np.ndarray:
    """Build ``target``'s label array for one timestep of a window batch.

    Args:
        target: The target to build.
        columns: ``{column: (N, num_steps, dim)}`` as stored by the feature
            extractor.
        step: Which timestep of the window to read the label at.

    Returns:
        ``(N, D)`` float32 for regression, ``(N,)`` int64 for classification.
    """
    sliced = {}
    for col in target.columns:
        if col not in columns:
            raise KeyError(
                f"target '{target.name}' needs column '{col}', which is not "
                'in the cached features — re-extract with it included.'
            )
        arr = np.asarray(columns[col])
        if arr.ndim != 3:
            raise ValueError(
                f"column '{col}' must be (N, num_steps, dim), got {arr.shape}"
            )
        sliced[col] = arr[:, step, :]

    labels = REDUCERS[target.reduce](sliced, target.columns)

    if target.kind == 'classification':
        lo, hi = int(labels.min()), int(labels.max())
        if lo < 0 or hi >= target.num_classes:
            raise ValueError(
                f"target '{target.name}' has class indices in [{lo}, {hi}] "
                f'but num_classes={target.num_classes}'
            )
    return labels


def label_dim(target: ProbeTarget, columns_dims: dict[str, int]) -> int:
    """Output width a probe needs for ``target``.

    Args:
        target: The target.
        columns_dims: ``{column: dim}`` of the source columns.
    """
    if target.kind == 'classification':
        return int(target.num_classes)
    dims = [columns_dims[c] for c in target.columns]
    if target.reduce == 'concat':
        return sum(dims)
    if target.reduce == 'sincos':
        return 2 * sum(dims)
    if target.reduce == 'blocks_z_sorted':
        return 4
    if target.reduce == 'blocks_pos_sorted':
        return 12
    raise ValueError(f'no label_dim rule for reducer {target.reduce!r}')


__all__ = [
    'BLOCK_POS_COLUMNS',
    'GROUPS',
    'REDUCERS',
    'TARGETS',
    'TARGETS_BY_NAME',
    'ProbeTarget',
    'build_labels',
    'label_dim',
    'required_columns',
    'select_targets',
]
