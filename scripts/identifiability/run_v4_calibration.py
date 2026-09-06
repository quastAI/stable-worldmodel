"""Phase 6: reproduce the Table-2 anisotropy boundary before any production sweep.

The V4 boundary is the one prediction in the programme that is *quantitative
and published*: recovery should fail once

    min rho_alpha <= (max rho_alpha)^2

because past that point the encoder can lower its objective by representing the
second Hermite function of a fast latent in place of a slow one it should have
kept. Two things must be shown, and this script is what shows them:

1. the sign flip in ``R^2(z->h)`` lands **on** the criterion, not merely
   somewhere in its vicinity; and
2. the second-Hermite substitution is *detectable* beyond it -- otherwise the
   observed degradation might be some other failure that happens to co-occur.

Why this runs before the production sweep
-----------------------------------------
It is the only cheap opportunity to find out that the metric suite cannot
resolve the effect it was built to measure. Discovering that after the full
severity x seed budget has been spent would invalidate the budget, not just the
V4 rows.

The ladder is the one :meth:`AnisotropicTransitions.calibration_ladder`
produces -- rungs bracketing the crossing, not a uniform sweep. At the frozen
``rho = 0.9`` with ``n = 9`` the crossing sits near severity 0.32, and a
uniform ``(0, .25, .5, .75, 1)`` ladder would place exactly one rung either side
of it with nothing nearby.

This script uses a **synthetic encoder** -- the analytically-optimal linear
response to each anisotropy level -- rather than training one. Training would
confound the boundary with optimisation noise; the point here is to verify that
the *measurement apparatus* resolves the boundary, and a synthetic encoder is
the only way to know where the truth lies.

Usage::

    python scripts/identifiability/run_v4_calibration.py
    python scripts/identifiability/run_v4_calibration.py n=16 num_samples=40000
"""

import json
from pathlib import Path

import hydra
import numpy as np
from loguru import logger as logging
from omegaconf import DictConfig

from stable_worldmodel.identifiability import metrics as ident_metrics
from stable_worldmodel.identifiability.ou import OUSampler, isotropy_criterion
from stable_worldmodel.identifiability.violations import AnisotropicTransitions


def hermite_substituted_encoder(z, rho, slow_index, strength):
    """A synthetic encoder that substitutes ``He2`` of the fastest latent.

    ``He2(x) = x^2 - 1``. ``strength`` interpolates between a faithful
    orthogonal image of ``z`` and one where the slowest coordinate has been
    replaced by the second Hermite function of the fastest -- exactly the
    failure the theory predicts past the boundary.

    Args:
        z: ``(B, n)`` latents.
        rho: ``(n,)`` per-dimension autocorrelation.
        slow_index: Which coordinate is slowest.
        strength: 0 = faithful, 1 = fully substituted.

    Returns:
        ndarray: ``(B, n)`` embeddings.
    """
    fast_index = int(np.argmax(rho))
    h = z.copy()
    hermite = z[:, fast_index] ** 2 - 1.0
    hermite = hermite / (hermite.std() + 1e-12)
    h[:, slow_index] = (
        (1.0 - strength) * z[:, slow_index] + strength * hermite
    )
    return h


@hydra.main(version_base=None, config_path='./config', config_name='v4_calibration')
def run(cfg: DictConfig):
    n = int(cfg.n)
    rho_bar = float(cfg.rho)
    num_samples = int(cfg.num_samples)
    seed = int(cfg.seed)

    violation = AnisotropicTransitions()
    critical = violation.critical_severity(n, rho_bar)
    ladder = violation.calibration_ladder(n, rho_bar)

    logging.info(
        f'n = {n}, rho_bar = {rho_bar}; criterion crosses at severity '
        f'{critical:.4f}'
    )

    rows = []
    header = (
        f'{"severity":>9}{"rho_min":>9}{"rho_max":>9}{"margin":>10}'
        f'{"past?":>7}{"R2(z->h)":>10}{"R2(h->z)":>10}'
        f'{"proc/dim":>10}{"He2 exc":>9}'
    )
    print()
    print(header)
    print('-' * len(header))

    for severity in ladder:
        rung = AnisotropicTransitions(severity=float(severity))
        rho = rung.rho_vector(n, rho_bar)
        iso = isotropy_criterion(rho)
        slow_index = int(np.argmin(rho))

        sampler = OUSampler(n=n, rho=rho, seed=seed)
        z, z_next, _ = sampler.sample_pairs(num_samples)

        # Substitution grows once the criterion is violated -- the theory's
        # claim, made concrete so the metrics can be checked against a case
        # whose ground truth is known.
        strength = 0.0 if iso['satisfied'] else min(
            1.0, -iso['margin'] / 0.05
        )
        h = hermite_substituted_encoder(z, rho, slow_index, strength)
        h_next = hermite_substituted_encoder(z_next, rho, slow_index, strength)

        scores = ident_metrics.compute_all(
            z, h, z_next, h_next, rho=rho, seed=seed
        )
        row = {
            'severity': float(severity),
            'rho_min': iso['rho_min'],
            'rho_max': iso['rho_max'],
            'margin': iso['margin'],
            'past_boundary': not iso['satisfied'],
            'substitution_strength': float(strength),
            **{
                k: scores[k]
                for k in (
                    'r2_z_to_h',
                    'r2_h_to_z',
                    'procrustes_mse_per_dim',
                    'hermite2_excess',
                    'orth_err_normalized',
                )
            },
        }
        rows.append(row)
        print(
            f'{severity:>9.4f}{iso["rho_min"]:>9.4f}{iso["rho_max"]:>9.4f}'
            f'{iso["margin"]:>10.5f}{str(not iso["satisfied"]):>7}'
            f'{scores["r2_z_to_h"]:>10.4f}{scores["r2_h_to_z"]:>10.4f}'
            f'{scores["procrustes_mse_per_dim"]:>10.4f}'
            f'{scores["hermite2_excess"]:>9.4f}'
        )

    # ------------------------------------------------------------ verdict
    below = [r for r in rows if not r['past_boundary']]
    above = [r for r in rows if r['past_boundary']]

    checks = {}
    if below and above:
        checks['recovery_degrades_past_boundary'] = bool(
            max(r['procrustes_mse_per_dim'] for r in below)
            < min(r['procrustes_mse_per_dim'] for r in above)
        )
        checks['r2_z_to_h_drops_past_boundary'] = bool(
            min(r['r2_z_to_h'] for r in below)
            > max(r['r2_z_to_h'] for r in above)
        )
        checks['hermite2_detectable_past_boundary'] = bool(
            max(r['hermite2_excess'] for r in below)
            < max(r['hermite2_excess'] for r in above)
        )
    else:
        checks['ladder_brackets_boundary'] = False

    print()
    for name, passed in checks.items():
        print(f'  {"PASS" if passed else "FAIL"}  {name}')

    out = Path(cfg.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as handle:
        json.dump(
            {
                'n': n,
                'rho_bar': rho_bar,
                'critical_severity': critical,
                'ladder': list(ladder),
                'rows': rows,
                'checks': checks,
                'metric_suite_version': ident_metrics.METRIC_SUITE_VERSION,
            },
            handle,
            indent=2,
        )
    print(f'\nwritten -> {out}')

    if not all(checks.values()):
        raise SystemExit(
            'V4 calibration failed: the metric suite does not resolve the '
            'published boundary. Fix this before spending the sweep budget -- '
            'every V4 row would otherwise be uninterpretable.'
        )


if __name__ == '__main__':
    run()
