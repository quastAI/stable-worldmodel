"""Stage-D arms whose state is not pixels.

``StateWM`` (arm P) and ``OracleWM`` (arm O) both plan from the ground-truth
content vector; they differ in the transition model -- a learned predictor
against MuJoCo itself. Neither exists in the stock repo, and both are needed
for the two y-axes the plan reports every other arm on.
"""

from .oracle_wm import *  # noqa: F403
from .state_wm import *  # noqa: F403
