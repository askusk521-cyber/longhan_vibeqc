"""Pytest path bootstrap for slice-A hessian tests.

Both ``python/vibeqc`` (the package) and the repository root (the ``tools``
namespace) must be importable. Putting this at the tests/python level keeps it
scoped to the Python test run and mirrors how the evidence scripts bootstrap
their own CLI path.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "python"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
