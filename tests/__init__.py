"""Test suite.

Adds ``src/`` to ``sys.path`` so modules are imported exactly as inside the
container (where ``PYTHONPATH=/app/src``).

Run from the repository root:
    python -m unittest discover -s tests -t . -v
"""
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
