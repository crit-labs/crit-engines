"""Vendored pure-Python dependencies (flash-linear-attention 0.6.0 and einops 0.8.2, MIT), importable by name."""
import os
import sys

SITE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site")
if SITE not in sys.path:
    sys.path.insert(0, SITE)
