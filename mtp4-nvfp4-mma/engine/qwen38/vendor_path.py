"""Puts the vendored packages (flash-linear-attention, einops) on sys.path."""
import os
import sys

VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")


def add():
    if VENDOR not in sys.path:
        sys.path.insert(0, VENDOR)


add()
