"""Tandem's per-frame math in Rust, when it is installed: the One Euro filters and the hand, arm, leg and core
solvers (native/, the `tandem_core` module).

    pip install ./native           # needs Rust (https://rustup.rs); builds and installs tandem_core

Without it, vtrack runs the same math in Python: the Python classes stay the reference, and
tests/test_native.py checks that both give the same numbers. Set TANDEM_NATIVE=0 to use Python even when
tandem_core is installed.
"""
from __future__ import annotations

import os

core = None
if os.environ.get("TANDEM_NATIVE", "1") != "0":
    try:
        import tandem_core as core  # noqa: F401
    except ImportError:
        core = None

AVAILABLE = core is not None


def use(native: bool | None) -> bool:
    """Whether to use Rust: `native` None means whenever it is installed."""
    return AVAILABLE if native is None else bool(native) and AVAILABLE


def describe() -> str:
    return f"Rust (tandem_core {core.__version__})" if AVAILABLE else "Python"
