import numpy as np
import pytest

from vtrack.geometry import (L_ELBOW, L_HIP, L_SHOULDER, L_WRIST, R_ELBOW, R_HIP, R_SHOULDER, R_WRIST)

MCP_X = {"index": -0.03, "middle": -0.01, "ring": 0.01, "pinky": 0.03}
START = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}


def make_hand(extended=(), thumb=False):
    """Synthetic 21-point hand, wrist at origin, palm along +y, curled fingers fold toward -z."""
    h = np.zeros((21, 3))
    for name, i in START.items():
        x = MCP_X[name]
        h[i] = (x, 0.09, 0)
        if name in extended:
            h[i + 1], h[i + 2], h[i + 3] = (x, 0.13, 0), (x, 0.16, 0), (x, 0.19, 0)
        else:
            h[i + 1], h[i + 2], h[i + 3] = (x, 0.13, 0), (x, 0.13, -0.03), (x, 0.11, -0.03)
    if thumb:
        h[1], h[2], h[3], h[4] = (-0.02, 0.02, 0), (-0.04, 0.035, 0), (-0.06, 0.05, 0), (-0.08, 0.065, 0)
    else:
        h[1], h[2], h[3], h[4] = (-0.02, 0.02, 0), (-0.04, 0.035, 0), (-0.03, 0.06, -0.015), (0.0, 0.07, -0.02)
    return h


@pytest.fixture
def make_pose():
    """Torso: right = +x, up = +y, forward = -z (so forward raise means negative z)."""
    def _make(r_elbow=(0.2, 1.1, 0), r_wrist=(0.2, 0.8, 0), l_elbow=(-0.2, 1.1, 0), l_wrist=(-0.2, 0.8, 0)):
        p = np.zeros((33, 3))
        p[R_SHOULDER], p[L_SHOULDER] = (0.2, 1.4, 0), (-0.2, 1.4, 0)
        p[R_HIP], p[L_HIP] = (0.1, 0.9, 0), (-0.1, 0.9, 0)
        p[R_ELBOW], p[R_WRIST], p[L_ELBOW], p[L_WRIST] = r_elbow, r_wrist, l_elbow, l_wrist
        return p
    return _make
