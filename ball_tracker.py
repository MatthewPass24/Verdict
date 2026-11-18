import numpy as np
from filterpy.kalman import KalmanFilter


class BallTracker:
    """Constant-velocity Kalman filter for a single ball.

    Differences from the original:
      * accepts several candidate detections and picks the best one
      * gates out detections that jump too far from the prediction
        (YOLO often mistakes socks, heads, line marks for a ball)
      * gives up after `max_missed` frames so it doesn't drift forever
    """

    def __init__(self, max_missed=15, gate_px=150):
        self.max_missed = max_missed
        self.gate_px = gate_px
        self.missed = 0
        self.initialized = False

        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]], dtype=float)
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]], dtype=float)
        self.kf.R *= 10.0
        self.kf.Q *= 0.01
        self.kf.P *= 100.0

    def _pick(self, candidates):
        """candidates: list of (x, y, w, h, conf). Returns best one or None."""
        if not candidates:
            return None
        if not self.initialized:
            return max(candidates, key=lambda c: c[4])
        px, py = float(self.kf.x[0, 0]), float(self.kf.x[1, 0])
        gate = self.gate_px * (1 + self.missed * 0.25)  # widen gate while lost
        best, best_d = None, 1e9
        for c in candidates:
            d = np.hypot(c[0] - px, c[1] - py)
            if d < gate and d < best_d:
                best, best_d = c, d
        return best

    def coast(self):
        """Advance the filter one frame WITHOUT a detection and without counting a miss
        (used when the detector is deliberately skipped to save compute)."""
        if not self.initialized:
            return None
        self.kf.predict()
        return float(self.kf.x[0, 0]), float(self.kf.x[1, 0]), False

    def update(self, candidates):
        """candidates: list of (x, y, w, h, conf) or None.
        Returns (x, y, is_measured) or None if the ball is lost."""
        if self.initialized:
            self.kf.predict()

        best = self._pick(candidates or [])
        measured = False

        if best is not None:
            x, y = best[0], best[1]
            if not self.initialized:
                self.kf.x = np.array([[x], [y], [0.0], [0.0]])
                self.initialized = True
            else:
                self.kf.update(np.array([[x], [y]]))
            self.missed = 0
            measured = True
        elif self.initialized:
            self.missed += 1
            if self.missed > self.max_missed:
                self.initialized = False  # lost the ball, re-acquire next time
                self.missed = 0
                self.kf.P = np.eye(4) * 100.0
                return None

        if not self.initialized:
            return None
        return float(self.kf.x[0, 0]), float(self.kf.x[1, 0]), measured