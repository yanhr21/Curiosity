"""Causal force-response gain diagnostic, not a learned dynamics model.

Secants describe the whole observed contact loop, not material stiffness. The
linear contraction cap is a heuristic, not a nonlinear stability guarantee.
"""
from collections import deque
import numpy as np


class ObservedForceGain:
    def __init__(self, base_gain=.000025):
        self.base_gain = float(base_gain)
        self.samples = [deque(maxlen=6), deque(maxlen=6)]
        self.slopes = [deque(), deque()]
        self.upper = np.zeros(2)
        self.applied = np.full(2, self.base_gain)

    def update(self, side, time, load, distance, dt, allow_increase):
        if dt <= 0 or not np.isfinite([time, load, distance, dt]).all():
            raise ValueError('Invalid causal force observation')
        samples = self.samples[side]; slopes = self.slopes[side]
        if samples and time <= samples[-1][0]:
            raise ValueError('Non-increasing control clock')
        samples.append((time, load, distance))
        if len(samples) == 6:
            then, old_load, old_distance = samples[0]
            dx = distance-old_distance; df = load-old_load
            if load >= .2 and old_load >= .2 and dx > 1e-6 and df > .005:
                slopes.append((time, df/dx))
        while slopes and slopes[0][0] < time-2.:
            slopes.popleft()
        if len(slopes) >= 5:
            self.upper[side] = max(v for _, v in slopes)
        upper = self.upper[side]
        candidate = min(.00015, .1/(dt*upper)) if upper > 0 else self.base_gain
        if not allow_increase:
            candidate = min(candidate, self.base_gain)
        self.applied[side] = min(candidate, self.applied[side]*1.05)
        return self.applied[side], upper, len(slopes)
