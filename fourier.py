from argparse import ArgumentError
import scipy.integrate as integ
import numpy as np
import cmath
import math

DT = 0.0001

def time_range(a: float, b: float, dt: float):
    t = a
    while t <= b:
        yield t
        t += dt

def integrate_c(n: int, points: list[complex], timeframes: list[float]):
    points = points + [points[0]]
    timeframes = timeframes + [timeframes[-1] + DT]

    a = timeframes[0]
    b = timeframes[-1]
    interval = b - a

    sample_count = round(interval / DT)
    theta = -1j * math.tau * n / interval
    time_samples = np.linspace(a, b, sample_count)

    return integ.simpson(
        np.interp(time_samples, timeframes, points) * np.exp((time_samples - a) * theta),
        time_samples,
    ) / interval

def pool_integrate_c(args):
    return integrate_c(*args)

class FourierSeries:
    def __init__(self, max_n: int, coefficients: np.ndarray[tuple, np.dtype[complex]], interval: float):
        if max_n * 2 + 1 > coefficients.shape[0]:
            raise ArgumentError(None, "coefficients does not match max_n size")

        self.coefficients = coefficients
        self.max_n = max_n
        self.interval = interval

    def from_points(max_n: int, points: list[complex], timeframes: list[float]):
        if len(points) <= 1:
            raise ArgumentError(None, "points must have at least 2 elements")
        return FourierSeries(max_n, np.fromiter(
            (integrate_c(i, points, timeframes) for i in range(-max_n, max_n + 1)),
            dtype=complex
        ), timeframes[-1] - timeframes[0])

    def arrows(self, t: float):
        return self.coefficients * np.exp(1j * cmath.tau / self.interval * t * np.arange(-self.max_n, self.max_n + 1))