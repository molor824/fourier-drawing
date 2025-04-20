from argparse import ArgumentError
import multiprocessing as mp
import scipy.integrate as integ
import numpy as np
import cmath
import math

DT = 0.0001

_cached_pool = None # it's expensive to create pool everytime because it basically copies the same python project cpu_count times

def integrate_c_pool(args):
    n, thetas, interpolated_points, time_samples = args
    return integ.simpson(np.exp(n * thetas) * interpolated_points, time_samples)

class FourierSeries:
    def __init__(self, max_n: int, coefficients: np.ndarray, interval: float):
        if max_n * 2 + 1 > coefficients.shape[0]:
            raise ArgumentError(None, "coefficients does not match max_n size")

        self.coefficients = coefficients
        self.max_n = max_n
        self.interval = interval

    def from_points(max_n: int, points: list[complex], timeframes: list[float]):
        global _cached_pool

        if len(points) <= 1:
            raise ArgumentError(None, "points must have at least 2 elements")

        points = points + [points[0]]
        timeframes = timeframes + [timeframes[-1] + DT]
        
        a = timeframes[0]
        b = timeframes[-1]
        interval = b - a

        sample_count = round(interval / DT)
        angular_velocity = -1j * math.tau / interval

        time_samples = np.linspace(a, b, sample_count)
        interpolated_points = np.interp(time_samples, timeframes, points)

        thetas = angular_velocity * (time_samples - a)

        if _cached_pool is None:
            _cached_pool = mp.Pool(mp.cpu_count())
        
        coefficients = np.array(
            _cached_pool.map(integrate_c_pool, ((n, thetas, interpolated_points, time_samples) for n in range(-max_n, max_n + 1)))
        ) / interval
        return FourierSeries(max_n, coefficients, interval)

    def arrows(self, t: float):
        return self.coefficients * np.exp(1j * cmath.tau / self.interval * t * np.arange(-self.max_n, self.max_n + 1))