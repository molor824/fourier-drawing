from argparse import ArgumentError
import multiprocessing as mp
import scipy.integrate as integ
import numpy as np
import cmath
import math

DT = 0.001

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

    velocity = -math.tau * n / interval

    def inner(t: float):
        point = np.interp(t, timeframes, points)
        return point * cmath.rect(1, velocity * t)
    return integ.trapezoid([inner(t) for t in time_range(a, b, DT)], dx=DT) / interval

def pool_integrate_c(args):
    return integrate_c(*args)

class FourierSeries:
    def __init__(self, max_n: int, coefficients: np.ndarray[tuple, np.dtype[complex]]):
        if max_n * 2 + 1 > coefficients.shape[0]:
            raise ArgumentError(None, "coefficients does not match max_n size")

        self.coefficients = coefficients
        self.max_n = max_n

    def from_points(max_n: int, points: list[complex], timeframes: list[float]):
        if len(points) <= 1:
            raise ArgumentError(None, "points must have at least 2 elements")
        with mp.Pool() as pool:
            return FourierSeries(max_n, np.array(
                pool.map(pool_integrate_c, ((i, points, timeframes) for i in range(-max_n, max_n + 1))),
                dtype=complex
            ))

    def arrows(self, t: float):
        return (self.coefficients[i + self.max_n] * cmath.rect(1.0, cmath.tau * i * t) for i in range(-self.max_n, self.max_n + 1))