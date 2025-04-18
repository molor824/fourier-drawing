from argparse import ArgumentError
import scipy.integrate as integ
import numpy as np
import cmath
import math

DT = 0.0001

def integrate_c(n: int, points: list[complex]):
    def time_range(dt: float):
        t = 0.0
        while t <= 1.0:
            yield t
            t += dt

    def inner(t: float):
        nonlocal n, points
        t_index = t * (len(points) + 1)
        index0 = math.floor(t_index) % len(points)
        index1 = math.ceil(t_index) % len(points)
        p0, p1 = points[index0], points[index1]
        point = (p1 - p0) * (t_index % 1) + p0
        return point * cmath.rect(1, -math.tau * n * t)
    return integ.simpson(list(map(inner, time_range(DT))), dx=DT)

class FourierSeries:
    def __init__(self, max_n: int, coefficients: np.ndarray[tuple, np.dtype[complex]]):
        if max_n * 2 + 1 > coefficients.shape[0]:
            raise ArgumentError(None, "coefficients does not match max_n size")

        self.coefficients = coefficients
        self.max_n = max_n

    def from_points(max_n: int, points: list[complex]):
        if len(points) <= 1:
            raise ArgumentError(None, "points must have at least 2 elements")
        return FourierSeries(max_n, np.fromiter(
            (integrate_c(i, points) for i in range(-max_n, max_n + 1)),
            dtype=complex
        ))

    def arrows(self, t: float):
        return (self.coefficients[i + self.max_n] * cmath.rect(1.0, cmath.tau * i * t) for i in range(-self.max_n, self.max_n + 1))