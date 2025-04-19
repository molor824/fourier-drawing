from argparse import ArgumentError
import scipy.integrate as integ
import numpy as np
import cmath
import math

DT = 0.0001

class FourierSeries:
    def __init__(self, max_n: int, coefficients: np.ndarray, interval: float):
        if max_n * 2 + 1 > coefficients.shape[0]:
            raise ArgumentError(None, "coefficients does not match max_n size")

        self.coefficients = coefficients
        self.max_n = max_n
        self.interval = interval

    def from_points(max_n: int, points: list[complex], timeframes: list[float]):
        if len(points) <= 1:
            raise ArgumentError(None, "points must have at least 2 elements")

        points = np.insert(np.array(points), -1, points[0])
        timeframes = np.insert(np.array(timeframes), -1, timeframes[-1] + DT)
        
        a = timeframes[0]
        b = timeframes[-1]
        interval = b - a

        sample_count = round(interval / DT)
        angular_velocity = -1j * math.tau / interval

        time_samples = np.linspace(a, b, sample_count)
        interpolated_points = np.interp(time_samples, timeframes, points)
        coefficient_numbers = np.arange(-max_n, max_n + 1)

        exponents = np.exp(np.outer(coefficient_numbers, angular_velocity * (time_samples - a)))
        coefficient_integrals = integ.simpson(exponents * interpolated_points, time_samples) / interval

        return FourierSeries(max_n, coefficient_integrals, interval)

    def arrows(self, t: float):
        return self.coefficients * np.exp(1j * cmath.tau / self.interval * t * np.arange(-self.max_n, self.max_n + 1))