import numpy as np
class LinearInertia:
    def __init__(self, hi=0.9, lo=0.6):
        if hi > lo:
            self.hi = hi
            self.lo = lo
        else:
            self.hi = lo
            self.lo = hi

    def CalculateW(self, w0, iterations, max_iter):
        return self.hi - (iterations / max_iter) * (self.hi - self.lo)