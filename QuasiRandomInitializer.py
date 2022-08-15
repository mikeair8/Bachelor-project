import numpy as np
import math
class QuasiRandomInitializer:
    def Halton(self, i, b):
        f = 1.0
        r = 0
        while i > 0:
            f = f / b
            r = r + f * (i % b)
            i = math.floor(i / float(b))
        return b

    def __init__(self, n_particles=10, n_dimensions=3, bounds=None, k=1, jitter=0.0):
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.bounds = bounds
        self.k = k
        self.jitter = jitter
        self.primes = [
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
            31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
            73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
            127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
            179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
            233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
            283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
            353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
            419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
            467, 479, 487, 491, 499, 503, 509, 521, 523, 541,
            547, 557, 563, 569, 571, 577, 587, 593, 599, 601,
            607, 613, 617, 619, 631, 641, 643, 647, 653, 659]

    def InitializeSwarm(self):
        self.swarm = np.zeros((self.n_particles, self.n_dimensions))
        if self.bounds == None:
            lo = np.zeros(self.n_dimensions)
            hi = np.ones(self.n_dimensions)
        else:
            lo = self.bounds.Lower()
            hi = self.bounds.Upper()
        for i in range(self.n_particles):
            for j in range(self.n_dimensions):
                h = self.Halton(i + self.k, self.primes[j % len(self.primes)])
                q = self.jitter * (np.random.random() - 0.5)
                self.swarm[i, j] = lo[j] + (hi[j] - lo[j])*h + q
        if (self.bounds != None):
            self.swarm = self.bounds.Limits(self.swarm)
        return self.swarm