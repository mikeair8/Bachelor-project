import numpy as np
class SphereInitializer:
    def __init__(self, n_particles, n_dimensions, bounds=None):
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.bounds = bounds

    def InitializeSwarm(self):
        self.swarm = np.zeros((self.n_particles, self.n_dimensions))
        if self.bounds == None:
            lo = np.zeros(self.n_dimensions)
            hi = np.ones(self.n_dimensions)
        else:
            lo = self.bounds.Lower()
            hi = self.bounds.Upper()
        radius = 0.5
        for i in range(self.n_particles):
            p = np.random.normal(size=self.n_dimensions)
            self.swarm[i] = radius + radius * p / np.sqrt(np.dot(p, p))
        self.swarm = np.abs(hi - lo) * self.swarm + lo
        if self.bounds != None:
            self.swarm = self.bounds.Limits(self.swarm)
        return self.swarm