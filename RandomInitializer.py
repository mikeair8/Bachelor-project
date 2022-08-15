import numpy as np

class RandomInitializer:
    ## The RandomInitializer class initializes a swarm with random positions.
    def __init__(self, n_particles=10, n_dimensions=3, bounds=None):
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.bounds = bounds

    def InitializeSwarm(self):
        if self.bounds == None:
            ## In the case of no boundaries, the swarm positions are set with completely random values.
            self.swarm = np.random.random((self.n_particles, self.n_dimensions))
        else:
            self.swarm = np.zeros((self.n_particles, self.n_dimensions))
            ## In the case of boundaries, the swarm positions are set random within the boundary range.
            lo = self.bounds.Lower()
            hi = self.bounds.Upper()
            for i in range(self.n_particles):
                for j in range(self.n_dimensions):
                    self.swarm[i, j] = round(lo[j] + (hi[j] - lo[j]) * np.random.random())
            self.swarm = self.bounds.Limits(self.swarm)
        return self.swarm