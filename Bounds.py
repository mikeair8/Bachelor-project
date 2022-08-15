import numpy as np
class Bounds:
    import numpy as np
    ## The goal of the bounds class is to ensure swarm positions are set within a certain range
    def __init__(self, lower, upper, enforce='Clip'):
        ##The enforce argument is used to determine what happens if a position is out of the boundaries.
        ##Clip sets the position to be either the lower or upper limit, depending on which side of the boundaries the position is.
        ##Resample sets a random position within the boundaries.
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.enforce = enforce.lower()

    def Upper(self):
        # Upper boundaries
        return self.upper

    def Lower(self):
        # Lower boundaries
        return self.lower

    def Limits(self, pos):
        # The limits function changes the positions so they are in a certain range in case they are out of the boundaries
        npart, ndim = pos.shape

        for i in range(npart):  ##Looping through every particle
            if self.enforce == "resample":
                for j in range(ndim):  ##Looping through every dimension
                    if (pos[i,j] <= self.lower[j]) or (pos[i,j] >= self.upper[j]):
                        pos[i, j] = self.lower[j] + (self.upper[j] - self.lower[j]) * np.random.random()

            else:
                for j in range(ndim):  ## Looping through every dimension
                    ## In case of enforce=clip, the positions that are out of boundaries are set to lower/upper limit depending on what side of the boundaries the positions are.
                    if pos[i, j] <= self.lower[j]:
                        pos[i, j] = self.lower[j]
                    if pos[i, j] >= self.upper[j]:
                        pos[i, j] = self.upper[j]

        return pos

    def Validate(self, pos):
        ## The validate function simply returns the positions and can be used to validate that the positions are within the boundaries.
        return pos