import gc

import numpy as np
import threading
import queue
import matplotlib.pyplot as plt
queue=queue.Queue()
import os
class PSO:
    def __init__(self, obj,  # the objective function (subclass Objective)
                 npart=10,  # number of particles in the swarm
                 ndim=3,  # number of dimensions in the swarm
                 max_iter=200,  # maximum number of steps
                 c1=1.49,  # cognitive parameter
                 c2=1.49,  # social parameter
                 #  best if w > 0.5*(c1+c2) - 1:
                 w=0.729,  # base velocity decay parameter
                 inertia=None,  # velocity weight decay object (None == constant)
                 bare=False,  # if True, use bare-bones update
                 bare_prob=0.5,  # probability of updating a particle's component
                 tol=None,  # tolerance (done if no done object and gbest < tol)
                 init=None,  # swarm initialization object (subclass Initializer)
                 done=None,  # custom Done object (subclass Done)
                 ring=False,  # use ring topology if True
                 neighbors=2,  # number of particle neighbors for ring, must be even
                 vbounds=None,  # velocity bounds object
                 bounds=None,
                 maxnodes=60,
                 minnodes=30,
                 runspernet=3):  # swarm bounds object

        self.obj = obj
        self.npart = npart
        self.ndim = ndim
        self.max_iter = max_iter
        self.init = init
        self.done = done
        self.vbounds = vbounds
        self.bounds = bounds
        self.tol = tol
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.bare = bare
        self.bare_prob = bare_prob
        self.inertia = inertia
        self.ring = ring
        self.neighbors = neighbors
        self.initialized = False
        self.maxnodes=maxnodes
        self.minnodes=minnodes
        self.worst_val_mapes=[]
        self.mean_val_mapes=[]
        self.best_val_mapes=[]
        self.worst_train_mapes = []
        self.mean_train_mapes = []
        self.best_train_mapes = []
        self.bestcount=0
        self.runspernet=runspernet

        if (ring) and (neighbors > npart):
            self.neighbors = npart

    def Optimize(self):
        self.Initialize()
        while (not self.Done()):
            self.Step()
        bestpos=self.gpos[-1]
        bestpos_t=self.transformpositions(np.array([bestpos]))
        self.gpos[-1]=bestpos_t[0]
        return self.gbest[-1], self.gpos[-1]  ## Returns the LATEST global best value and the LATEST global best position
    def Initialize(self):
        self.Initialized=True
        self.iterations=0 ##Starting our iterations at 0
        self.pos=self.init.InitializeSwarm() ## Initial positions
        self.vel=np.zeros((self.npart,self.ndim)) ## Velocity, initially zero
        self.xpos=self.pos.copy() ## Each particle's best positions
        self.xbest=self.Evaluate(self.pos) ## Each particle's best position output value
        self.gidx=[] ## Particle number that found a global best position
        self.gbest=[] ## The swarm's best global position output value
        self.gpos=[] ## The swarm's best global position
        self.giter=[] ##What iteration a global best was found
        self.gidx.append(np.argmin(self.xbest))## Adding the particle ID of the particle that found the best global position
        self.gbest.append(self.xbest[self.gidx[-1]])
        self.gpos.append(self.xpos[self.gidx[-1]].copy()) ## Adding best global position to gpos
        self.giter.append(0) ## Adding iteration 0 because the 0th iteration will give a global best so far
    def Step(self):
        ## The step function represents a single swarm iteration, we start by calculating W for the iteration
        if (self.inertia!=None):
           w=self.inertia.CalculateW(self.w,self.iterations,self.max_iter)
           ## If we have an inertia to schedule the W value over the iterations, it's calculated every iteration
        else:
           w=self.w
        if self.bare:
           ## In the case of barebones PSO
           self.pos=self.BareBonesUpdate()
        else: ## If not barebones PSO, it's canonical PSO
            for i in range(self.npart): ## Looping through every particle
                lbest,lpos=self.NeighborhoodBest(i)
                c1=self.c1*np.random.random(self.ndim) ## Cognical/individual constant, multiplied by a random number in the range [0,1)
                c2=self.c2*np.random.random(self.ndim) ## Social constant, multiplied by a random number in the range [0,1)
                self.vel[i]=w*self.vel[i]+ c1*(self.xpos[i]-self.pos[i])+c2*(lpos-self.pos[i]) ## Calculating the velocity/position update for each particle

            if (self.vbounds!=None):## In case of velocity bounds, we change the velocity to be within the velocity bounds
                self.vel=self.vbounds.Limits(self.vel)
            self.pos=self.pos+self.vel ## Updating the particle positions by adding the velocity/position update vector to each position
        if (self.bounds!=None): ## In the case of positional bounds, we set the positions to be within the bounds
            self.pos=self.bounds.Limits(self.pos)
        print("ITERATION:", self.iterations)
        p=self.Evaluate(self.pos) ## Calculating the objective function values at all the positions
        self.plotworst(self.worst_val_mapes,self.worst_train_mapes)
        self.plotmean(self.mean_val_mapes,self.mean_train_mapes)
        self.plotbest(self.best_val_mapes,self.best_train_mapes)
        for i in range(self.npart): ##Looping through all the particles
            if (p[i]<self.xbest[i]):## Checking if the current objective function values of the particles are better than their own personal best objective function values
               self.xbest[i]=p[i]
               self.xpos[i]=self.pos[i]
            if (p[i]<self.gbest[-1]): ## Checking if any of the current objective function values are better than the current global best, if so:
               self.bestcount+=1
               self.gbest.append(p[i]) ## Update the global best objective function value
               self.models[i].save('BESTMODELPSO.h5')
               img_name="image"+str(i)+".png"
               new_img_name="bestmodelimage"+str(self.bestcount)+".png"
               os.rename(img_name,new_img_name)
               print("made"+new_img_name)
               self.gpos.append(self.pos[i].copy()) ## Update the global best position
               self.gidx.append(i) ## Update the global best particle ID
               self.giter.append(self.iterations) ## Update the global best position found iteration
        del self.models
        print("Best MAE so far",self.gbest[-1])
        bestpos=self.gpos[-1]
        bestpos_t = self.transformpositions(np.array([bestpos]))
        print("Best architecture so far:",bestpos_t[0])
        print("Mean MAE this iteration:",np.mean(p))
        print("Best MAE updates",self.gbest)
        print("######################################################")
        del p
        gc.collect()
        print("Giter:",self.giter)
        self.iterations+=1 ## Increase the iterator
    def Done(self):
        if (self.done == None): ## Runs code below if we are not done
            if (self.tol == None): ## Checks if we dont have a tolerance
                 return (self.iterations == self.max_iter) ## Return true if we are at the last iteration
            else:
                 return (self.gbest[-1] < self.tol) or  (self.iterations == self.max_iter)## Return true if we have reached a value below the global best
        else:
           return self.done.Done(self.gbest,
                          gpos=self.gpos,
                          pos=self.pos,
                          max_iter=self.max_iter,
                          iteration=self.iterations)

    def NeighborhoodBest(self, n):
        if (not self.ring):
            return self.gbest[-1], self.gpos[-1]

        lbest = 1e9
        for i in self.RingNeighborhood(n):
            if (self.xbest[i] < lbest):
                lbest = self.xbest[i]
        lpos = self.xpos[i]
        return lbest, lpos

    def RingNeighborhood(self, n):
        idx = np.array(range(n - self.neighbors // 2, n + self.neighbors // 2 + 1))
        i = np.where(idx >= self.npart)
        if (len(i) != 0):
            idx[i] = idx[i] % self.npart
        i = np.where(idx < 0)
        if (len(i) != 0):
            idx[i] = self.npart + idx[i]
        return idx
    def BareBonesUpdate(self):
        pos=np.zeros((self.npart,self.ndim))## Setting positions to 0 initially
        for i in range(self.npart):## Looping through every particle
            lbest,lpos=self.NeighborhoodBest(i)## Pulling out the neighborhood best objective function value and its position
            for j in range(self.ndim):## Looping through every dimension
                if (np.random.random()<self.bare_prob):## Generating a random number in the range [0,1) and
                    # checking if it's under bare_prob=0.5 in this case,
                    # so 50% chance of the code below being run
                   m=0.5*(lpos[j]+self.xpos[i,j])# Calculating the mean position between the neighborhood best and the personal bests of the particles in the neighborhood
                   s=np.abs(lpos[j]-self.xpos[i,j])## Calculating the mean distance between the neighborhood best and the personal bests of the particles in the neighborhood
                   pos[i,j]=np.random.normal(m,s)## Creates a normal distribution curve and picks a random value from it, with positions closer to the mean having a higher probability
                else:
                   pos[i,j]=self.xpos[i,j]## If the condition of the if statement above is not met, the positions are simply updated to their personal bests
        return pos## Return the positions
    def Evaluate(self,pos):
        p=np.zeros((self.npart))## Creating an array of zeros for each particle
        p2=np.zeros((self.npart))
        self.models=np.zeros((self.npart))
        self.images=np.zeros((self.npart))
        self.models=list(self.models)
        self.images=list(self.images)
        threads=[]
        particle_mapes=[]
        thread_response=[]
        t_pos=self.transformpositions(pos)
        for j in range(self.runspernet):
            for i in range(20):## Looping through every particle
                t=threading.Thread(target=self.obj.Evaluate,args=[t_pos[i],queue,i])## Calculating the objection function values at the positions of the particles and putting them in p
                threads.append(t)
            for t in threads:
                t.start()
            for i in range(len(threads)):
                threads[i].join()
                mapes=queue.get()
                particle_mapes.append(mapes)
            threads = []
        new_particle_mapes=[]
        for i in range(self.npart):
            individual_vals=[]
            individual_mapes=[]
            for j in range(len(particle_mapes)):
                mapes=particle_mapes[j]
                if i==mapes[2]:
                   individual_vals.append(mapes[0])
                   individual_mapes.append(mapes)
            k=np.argmin(individual_vals)
            new_particle_mapes.append(individual_mapes[k])
        del particle_mapes
        particle_mapes=new_particle_mapes
        for mapes in particle_mapes:
            p[mapes[2]]=mapes[0]
            p2[mapes[2]] = mapes[1]
            self.models[mapes[2]]=mapes[3]
            history=mapes[4]
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model MAE')
            plt.ylabel('MAE')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            imgname = "image" + str(mapes[2])
            plt.savefig(imgname)
            plt.close()
            del history
        del threads
        gc.collect()
        self.worst_val_mapes.append(np.max(p))
        self.mean_val_mapes.append(np.mean(p))
        self.best_val_mapes.append(np.min(p))
        self.worst_train_mapes.append(np.max(p2))
        self.mean_train_mapes.append(np.mean(p2))
        self.best_train_mapes.append(np.min(p2))
        return p## Return the objection function values for every particle
    def transformpositions(self,pos):
        node_intervals=np.linspace(0,1,self.maxnodes-self.minnodes+2)
        transformed_pos=np.zeros((pos.shape))
        for i in range(len(pos)):
            for j in range(len(pos[i])):
                for k in range(len(node_intervals) - 1):
                    if pos[i,j] >= node_intervals[k] and pos[i,j] < node_intervals[k + 1]:
                        transformed_pos[i,j]=self.minnodes+k
                    if pos[i,j] == 0:
                        transformed_pos[i,j]=self.minnodes
                        break
                    if pos[i,j] == 1:
                        transformed_pos[i,j]=self.maxnodes
        return transformed_pos

    def plotworst(self,val_mapes, train_mapes):
        iterations = np.linspace(0, len(val_mapes)-1, len(train_mapes))
        plt.plot(iterations, val_mapes, 'g')
        plt.plot(iterations, train_mapes, 'r')
        plt.legend(['Validation MAE', 'Train MAE'])
        plt.xlabel('Generation')
        plt.xticks(np.arange(0, len(iterations), 5))
        plt.title("Worst swarm MAES over generations")
        plt.savefig("figPSO1.png")
        plt.close()

    def plotmean(self,val_mapes, train_mapes):
        iterations = np.linspace(0, len(val_mapes)-1, len(train_mapes))
        plt.plot(iterations, val_mapes, 'g')
        plt.plot(iterations, train_mapes, 'r')
        plt.legend(['Validation MAE', 'Train MAE'])
        plt.xlabel('Generation')
        plt.xticks(np.arange(0, len(iterations), 5))
        plt.title("Mean swarm MAES over generations")
        plt.savefig("figPSO2.png")
        plt.close()

    def plotbest(self,val_mapes, train_mapes):
        iterations = np.linspace(0, len(val_mapes)-1, len(train_mapes))
        plt.plot(iterations, val_mapes, 'g')
        plt.plot(iterations, train_mapes, 'r')
        plt.legend(['Validation MAE', 'Train MAE'])
        plt.xlabel('Generation')
        plt.xticks(np.arange(0, len(iterations), 5))
        plt.title("Best swarm MAES over generations")
        plt.savefig("figPSO3.png")
        plt.close()

    def Results(self):
        ## The results function is used to return information about a completed search
        if (not self.Initialized): ## In the case of calling the function without starting a search, nothing is returned
           return None
        return{
            ## In the case of a completed search, we return the following information:
            "npart": self.npart, #n particles in the search
            "ndim": self.ndim, #n dimensions in the search
            "max_iter": self.max_iter,#max iterations in the search
            "iterations": self.iterations,#iterations the search reached
            "tol": self.tol,# Tolerance value, which can be used to stop the search before reaching max iterations
            "gbest": self.gbest,# Global best objective function value
            "giter": self.giter, # Iterations where global best values were update
            "gpos": self.gpos, # Position of the global best value
            "gidx": self.gidx, # ID's of the particles that updated global bests
            "pos": self.pos, # The final positions of the particles
            "w":self.w, #The overall velocity affecting parameter
            "c1": self.c1,# Return the c1/ cognitive parameter value
            "c2":self.c2, # Return the c2/ social parameter value
            "worst_val_mapes":self.worst_val_mapes,
            "mean_val_mapes":self.mean_val_mapes,
            "best_val_mapes":self.best_val_mapes,
            "worst_train_mapes": self.worst_train_mapes,
            "mean_train_mapes": self.mean_train_mapes,
            "best_train_mapes": self.best_train_mapes,
        }
