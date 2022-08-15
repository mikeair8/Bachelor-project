import gc

import numpy as np
import threading
import queue
import matplotlib.pyplot as plt
queue=queue.Queue()
import os
class DE:
    def __init__(self,
                 obj,
                 npart=10,
                 ndim=2,
                 max_iter=200,
                 tol=None,
                 init=None,
                 done=None,
                 bounds=None,
                 CR=0.5,
                 F=0.8,
                 mode='rand',
                 cmode='bin',
                 maxnodes=60,
                 minnodes=30
                 ):
        self.obj=obj ## The objective function
        self.npart=npart ## Number of particles
        self.ndim=ndim ## Number of input dimensions
        self.max_iter=max_iter ## Max number of iterations
        self.tol=tol ## Tolerance value
        self.init=init ## Initializer type
        self.done=done ## Done object
        self.bounds=bounds ## Lower and upper bounds for the input dimensions
        self.CR=CR ## Crossover probability
        self.F=F ## Mutation probability
        self.mode=mode ## Whether v1 in the mutation should be selected randomly or best individial (rand/best)
        self.cmode=cmode ## Crossover mode, whether to crossover like GA or coin flipping(random)
        self.tmode=False
        self.Initialized=False
        self.maxnodes = maxnodes
        self.minnodes = minnodes
        self.worst_val_mapes = []
        self.mean_val_mapes = []
        self.best_val_mapes = []
        self.worst_train_mapes = []
        self.mean_train_mapes = []
        self.best_train_mapes = []
        self.bestcount = 0

    def Step(self):
        new_pos=self.CandidatePositions() ## Pulling out new positions which we will move to in case they are better
        print("ITERATION:", self.iterations)
        p=self.Evaluate(new_pos) ## Calculating the objective function values of the candidate positions
        self.plotworst(self.worst_val_mapes, self.worst_train_mapes)
        self.plotmean(self.mean_val_mapes, self.mean_train_mapes)
        self.plotbest(self.best_val_mapes, self.best_train_mapes)
        for i in range(self.npart): ## Looping through every particle
            if p[i]<self.vpos[i]: ## Checking whether the candidate position of a particle is better than the current position
               self.vpos[i]=p[i] ## In the case of the new candidate position's objective function value being better, we update the current objective function value to the candidate position's objective function value
               self.pos[i]=new_pos[i]
            if p[i]<self.gbest[-1]: ## Checking whether the candidate objective function value is better than the global best
               self.bestcount += 1
               self.gbest.append(p[i])## In case it's better, we update the global best objective function value
               self.models[i].save('BESTMODELDE.h5')
               img_name = "image" + str(i) + ".png"
               new_img_name = "bestmodelimage" + str(self.bestcount) + ".png"
               os.rename(img_name, new_img_name)
               print("made" + new_img_name)
               self.gpos.append(new_pos[i].copy())## In case it's better, we update the global best position
               self.gidx.append(i) ## In case it's better, We update the index ID of the particle that found a new global best position
               self.giter.append(self.iterations) ## In case it's better, we update the iteration where a global best has been found
        print("Best MAE so far", self.gbest[-1])
        bestpos = self.gpos[-1]
        bestpos_t = self.transformpositions(np.array([bestpos]))
        print("Best architecture so far:", bestpos_t[0])
        print("Mean MAE this iteration:", np.mean(p))
        print("Best MAE updates:",self.gbest)
        print("######################################################")
        del p
        gc.collect()
        print("Giter:", self.giter)
        self.iterations+=1 ## Increase the iteration

    def CandidatePositions(self):
        pos=np.zeros((self.npart,self.ndim)) ## Creating a container for the new positions
        for i in range(self.npart): ## Looping through every particle
            pos[i]=self.Candidate(i) ## Generating a new position
        if self.bounds!=None: ## Checking whether we have bounds
           pos=self.bounds.Limits(pos) ## In the case of bounds, we update the positions to be within the bounds
        return pos ## Returning the positions

    def Candidate(self,idx):
        k=np.argsort(np.random.random(self.npart)) ## Generating npart random numbers between [0,1) and providing the indexes that would sort the array
        while(idx in k[:3]): ## Checking whether the current particle is one of the first three in k
             k=np.argsort(np.random.random(self.npart)) ## If it is, we resample the order to make sure the mutation vector for particle idx is made from three other particles
        v1=self.pos[k[0]] ## Choosing our v1 vector
        v2=self.pos[k[1]] ## Choosing our v2 vector
        v3=self.pos[k[2]] ## Choosing our v3 vector
        if self.mode=='best': ## In case we want to choose v1 as the best position
           v1=self.gpos[-1] ## We choose v1 to be the best position
        elif self.mode=='Toggle': ## In case we want to switch between random and best by every iteration
            if self.tmode:
               self.tmode=False
               v1=self.gpos[-1]
            else:
               self.tmode=True
        v=v1+self.F*(v2-v3) ## Generating the mutation vector from v1, F, v2 and v3
        u=np.zeros(self.ndim) ## Generating a container for all the candidate positions
        I=np.random.randint(0,self.ndim-1) ## Choosing a dimension that will be mutated
        if self.cmode=='bin': ## Checking if we use the binomial mutation method
           for j in range(self.ndim): ## Looping through every dimension of the particle
               if np.random.random()<=self.CR or j==I: ## Checking if we should mutate the particle in dimension j
                  u[j]=v[j] ## Mutation
               elif j!=I:
                  u[j]=self.pos[idx,j]
        else:
            u=self.pos[idx].copy()
            u[I:]=v[I:] ## Mutate the GA method
        return u

    def Optimize(self):
        self.Initialize()
        while (not self.Done()):
            self.Step()
        bestpos=self.gpos[-1]
        bestpos_t=self.transformpositions(np.array([bestpos]))
        self.gpos[-1]=bestpos_t[0]
        return self.gbest[-1], self.gpos[-1]  ## Returns the LATEST global best value and the LATEST global best position

    def Results(self):
        """Return the current results"""

        if (not self.initialized):
            return None

        return {
            "npart": self.npart,  # number of particles
            "ndim": self.ndim,  # number of dimensions
            "max_iter": self.max_iter,  # maximum possible iterations
            "iterations": self.iterations,  # iterations actually performed
            "tol": self.tol,  # tolerance value, if any
            "gbest": self.gbest,  # sequence of global best function values
            "giter": self.giter,  # iterations when global best updates happened
            "gpos": self.gpos,  # global best positions
            "gidx": self.gidx,  # particle number for new global best
            "pos": self.pos,  # current particle positions
            "vpos": self.vpos,  # and objective function values
        }

    def Done(self):
        if (self.done == None): ## Runs code below if we are not done
            if (self.tol == None): ## Checks if we dont have a tolerance
                 return (self.iterations == self.max_iter) ## Return true if we are at the last iteration
            else:
                 return (self.gbest[-1] < self.tol) or (self.iterations == self.max_iter)## Return true if we have reached a value below the global best
        else:
           return self.done.Done(self.gbest,
                          gpos=self.gpos,
                          pos=self.pos,
                          max_iter=self.max_iter,
                          iteration=self.iterations)

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

    def Initialize(self):
        """Set up the swarm"""

        self.initialized = True
        self.iterations = 0

        self.pos = self.init.InitializeSwarm()  # initial swarm positions
        self.vpos = self.Evaluate(self.pos)  # and objective function values

        #  Swarm bests
        self.gidx = [] ## Swarm ID's that found best positions
        self.gbest = [] ## Global best objection function values
        self.gpos = [] ## Global best positions
        self.giter = [] ## Iterations where global best positions were found
        self.gidx.append(np.argmin(self.vpos))
        self.gbest.append(self.vpos[self.gidx[-1]])
        self.gpos.append(self.pos[self.gidx[-1]].copy())
        self.giter.append(0)
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
        plt.savefig("figDE1.png")
        plt.close()

    def plotmean(self,val_mapes, train_mapes):
        iterations = np.linspace(0, len(val_mapes)-1, len(train_mapes))
        plt.plot(iterations, val_mapes, 'g')
        plt.plot(iterations, train_mapes, 'r')
        plt.legend(['Validation MAE', 'Train MAE'])
        plt.xlabel('Generation')
        plt.xticks(np.arange(0, len(iterations), 5))
        plt.title("Mean swarm MAES over generations")
        plt.savefig("figDE2.png")
        plt.close()

    def plotbest(self,val_mapes, train_mapes):
        iterations = np.linspace(0, len(val_mapes)-1, len(train_mapes))
        plt.plot(iterations, val_mapes, 'g')
        plt.plot(iterations, train_mapes, 'r')
        plt.legend(['Validation MAE', 'Train MAE'])
        plt.xlabel('Generation')
        plt.xticks(np.arange(0, len(iterations), 5))
        plt.title("Best swarm MAES over generations")
        plt.savefig("figDE3.png")
        plt.close()
