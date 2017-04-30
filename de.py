from __future__ import print_function
import random
from math import cos, pow, sqrt

class Individual(object):
    def __init__(self, v=None):
        if v is not None:
            self.v = v

class DE(object):
    def __init__(self, f, D, bounds, popsize, F = 1, CR = 0.5):
        self.f = f
        self.D = D
        self.bounds = bounds
        self.popsize = popsize
        self.F = F
        self.CR = CR
        self.population = []
    
    def _initialize(self):
        for i in range(self.popsize):
            ind = Individual([random.uniform(lb,ub) for lb,ub in self.bounds])
            self.population.append(ind)

    def get_N_unique(self, v, N):
        """ Returns indices associated with unique individuals """
        assert(N < len(v))
        indices_list = list(range(len(v)))
        # Short circuit if you want as many indices as the length
        if N == len(v):
            return indices_list
        indices = []
        for i in range(N):
            while True:
                j = random.choice(indices_list)
                if j in indices:
                    continue
                else:
                    indices.append(j)
                    break
        return indices

    def new_ind(self, orig, ind1, ind2, ind3, R):
        """ 
        Return a hybrid individual from the original and three
        other individuals
        """
        vn = [0]*len(orig.v)
        for i in range(len(vn)):
            if (i == R or random.uniform(0, 1) < self.CR):
                vn[i] = ind1.v[i] + self.F*(ind2.v[i] - ind3.v[i])
        return Individual(vn)

    def optimize(self):
        self._initialize()
        for igen in range(0, 1000):
            for j in range(len(self.population)):
                # Get a random individual and three others
                # (actually their indices)
                iorig, ia, ib, ic = self.get_N_unique(self.population, 4)

                # Get a random index
                R = random.randint(0, len(self.population))

                # Orig = 
                cand = self.new_ind(self.population[iorig], self.population[ia], 
                                    self.population[ib], self.population[ic],
                                    R)
                
                # If the original is worse than the candidate, swap
                if (self.f(self.population[iorig].v) > self.f(cand.v)):
                    self.population.remove(self.population[iorig])
                    self.population.append(cand)

            data = [(self.f(ind.v), ind) for ind in self.population]
            data.sort(key=lambda p: p[0])
            obj, self.population = zip(*data)
            self.population = list(self.population)
            if obj[0] < 1e-6:
                return
        
if __name__=='__main__':
    import matplotlib.pyplot as plt
    xx,yy = [],[]
    for D in [10, 20, 30, 40, 50, 60]:
        def Griewangk(x):
            sum1 = 0 
            prod1 = 1
            for i in range(len(x)):
                sum1 += pow(x[i], 2)
                prod1 *= cos(x[i]/sqrt(i+1))
            f = sum1/4000.0 - prod1 + 1
            return f

        class FuncCaller(object):
            def __init__(self, f):
                self.Neval = 0
                self.f = f
            def __call__(self, x):
                self.Neval += 1
                return self.f(x)

        fc = FuncCaller(Griewangk)
        de = DE(fc, D, [(-10,10)]*D, D*5)
        de.optimize()
        print(D, fc.Neval)
        xx.append(D)
        yy.append(fc.Neval)
    plt.plot(xx, yy)
    plt.show()