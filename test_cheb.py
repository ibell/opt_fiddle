import ChebTools as CT
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

N = 50
f = lambda x: np.exp(-100*(x-0.4)**2) + 1
xmin, xmax = -1,1
x = np.linspace(xmin, xmax)
cheb = CT.generate_Chebyshev_expansion(N, f, xmin, xmax)
nodes = cheb.get_nodes_n11()
print(cheb.coef())
ytarget = cheb.y(nodes)

def obj(c):
    ex = CT.ChebyshevExpansion(c,xmin,xmax)
    yy = ex.y(nodes)
    err = yy - ytarget
    ssq = np.sum(np.power(err, 2))
    # print(ssq)
    return ssq

# c = scipy.optimize.minimize(obj, [0]*(N+1), method ='Nelder-Mead', options =dict(maxiter=100000, maxfev = 100000, disp = True))
# print(c)

# import PyALPS
# levels = PyALPS.Levels(obj, N+1, 100, 7)
# objs = []
# for i in range(1000):
#     levels.do_generation()
#     # levels.print_diagnostics()
#     o = levels.get_best()
#     objs.append([_obj for _obj, c in o])

res = scipy.optimize.differential_evolution(obj, [(-10,10)]*(N+1), disp = True, maxiter = 10000)
print(cheb.coef())
print(res.x)

y = np.array(objs)
plt.axhline(res.fun, dashes = [2,2])
plt.plot(y)
plt.yscale('log')
plt.show()

ex = CT.ChebyshevExpansion(res.x, xmin, xmax)

plt.plot(nodes, np.array(ytarget))
plt.plot(nodes, np.array(ex.y(nodes)))
plt.show()

plt.plot(nodes, (np.array(ytarget)/np.array(ex.y(nodes))-1))
plt.show()