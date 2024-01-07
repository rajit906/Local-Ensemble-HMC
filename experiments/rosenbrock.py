'''
Experiment Code for Rosenbrock Density Function
'''

from ..hmc.hmc_numpy import *
from helper import *
import numpy as np
from mpl_toolkits import mplot3d

b = 10
f = lambda x,y: (x-1)**2 + b*(y-x**2)**2
df = lambda x,y: np.array([2*(x-1) - 4*b*(y - x**2)*x, \
                         2*b*(y-x**2)])

F = lambda X: f(X[0],X[1])
dF = lambda X: df(X[0],X[1])


def energy_func(pos, cache={}):
    '''
    Energy Function for Rosenbrock Density Function
    '''
    return f(pos[0],pos[1])

def energy_grad(pos, cache={}):
    '''
    Gradient of Energy Function (Potential Function) for Rosenbrock Density
    '''
    return df(pos[0],pos[1])

def energy_hess(pos, cache = {}):
  '''
    Hessian of Energy Function for Rosenbrock Density
  '''
  return np.array([[1200*pos[0]**2-400*pos[1]+2, -400*pos[0]],[-400*pos[0],200]])

n_sample, n_step, dt, pos_initial = 10000, 10, 1e-2, np.ones(2) + np.array([1., 1.])
N_walkers, N_steps, t = 10, 5, 1e-2
mom_resample_coeff = 1.
dtype = np.float64
sampler = LocalEnsembleHmcSampler(
    energy_func=energy_func,
    mass_matrix=np.eye(2),
    energy_grad=energy_grad,
    #energy_hess=energy_hess,
    N_walkers = N_walkers,
    N_steps = N_steps,
    tau = t,
    mom_resample_coeff=mom_resample_coeff,
    dtype=dtype)
pos,mom,acc = sampler.get_samples(pos_initial,dt,n_step,n_sample)

fig, axes = plt.subplots(nrows = 2, ncols=2, figsize=(20, 4))
xlist = np.linspace(pos[:,0].min(), pos[:,0].max(), 100)
ylist = np.linspace(pos[:,1].min(), pos[:,1].max(), 100)
X, Y = np.meshgrid(xlist, ylist)
Sgrid = np.asarray([X.flatten(),Y.flatten()]).T
Z = np.zeros([Sgrid.shape[0]])
for i in range(len(Z)):
  Z[i] = energy_func(Sgrid[i,:])
cp = axes[0][0].contourf(X, Y, Z.reshape(X.shape))
fig.colorbar(cp) # Add a colorbar to a plot"
colors = cm.rainbow(np.linspace(0, 1, pos.shape[0]))
axes[0][0].scatter(*pos.T, color=colors, s=1)
axes[0][0].set_aspect('equal')
axes[0][0].grid()
def l2(a):
    return np.linalg.norm(a - np.ones_like(a))
l2_norm = np.apply_along_axis(l2,1,pos)
#cum_avg = np.cumsum(l2_norm)/np.arange(1,n_sample+1)
axes[1][0].plot(np.arange(1,n_sample+1), l2_norm)
axes[1][0].set_xlabel(r'Iterations')
axes[1][0].set_ylabel(r'Running Error from Minimum')
IAC,G = tau(pos[:,0])
axes[0][1].plot(np.arange(len(G)),G)
IAC,G = tau(pos[:,1])
axes[1][1].plot(np.arange(len(G)),G)
fig.suptitle("stepsize = "+str(dt) + ", Leapfrog Steps = "+str(n_step) + ", No. of Samples = " + str(n_sample) + ", Acceptance Ratio = " + str(acc), fontsize='xx-large')
if hasattr(sampler, 'N_walkers'):
  plt.title("No. of Walkers = "+str(N_walkers) + ", No. of Steps = "+str(N_steps) + ", Walker stepsize = " + str(t), fontsize='xx-large', x = 0, y = -0.5)
