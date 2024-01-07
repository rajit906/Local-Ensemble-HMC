'''
Experiment Code for 2-Component Gaussian Mixture Model
'''

from ..hmc.hmc import *
from helper import *
import numpy as np

d = 10
a = np.ones(d)

def energy_func(x, cache = {}):
    '''
    Energy Function for Gaussian Mixture Model
    '''
    return np.exp(-(np.sum((x - a)**2/2) - np.log(1 + np.exp(-2*x.dot(a))))) 

def energy_grad(x, cache = {}):
    '''
    Gradient of Energy Function for GMM
    '''
    return ((x - a) + 2 * np.outer(1./(1 + np.exp(2*x.dot(a))), a)).reshape(-1)


mom_resample_coeff = 1.
dtype = np.float64
N_walkers, N_steps, t = 10, 5, 0.1
n_sample, n_step, dt, pos_initial = 10000, 20, 0.1, np.random.randn(10)
sampler = LocalEnsembleHmcSampler(
    energy_func=energy_func,
    mass_matrix=np.eye(d),
    energy_grad=energy_grad,
    N_walkers = N_walkers,
    N_steps = N_steps,
    tau = t,
    mom_resample_coeff=mom_resample_coeff,
    dtype=dtype)

pos,mom,acc = sampler.get_samples(pos_initial,dt,n_step,n_sample)

truth = sample_f(a, d, n_sample)
nb_bins = 100
dir1 = a
dir1 /= np.linalg.norm(dir1)
errors = np.array([])
true_counts_1, true_bins_1, _ = plt.hist(truth.dot(dir1), bins = nb_bins, alpha = 0.5)
for s in range(n_sample):
  if not s%50:
    current_counts_1, _ = np.histogram(pos[0:s].dot(dir1), bins = true_bins_1)
    e_i = np.sum(np.abs(current_counts_1-true_counts_1))
    errors = np.append(errors, e_i)

fig, axes = plt.subplots(nrows = 2, ncols=3, figsize=(20, 4))

xlist = np.linspace(pos[:,0].min(), pos[:,0].max(), nb_bins)
ylist = np.linspace(pos[:,1].min(), pos[:,1].max(), nb_bins)
X, Y = np.meshgrid(xlist, ylist)
Sgrid = np.asarray([X.flatten(),Y.flatten()]).T
Z = np.zeros([Sgrid.shape[0]])
for i in range(len(Z)):
  Z[i] = energy_func(Sgrid[i,:])
cp = axes[0][0].contourf(X, Y, Z.reshape(X.shape))
fig.colorbar(cp)
colors = cm.rainbow(np.linspace(0, 1, pos.shape[0]))
axes[0][0].scatter(*pos.T, color=colors, s=1)
axes[0][0].set_aspect('equal')
axes[0][0].grid()
axes[0][1].hist(truth @ dir1,bins=100)
axes[0][1].hist(pos @ dir1,bins=100)
axes[0][1].legend(["Truth", "Samples"])
axes[0][1].set_title("Histogram of Projections onto line (1,1)")
axes[0][2].hist(truth @ dir2,bins=100)
axes[0][2].hist(pos @ dir2,bins=100)
axes[0][2].legend(["Truth", "Samples"])
axes[0][2].set_title("Histogram of Projections onto line (1,-1)")
axes[1][0].plot(np.arange(0,n_sample/50), errors)
axes[1][0].set_xlabel(r'Iterations')
axes[1][0].set_ylabel(r'TV Error')
IAC,G = tau(pos[:,0])
axes[1][1].plot(np.arange(len(G)),G)
IAC,G = tau(pos[:,1])
axes[1][2].plot(np.arange(len(G)),G)
fig.suptitle("Stepsize = "+str(dt) + ", Leapfrog Steps = "+str(n_step) + ", No. of Samples = " + str(n_sample) + ", Acceptance Ratio = " + str(acc), fontsize='xx-large', y = 1)
if hasattr(sampler, 'N_walkers'):
  plt.title("No. of Walkers = "+str(N_walkers) + ", No. of Steps = "+str(N_steps) + ", Walker stepsize = " + str(t), fontsize='xx-large', x = -0.75, y = -0.5)