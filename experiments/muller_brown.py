'''
Experiment Code for (Tempered) Muller-Brown density
'''

from ..hmc.hmc_numpy import *
from helper import *
import numpy as np

A = np.array([-200, -100, -170, 15])
a = np.array([-1, -1, -6.5, 0.7])
b = np.array([0, 0, 11, 0.6])
c = np.array([-10, -10, -6.5, 0.7])
x_m = np.array([1, 0, -0.5, -1])
y_m = np.array([0, 0.5, 1.5, 1])

x = np.linspace(-1.5, 0.7, 1000)
y = np.linspace(0, 1.9, 1000)
XX, YY = np.meshgrid(x, y)

Z = (A[0]*np.exp( a[0]*(XX-x_m[0])**2 + b[0]*(XX-x_m[0])*(YY-y_m[0]) + c[0]*(YY-y_m[0])**2 )
    +A[1]*np.exp( a[1]*(XX-x_m[1])**2 + b[1]*(XX-x_m[1])*(YY-y_m[1]) + c[1]*(YY-y_m[1])**2 )
    +A[2]*np.exp( a[2]*(XX-x_m[2])**2 + b[2]*(XX-x_m[2])*(YY-y_m[2]) + c[2]*(YY-y_m[2])**2 )
    +A[3]*np.exp( a[3]*(XX-x_m[3])**2 + b[3]*(XX-x_m[3])*(YY-y_m[3]) + c[3]*(YY-y_m[3])**2 ))

fig, ax = plt.subplots()

c=ax.contourf(XX, YY, Z)
plt.colorbar(c)
ax.set_xlabel('x')
ax.set_ylabel('y')

m1 = (-0.558223634633024, 1.441725841804669)
m2 = (-0.050010822998206, 0.466694104871972)
s1 = (-0.822001558732732, 0.624312802814871)
plt.plot(*m1, 'm*'), plt.text(*m1, "  Min 1")
plt.plot(*m2, 'm*'), plt.text(*m2, "  Min 2")
plt.plot(*s1, 'bo'), plt.text(*s1, "  Saddle 1")

import numpy as np

def energy_func(pos, cache={}):
    '''
    Energy Function for Muller-Brown
    '''
    x, y = pos[0], pos[1]
    A = np.array([-200, -100, -170, 15])
    a = np.array([-1, -1, -6.5, 0.7])
    b = np.array([0, 0, 11, 0.6])
    c = np.array([-10, -10, -6.5, 0.7])
    x0 = np.array([1, 0, -0.5, -1])
    y0 = np.array([0, 0.5, 1.5, 1])

    term1 = A * np.exp(a * (x - x0)**2 + b * (x - x0) * (y - y0) + c * (y - y0)**2)

    return 1/5 * np.sum(term1, axis=-1)

def energy_grad(pos, cache={}):
    '''
    Potential Function for Muller-Brown
    '''
    x, y = pos[0], pos[1]
    A = np.array([-200, -100, -170, 15])
    a = np.array([-1, -1, -6.5, 0.7])
    b = np.array([0, 0, 11, 0.6])
    c = np.array([-10, -10, -6.5, 0.7])
    x0 = np.array([1, 0, -0.5, -1])
    y0 = np.array([0, 0.5, 1.5, 1])
    d_dx, d_dy = 0, 0
    for i in range(len(A)):
      d_dx += A[i] * np.exp(a[i] * (x-x0[i])**2 + b[i] * (x-x0[i]) * (y-y0[i]) + c[i] * (y-y0[i])**2) * (2 * a[i] * (x-x0[i]) + b[i] * (y-y0[i]))
      d_dy += A[i] * np.exp(a[i] * (x-x0[i])**2 + b[i] * (x-x0[i]) * (y-y0[i]) + c[i] * (y-y0[i])**2) * (b[i] * (x - x0[i]) + 2 * c[i] * (y - y0[i]))
    return 1/5 * np.array([d_dx, d_dy])

energy_func(np.array([-0.1,0.1]))

mom_resample_coeff = 1.
dtype = np.float64
N_walkers, N_steps, t = 5, 3, 0.005
sampler = LocalEnsembleHmcSampler(
    energy_func=energy_func,
    mass_matrix=np.eye(2),
    energy_grad=energy_grad,
    N_walkers = N_walkers,
    N_steps = N_steps,
    tau = t,
    mom_resample_coeff=mom_resample_coeff,
    dtype=dtype)
n_sample, n_step, dt, pos = 10000, 50, 0.01, np.array([0.5, 0])
pos,mom,acc = sampler.get_samples(pos,dt,n_step,n_sample)