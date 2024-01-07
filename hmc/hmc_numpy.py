import numpy as np
import scipy.linalg as la
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch as th

autograd_available = True
try:
    from autograd import grad
except ImportError:
    autograd_available = False

logger = logging.getLogger(__name__)


class DynamicsError(Exception):
    """Base class for exceptions due to error in simulation of dynamics. """
    pass

class AbstractHmcSampler(object):
    """ Abstract Hamiltonian Monte Carlo sampler base class. """

    def __init__(self, energy_func, energy_grad=None, prng=None,
                 mom_resample_coeff=1., dtype=np.float64):
        self.energy_func = energy_func
        if energy_grad is None and autograd_available:
            e_grad = grad(energy_func, 0)
            # force energy gradient to ignore cached results if using Autograd
            # as otherwise gradient may be incorrectly calculated
            self.energy_grad = lambda pos, cache={}: e_grad(pos)
        elif energy_grad is None and not autograd_available:
            raise ValueError('Autograd not available therefore energy gradient'
                             ' must be provided.')
        else:
            self.energy_grad = energy_grad
        self.prng = prng if prng is not None else np.random.RandomState()
        if mom_resample_coeff < 0 or mom_resample_coeff > 1:
                raise ValueError('Momentum resampling coefficient must be in '
                                 '[0, 1]')
        self.mom_resample_coeff = mom_resample_coeff
        self.dtype = dtype

    def kinetic_energy(self, pos, mom, cache={}):
        raise NotImplementedError()

    def simulate_dynamic(self, n_step, dt, pos, mom, cache={}):
        raise NotImplementedError()

    def sample_independent_momentum_given_position(self, pos, cache={}):
        raise NotImplementedError()

    def resample_momentum(self, pos, mom, cache={}):
        if self.mom_resample_coeff == 1:
            return self.sample_independent_momentum_given_position(pos, cache)
        elif self.mom_resample_coeff == 0:
            return mom
        else:
            mom_i = self.sample_independent_momentum_given_position(pos, cache)
            return (self.mom_resample_coeff * mom_i +
                    (1. - self.mom_resample_coeff**2)**0.5 * mom)

    def hamiltonian(self, pos, mom, cache={}):
        return (self.energy_func(pos, cache) +
                self.kinetic_energy(pos, mom, cache))

    def get_samples(self, pos, dt, n_step_per_sample, n_sample, mom=None):

        n_dim = pos.shape[0]
        pos_samples, mom_samples = np.empty((2, n_sample, n_dim), self.dtype)
        cache = {}
        if mom is None:
            mom = self.sample_independent_momentum_given_position(pos, cache)
        pos_samples[0], mom_samples[0] = pos, mom

        # check if number of steps specified by tuple and if so extract
        # interval bounds and check valid
        if isinstance(n_step_per_sample, tuple):
            randomise_steps = True
            step_interval_lower, step_interval_upper = n_step_per_sample
            assert step_interval_lower < step_interval_upper
            assert step_interval_lower > 0
        else:
            randomise_steps = False

        hamiltonian_c = self.hamiltonian(pos, mom, cache)
        n_reject = 0

        for s in range(1, n_sample):
            if randomise_steps:
                n_step_per_sample = self.prng.random_integers(
                    step_interval_lower, step_interval_upper)
            # simulate Hamiltonian dynamic to get new state pair proposal
            try:
                pos_p, mom_p, cache_p = self.simulate_dynamic(
                    n_step_per_sample, dt, pos_samples[s-1],
                    mom_samples[s-1], s, cache)
                hamiltonian_p = self.hamiltonian(pos_p, mom_p, cache_p)
                proposal_successful = True
            except DynamicsError as e:
                logger.info('Error occured when simulating dynamic. '
                            'Rejecting.\n' + str(e))
                proposal_successful = False
            # Metropolis-Hastings accept step on proposed update
            if True: #(proposal_successful and self.prng.uniform() <
                    #np.exp(hamiltonian_c - hamiltonian_p)):
                # accept move
                pos_samples[s], mom_samples[s], cache = pos_p, mom_p, cache_p
                hamiltonian_c = hamiltonian_p
            else:
                # reject move
                pos_samples[s] = pos_samples[s-1]
                # negate momentum on rejection to ensure reversibility
                mom_samples[s] = -mom_samples[s-1]
                n_reject += 1
            # momentum update transition: leaves momentum conditional invariant
            mom_samples[s] = self.resample_momentum(
                pos_samples[s], mom_samples[s], cache)
            if self.mom_resample_coeff != 0:
                hamiltonian_c = self.hamiltonian(pos_samples[s],
                                                 mom_samples[s], cache)
        return pos_samples, mom_samples, 1. - (n_reject * 1. / n_sample)

class IsotropicHmcSampler(AbstractHmcSampler):
    """Standard unconstrained HMC sampler with identity mass matrix. """

    def kinetic_energy(self, pos, mom, cache={}):
        return 0.5 * mom.dot(mom)

    def simulate_dynamic(self, n_step, dt, pos, mom, s,cache={}):
        mom = mom - 0.5 * dt * self.energy_grad(pos, cache)
        pos = pos + dt * mom
        for s in range(1, n_step):
            mom -= dt * self.energy_grad(pos, cache)
            pos += dt * mom
        mom -= 0.5 * dt * self.energy_grad(pos, cache)
        return pos, mom, None

    def sample_independent_momentum_given_position(self, pos, cache={}):
        return self.prng.normal(size=pos.shape[0]).astype(self.dtype)


class EuclideanMetricHmcSampler(IsotropicHmcSampler):
    """Standard unconstrained HMC sampler with constant mass matrix. """

    def __init__(self, energy_func, mass_matrix, energy_grad=None, prng=None,
                 mom_resample_coeff=1., dtype=np.float64):
        super(EuclideanMetricHmcSampler, self).__init__(
            energy_func, energy_grad, prng, mom_resample_coeff, dtype)
        self.mass_matrix = mass_matrix
        self.mass_matrix_chol = la.cholesky(mass_matrix, lower=True)

    def kinetic_energy(self, pos, mom, cache={}):
        return 0.5 * mom.dot(la.cho_solve(
            (self.mass_matrix_chol, True), mom))

    def simulate_dynamic(self, n_step, dt, pos, mom, iter, cache={}):
        mom = mom - 0.5 * dt * self.energy_grad(pos, cache)
        pos = pos + dt * la.cho_solve((self.mass_matrix_chol, True), mom)
        for s in range(1, n_step):
            mom -= dt * self.energy_grad(pos, cache)
            pos += dt * la.cho_solve((self.mass_matrix_chol, True), mom)
        mom -= 0.5 * dt * self.energy_grad(pos, cache)
        return pos, mom, None

    def sample_independent_momentum_given_position(self, pos, cache={}):
        return (self.mass_matrix_chol.dot(self.prng.normal(size=pos.shape[0]))).astype(self.dtype)

class LocalEnsembleHmcSampler(EuclideanMetricHmcSampler):
    """Standard unconstrained HMC sampler with dynamic mass matrix estimated by a local ensemble of walkers. """

    def __init__(self, energy_func, mass_matrix, N_walkers, N_steps, tau, energy_grad=None, energy_hess = None, prng=None,
                 mom_resample_coeff=1., dtype=np.float64):
        super(EuclideanMetricHmcSampler, self).__init__(
            energy_func, energy_grad, prng, mom_resample_coeff, dtype)
        self.mass_matrix = mass_matrix
        self.mass_matrix_inv = la.cholesky(self.mass_matrix, lower=True)
        self.N_walkers = N_walkers
        self.N_steps = N_steps
        self.tau = tau
        self.dim = len(mass_matrix)
        self.mass_matrix_chol = la.cholesky(mass_matrix, lower=True)
        self.mass_matrix_inv_chol = np.linalg.inv(self.mass_matrix_chol)

    def kinetic_energy(self, pos, mom, cache={}):
        return 0.5 * mom @ self.mass_matrix_inv @ mom - np.log(np.sqrt(np.linalg.det(self.mass_matrix)))

    def simulate_walkers(self, pos, mom, lam):
        pos_initial = np.copy(pos)
        mom_initial = np.copy(mom)
        samples = np.array([])
        for i in range(self.N_walkers):
          pos_w = pos_initial
          mom_w = mom_initial
          for j in range(self.N_steps):
            #pos_w = self.tau * self.mass_matrix_inv @ mom_w
            #mom_w = -pos_w - self.tau * self.energy_grad(pos_w) + np.sqrt(2 * self.tau) * np.random.randn(pos_w.shape[0])
            pos_w += -self.tau * self.energy_grad(pos_w) + np.sqrt(2 * self.tau) * np.random.randn(pos_w.shape[0])
            samples = np.append(samples,pos_w)
        data = np.reshape(samples, (-1,self.dim))
        data -= data.mean(axis=0)
        cov = 1/(self.N_walkers*self.N_steps) * data.T @ data
        return cov + lam * np.eye(cov.shape[0])

    def simulate_dynamic(self, n_step, dt, pos, mom, iter, cache={}):
        mom = mom - 0.5 * dt * self.energy_grad(pos, cache)
        pos = pos + dt * self.mass_matrix_inv @ mom
        for s in range(1, n_step):
            mom -= dt * self.energy_grad(pos, cache)
            pos += dt * self.mass_matrix_inv @ mom
        mom -= 0.5 * dt * self.energy_grad(pos, cache)
        if not iter % 1: #We can set this to be %2 or %5 to compute the covariance every 2 or 5 steps respectively.
          self.mass_matrix_inv = self.simulate_walkers(pos, mom, lam = 0.)
          self.mass_matrix_inv_chol = la.cholesky(self.mass_matrix_inv, lower=True)
        return pos, mom, None

    def sample_independent_momentum_given_position(self, pos, cache={}):
        self.mass_matrix_chol = la.cholesky(la.inv(self.mass_matrix_inv),lower=True)
        return (self.mass_matrix_chol.dot(self.prng.normal(size=pos.shape[0]))).astype(self.dtype)