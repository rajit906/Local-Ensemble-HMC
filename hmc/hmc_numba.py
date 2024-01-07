import numba
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from tqdm import tqdm

def hmc(q0, n_samples, epsilon, L, M_inv, use_k=True):
    """
    ----------
    q0 : float
    n_samples : int
        Number of samples to draw
    epsilon : float
        Step size
    L : int
        Number of steps
    M_inv: np.array
        Inverse Mass Matrix
    use_k : bool
        True if we do RHMC. False if we do fixed time steps.
    Returns
    -------
    q_samples : np.array
        Array of shape (n_samples,). Samples for x
    accept_list : np.array
        Array of shape (n_samples,). Records whether we accept or reject at each step
    """

    @numba.jit(nopython=True)
    def potential(q):
        potential = 0.5 * np.dot(q,np.dot(sigma_inv,q))
        return potential

    @numba.jit(nopython=True)
    def grad_potential(q):
        grad_potential = np.dot(sigma_inv,q)
        return grad_potential

    @numba.jit(nopython=True)
    def leapfrog_step(q, p, epsilon, M_inv):
        pp = np.copy(p)
        qq = np.copy(q)
        pp = pp-(0.5 * epsilon * grad_potential(qq))
        qq = qq+(epsilon * np.dot(M_inv, pp))
        pp = pp-(0.5 * epsilon * grad_potential(qq))
        return qq, pp

    def update_inv_mass_matrix(N_w, N_s, tau, q, p, M_inv_sqrt):
        q_m, p_m = np.copy(q), np.copy(p)
        d = q_m.shape[0]
        data = np.array([q_m])
        for w in range(N_w):
          q_w = q_m
          for s in range(N_s):
            q_w += -tau * np.dot(np.eye(d),grad_potential(q_w)) + np.sqrt(2 * tau) * np.random.randn(d)
            data = np.append(data,q_w)
        data = data.reshape((-1,d))
        data -= data.mean(axis=0)
        cov = 1/(N_w*N_s) * data.T @ data
        return cov #+ lam * np.eye(cov.shape[0])

    #@numba.jit(nopython=True)
    def hmc_step(q0, epsilon, L, M_inv):
        d=q0.size
        # Resample momentum
        M = np.linalg.inv(M_inv)
        M_sqrt = np.linalg.cholesky(M)
        M_inv_sqrt = np.linalg.cholesky(M_inv)
        p0 = np.dot(M_sqrt, np.random.randn(d))
        # Initialize q, delta_U
        q = np.copy(q0)
        p = np.copy(p0)
        # Take L steps
        for ii in range(L):
            q, p = leapfrog_step(q=q, p=p, epsilon=epsilon, M_inv=M_inv)
        # Accept or reject
        current_E = potential(q0) + 0.5 * np.dot(p0, np.dot(M_inv,p0)) #old mass matrix
        M_inv = update_inv_mass_matrix(5, 3, epsilon/5, q, p, M_inv_sqrt)
        proposed_E = potential(q) + 0.5 * np.dot(p, np.dot(M_inv,p)) #new mass matrix
        accept = np.log(np.random.rand()) < (current_E - proposed_E)
        if not accept:
             q = q0
        return q, p, accept, M_inv


    q = q0
    q_samples, accept_list = [], []
    for it in tqdm(range(n_samples)):
        if use_k:
          eps=epsilon*np.random.exponential()
        else:
          eps=epsilon
        q, p, accept, M_inv = hmc_step(q0=q, epsilon=eps, L=L, M_inv=M_inv)
        q_samples.append(q)
        accept_list.append(accept)

    q_samples = np.array(q_samples)
    accept_list = np.array(accept_list)
    return q_samples, accept_list