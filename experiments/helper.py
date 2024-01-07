'''
Helper Functions for Experiments
'''
import numpy as np

def auto_corr_fast(M):
    '''
    Computes Autocorrelation with lag (kappa)
    '''
    kappa = 200
    M = M - np.mean(M)
    N = len(M)
    fvi = np.fft.fft(M, n=2*N)
    G = np.real( np.fft.ifft( fvi * np.conjugate(fvi) )[:N] )
    G /= N - np.arange(N); G /= G[0]
    G = G[:kappa]
    return G

def tau(M):
    '''
    Computes integrated autocorrelation time
    '''
    autocorr = auto_corr_fast(M)
    return 1 + 2*np.sum(autocorr), autocorr

def sample_f(a, d, n_sample):
    '''
    Uses coin-flipping to sample from a 2-component GMM
    '''
    coin = np.random.binomial(1, 0.5, n_sample) * 2. - 1.
    return np.outer(coin, a) + np.random.randn(n_sample, d)