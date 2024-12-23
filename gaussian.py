import numpy as np
from scipy.stats import multivariate_normal, _multivariate
from scipy.linalg import sqrtm


def brownian_motion(M: int, N: int, d: int, T: float) -> np.ndarray:
    """
    Generate a Brownian motion.

    Parameters
    ----------
    M : int
        Number of Brownian motions.
    N : int
        Number of time steps.
    d : int
        Dimension of the Brownian motion.
    T : float
        Time horizon.

    Returns
    -------
    np.ndarray
        Brownian motion.
    """
    dt = T / N
    dW = np.sqrt(dt) * np.random.randn(M, N, d)
    W = np.cumsum(dW, axis=1)
    return W


def brownian_bridge(M: int, N: int, d: int, T: float, a: np.ndarray=None, b: np.ndarray=None) -> np.ndarray:
    """
    Generate a Brownian bridge.

    Parameters
    ----------
    M : int
        Number of Brownian bridges.
    N : int
        Number of time steps.
    d : int
        Dimension of the Brownian bridge.
    T : float
        Time horizon.

    Returns
    -------
    np.ndarray
        Brownian bridge.
    """
    W = brownian_motion(M, N, d, T)
    t = np.linspace(0, T, N)
    u = t[None, :, None] / T
    Z = W - u * W[:,-1,:][:,None,:]

    if (a is not None) and (b is not None): # BB from a to b
        assert a.shape == b.shape, "a and b must have the same shape"
        # a, b = np.expand_dims(a, axis=1), np.expand_dims(b, axis=1)
        Z = a[:,None,:] + (b - a)[:,None,:] * u + Z

    return Z


def solve_static_entropy_ot(nu0: _multivariate, nu1: _multivariate, sigma2: float) -> _multivariate:
    """
    Solve the static entropy-reguralized optimal transport problem between two Gaussian measures nu0 and nu1:
        pi^{s,*} = argmin_{pi coupling between nu0 and nu1} E_{pi}[||X-Y||^2] - 2*sigma^2 H(pi)
    where H is the entropy.
    
    Formula for C_s & D_s taken from paper "The Schrödinger Bridge between Gaussian Measures has a Closed Form".
    """
    assert nu0.dim == nu1.dim, "nu0 and nu1 must have same dim"
    mu0, Sigma0 = nu0.mean, nu0.cov
    mu1, Sigma1 =  nu1.mean, nu1.cov
    d = nu0.dim
    I = np.identity(n = d)
    sigma4 = sigma2**2

    Sigma0_sqrt = sqrtm(Sigma0)
    Sigma0_sqrt_inv = np.linalg.inv(Sigma0_sqrt)
    D_s = sqrtm(4 * Sigma0_sqrt @ Sigma1 @ Sigma0_sqrt + sigma4 * I)
    C_s = 0.5 * (Sigma0_sqrt @ D_s @ Sigma0_sqrt_inv - sigma2 * I)

    mu = np.concatenate([mu0, mu1])
    Sigma = np.zeros((2*d,2*d))
    Sigma[:d,:d] = Sigma0
    Sigma[:d,d:] = C_s
    Sigma[d:,:d] = C_s.T
    Sigma[d:,d:] = Sigma1

    pi_s = multivariate_normal(mean=mu, cov=Sigma)
    return pi_s


def sample_dynamic_sb(nu0: _multivariate, nu1: _multivariate, M: int, N: int, gamma: float) -> np.ndarray:
    """
    Sample M paths from the solution pi* of the dynamic Schrödinger bridge problem between two Gaussian measures nu0 and nu1 and with Brownian motion as reference measure:
        pi* = argmin_{pi_0=nu0, pi_1=nu1} KL(pi||^p)
    where p is the reference measure, defined by p = p_0 * prod_{k=0}={N-1} p_{k+1|k} where p_0 = nu0 and p_{k+1|k} = N(x_{k+1}| x_k, gamma I) such that p(x_N|x_0) = N(x_N| x_0, N*gamma I).
    """
    T = sigma2 = N * gamma
    pi_s = solve_static_entropy_ot(nu0, nu1, sigma2)
    d = nu0.dim
    x0xN = pi_s.rvs(M)
    x0, xN = x0xN[:,:d], x0xN[:,d:]
    Z = brownian_bridge(M, N, d, T, x0, xN)
    return Z

    
