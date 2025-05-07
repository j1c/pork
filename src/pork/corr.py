import numpy as np
from scipy import stats


def _onion_method(d: int) -> np.ndarray:
    """
    Generate a random correlation matrix using the onion method.

    Parameters:
    n (int): The size of the correlation matrix.

    Returns:
    np.ndarray: A random correlation matrix of shape (n, n).
    """
    S = np.array([[1.0]])  # Start with a 1x1 matrix

    for k in range(2, d + 1):  # MATLAB uses 2:d which is inclusive
        # Sample from beta distribution with parameters (k-1)/2 and (d-k)/2
        y = stats.beta.rvs((k - 1) / 2, (d - k) / 2)
        r = np.sqrt(y)

        # Generate random vector and normalize it
        theta = np.random.randn(k - 1)
        theta = theta / np.linalg.norm(theta)

        w = r * theta

        # Compute eigendecomposition
        eigvals, U = np.linalg.eigh(S)  # Using eigh for symmetric matrices

        # Create square root of S
        E_sqrt = np.diag(np.sqrt(eigvals))
        R = U @ E_sqrt @ U.T  # R is a square root of S

        q = R @ w

        # Increase the matrix size
        # Equivalent to MATLAB's S = [S q; q' 1]
        S_new = np.zeros((k, k))
        S_new[: k - 1, : k - 1] = S
        S_new[: k - 1, k - 1] = q
        S_new[k - 1, : k - 1] = q
        S_new[k - 1, k - 1] = 1.0

        S = S_new

    return S


def vine_method(d: int, beta: float) -> np.ndarray:
    """
    Generate a random correlation matrix using the vine method.

    Samples partial correlations from a beta distribution on (-1, 1)


    Parameters
    ----------
    n : int
        The size of the correlation matrix.
    beta : float
        Parameter controlling the beta distribution for partial correlations.

    Returns
    -------
    np.ndarray: A random correlation matrix of shape (n, n).

    References
    ----------

    """
    partial = stats.beta.rvs(beta, beta, size=(d, d))
    partial = (partial - 0.5) * 2  # Scale to (-1, 1)
    corr = np.identity(d)

    for i in range(d - 1):
        for j in range(i + 1, d):
            p = partial[i, j]

            for k in range(i - 1, -1, -1):
                p = p * np.sqrt((1 - partial[k, i] ** 2) * (1 - partial[k, j] ** 2)) + partial[k, i] * partial[k, j]

            corr[i, j] = p
            corr[j, i] = p

    perm_idx = np.random.permutation(d)
    corr = corr[perm_idx, :][:, perm_idx]

    return corr


def random_factors_method(d: int, k: int) -> np.ndarray:
    """
    Generate a random correlation matrix using the random factors method.


    Parameters
    ----------
    d : int
        The size of the correlation matrix.
    k : int
        The number of factors.

    Returns
    -------
    corr : np.ndarray, shape (d, d)
        A random correlation matrix of shape (d, d).
    """
    if k >= d:
        raise ValueError("Number of factors must be less than the size of the matrix.")

    W = np.random.randn(d, k)
    D = np.diag(np.random.rand(d))
    corr = W @ W.T + D

    normalizer = np.diag(1 / np.sqrt(np.diag(corr)))
    corr = normalizer @ corr @ normalizer

    return corr
