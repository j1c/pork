import numpy as np


def is_pos_semidefinite(arr: np.ndarray, eps: float = 1e-8) -> bool:
    """
    Check if a matrix is positive definite.

    Parameters
    ----------
    arr : np.ndarray
        The array to check. Assumes square, symmetric matrix.
    eps : float
        The tolerance for checking positive semi-definiteness.

    Returns
    -------
    res : bool
        True if the matrix is positive definite, False otherwise.
    """
    eigvals = np.linalg.eigvalsh(arr)
    eigvals[np.abs(eigvals) < eps] = 0
    res = np.all(eigvals >= 0)
    return res


def is_square(arr: np.ndarray) -> bool:
    """
    Check if a matrix is square.

    Parameters
    ----------
    arr : np.ndarray
        The matrix to check.

    Returns
    -------
    res : bool
        True if the matrix is square, False otherwise.
    """
    res = arr.shape[0] == arr.shape[1]
    return res


def is_symmetric(arr: np.ndarray) -> bool:
    """
    Check if a matrix is symmetric.

    Parameters
    ----------
    arr : np.ndarray
        The matrix to check.

    Returns
    -------
    res : bool
        True if the matrix is symmetric, False otherwise.
    """
    res = np.allclose(arr, arr.T)
    return res


def compute_determinant(arr: np.ndarray) -> float:
    """
    Compute the determinant of a matrix.

    Parameters
    ----------
    arr : np.ndarray
        The matrix to compute the determinant for.

    Returns
    -------
    det : float
        The determinant of the matrix.
    """
    det = np.linalg.det(arr)
    return det
