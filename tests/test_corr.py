import pytest

from pork.corr import vine_method, random_factors_method
from pork.utils import is_pos_semidefinite, is_symmetric, is_square


@pytest.fixture
def corr_matrices():
    """
    Fixture to provide a set of correlation matrices for testing.
    """
    d = 5
    beta = 2.0
    k = 3

    vine_matrix = vine_method(d, beta)
    random_factors_matrix = random_factors_method(d, k)

    matrices = [
        ("vine_method", vine_matrix),
        ("random_factors_method", random_factors_matrix),
    ]

    return matrices


def test_corr_matrices_shape(corr_matrices):
    for method, matrix in corr_matrices:
        # Check if the matrix is square
        assert is_square(matrix), f"{method} matrix is not square"


def test_corr_matrices_positive_definite(corr_matrices):
    for method, matrix in corr_matrices:
        # Check if the matrix is positive semi-definite
        assert is_pos_semidefinite(matrix), (
            f"{method} matrix is not positive semi-definite"
        )


def test_corr_matrices_symmetric(corr_matrices):
    for method, matrix in corr_matrices:
        # Check if the matrix is symmetric
        assert is_symmetric(matrix), f"{method} matrix is symmetric"
