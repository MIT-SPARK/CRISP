"""
Thanks to Jingnan and Henk for this piece of code. We have used it only for verifying the correctness of our PACE implementation.
"""

import numpy as np
from scipy.sparse import csc_matrix
import scipy
import os
import sys


def get_rotation_relaxation_constraints():
    # Leading diagonal equal to 1
    A0 = csc_matrix((np.array([1]), (np.array([1]) - 1, np.array([1]) - 1)), shape=(10, 10))
    # columns unit length
    A1 = csc_matrix((
        np.array([1, -1, -1, -1]), (np.array([1, 2, 3, 4]) - 1, np.array([1, 2, 3, 4]) - 1)), shape=(10, 10))
    A2 = csc_matrix((
        np.array([1, -1, -1, -1]), (np.array([1, 5, 6, 7]) - 1, np.array([1, 5, 6, 7]) - 1)), shape=(10, 10))
    A3 = csc_matrix((
        np.array([1, -1, -1, -1]), (np.array([1, 8, 9, 10]) - 1, np.array([1, 8, 9, 10]) - 1)), shape=(10, 10))
    # columns orthogonal
    A4 = csc_matrix((
        np.array([1, 1, 1]), (np.array([2, 3, 4]) - 1, np.array([5, 6, 7]) - 1)), shape=(10, 10))
    A4 = A4 + A4.transpose()

    A5 = csc_matrix((
        np.array([1, 1, 1]), (np.array([2, 3, 4]) - 1, np.array([8, 9, 10]) - 1)), shape=(10, 10))
    A5 = A5 + A5.transpose()

    A6 = csc_matrix((
        np.array([1, 1, 1]), (np.array([5, 6, 7]) - 1, np.array([8, 9, 10]) - 1)), shape=(10, 10))
    A6 = A6 + A6.transpose()

    # Columns right-handedness
    A7 = csc_matrix((
        np.array([1, -1, -1]), (np.array([3, 4, 1]) - 1, np.array([7, 6, 8]) - 1)), shape=(10, 10))
    A7 = A7 + A7.transpose()

    A8 = csc_matrix((
        np.array([1, -1, -1]), (np.array([4, 2, 1]) - 1, np.array([5, 7, 9]) - 1)), shape=(10, 10))
    A8 = A8 + A8.transpose()

    A9 = csc_matrix((
        np.array([1, -1, -1]), (np.array([2, 1, 3]) - 1, np.array([6, 10, 5]) - 1)), shape=(10, 10))
    A9 = A9 + A9.transpose()

    A10 = csc_matrix((
        np.array([1, -1, -1]), (np.array([6, 1, 7]) - 1, np.array([10, 2, 9]) - 1)), shape=(10, 10))
    A10 = A10 + A10.transpose()

    A11 = csc_matrix((
        np.array([1, -1, -1]), (np.array([7, 5, 1]) - 1, np.array([8, 10, 3]) - 1)), shape=(10, 10))
    A11 = A11 + A11.transpose()

    A12 = csc_matrix((
        np.array([1, -1, -1]), (np.array([5, 1, 6]) - 1, np.array([9, 4, 8]) - 1)), shape=(10, 10))
    A12 = A12 + A12.transpose()

    A13 = csc_matrix((
        np.array([1, -1, -1]), (np.array([4, 3, 1]) - 1, np.array([9, 10, 5]) - 1)), shape=(10, 10))
    A13 = A13 + A13.transpose()

    A14 = csc_matrix((
        np.array([1, -1, -1]), (np.array([2, 1, 4]) - 1, np.array([10, 6, 8]) - 1)), shape=(10, 10))
    A14 = A14 + A14.transpose()

    A15 = csc_matrix((
        np.array([1, -1, -1]), (np.array([3, 2, 1]) - 1, np.array([8, 9, 7]) - 1)), shape=(10, 10))
    A15 = A15 + A15.transpose()

    A = [A0.toarray(),
         A1.toarray(),
         A2.toarray(),
         A3.toarray(),
         A4.toarray(),
         A5.toarray(),
         A6.toarray(),
         A7.toarray(),
         A8.toarray(),
         A9.toarray(),
         A10.toarray(),
         A11.toarray(),
         A12.toarray(),
         A13.toarray(),
         A14.toarray(),
         A15.toarray()]

    b = np.zeros(16)
    b[0] = 1.0

    return A, b


def get_vectorization_permutation():
    P = csc_matrix((
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]), (
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]) - 1,
            np.array([1, 4, 7, 2, 5, 8, 3, 6, 9]) - 1)), shape=(9, 9))
    P = P.toarray()
    return P


if __name__ == "__main__":
    R = random_SOd(3)
    r = np.reshape(R, (9, 1))
    rtran = np.reshape(R.T, (9, 1))

    rh = np.concatenate((np.array([[1.0]]), r))

    X = rh @ rh.T

    A, b = get_rotation_relaxation_constraints()

    # Check
    # (1) Ai's should be symmetric
    # (2) trace(Ai * X) == bi
    errors = []
    syms = []
    for i in range(len(A)):
        sym = scipy.sparse.linalg.norm(A[i] - A[i].transpose(), ord='fro')
        syms.append(sym)

        error = np.trace(A[i] @ X) - b[i]
        errors.append(error)
    errors = np.array(errors).reshape((len(A),))
    syms = np.array(syms).reshape((len(A),))
    print(syms)
    print(np.linalg.norm(errors))

    # Check (3) rtran = P * r
    P = get_vectorization_permutation()
    err_P = np.linalg.norm(rtran - P @ r, ord='fro')
    print(err_P)

# # the A matrices
# A = [np.zeros((10, 10))] * 16
# # A0
# A[0][0, 0] = 1
# # A1
# A[1][0, 0] = 1
# A[1][1, 1] = -1
# A[1][2, 2] = -1
# A[1][3, 3] = -1
# # A2
# A[2][0, 0] = 1
# A[2][4, 4] = -1
# A[2][5, 5] = -1
# A[2][6, 6] = -1
# # A3
# A[3][0, 0] = 1
# A[3][7, 7] = -1
# A[3][8, 8] = -1
# A[3][9, 9] = -1
# # A4
# A[4][1, 4] = 1
# A[4][2, 5] = 1
# A[4][3, 6] = 1
# # A5
# A[5][1, 7] = 1
# A[5][2, 8] = 1
# A[5][3, 9] = 1
# # A6
# A[6][4, 7] = 1
# A[6][5, 8] = 1
# A[6][6, 9] = 1
# # A7
# A[7][2, 6] = 1
# A[7][3, 5] = -1
# A[7][0, 7] = -1
# # A8
# A[8][3, 4] = 1
# A[8][1, 6] = -1
# A[8][0, 8] = -1
# # A9
# A[9][1, 5] = 1
# A[9][0, 9] = -1
# A[9][2, 4] = -1
# # A10
# A[10][5, 9] = 1
# A[10][0, 1] = -1
# A[10][6, 8] = -1
# # A11
# A[11][6, 7] = 1
# A[11][4, 9] = -1
# A[11][0, 2] = -1
# # A12
# A[12][4, 8] = 1
# A[12][0, 3] = -1
# A[12][5, 7] = -1
# # A13
# A[13][3, 8] = 1
# A[13][2, 9] = -1
# A[13][0, 4] = -1
# # A14
# A[14][1, 9] = 1
# A[14][0, 5] = -1
# A[14][3, 7] = -1
# # A15
# A[15][2, 7] = 1
# A[15][1, 8] = -1
# A[15][0, 6] = -1
# # make symmetrical
# A = [x + x.T for x in A]

# # the P matrix
# P = np.zeros((9, 9))
# P[0, 0] = 1
# P[1, 3] = 1
# P[2, 6] = 1
# P[3, 1] = 1
# P[4, 4] = 1
# P[5, 7] = 1
# P[6, 2] = 1
# P[7, 5] = 1
# P[8, 8] = 1