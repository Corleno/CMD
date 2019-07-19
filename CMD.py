"""
Correlation Matrix Decomposition
"""

import numpy as np


def R2pi(x):
    """
    convert R to (0,pi)
    :param x: array
    :return: array
    """
    return 1/(1+np.exp(-x))*np.pi


def theta_vec2Theta(theta_vec, N):
    """
    convert theta vector to Theta upper triangle matrix
    :param theta_vec: 1D array with length N(N+1)/2
    :param N: scalar
    :return: 2D array with dim N+1, N
    """
    if N*(N+1)/2 != len(theta_vec):
        raise ValueError("Dimension size is wrong!")
    mat = np.zeros([N+1, N])
    tril_indices = np.tril_indices(N)
    mat[tril_indices[0]+1, tril_indices[1]] = theta_vec
    return mat


def Theta2R(Theta):
    """
    convert upper triangle matrix Theta to correlation matrix R.
    :param Theta: upper triangle matrix with size N+1, N
    :return: correlation matrix with size N+1 by N+1
    """
    N = Theta.shape[0]-1
    X = np.zeros([N+1, N+1])
    S = np.sin(Theta)
    C = np.cos(Theta)

    for i in range(N+1):
        for j in range(i+1):
            if i == j:
                if j == 0:
                    X[j, j] = 1
                else:
                    X[j, j] = np.exp(np.log(S[j, :j]).sum())
            else:
                X[i, j] = C[i, j]*np.exp(np.log(S[i, :j]).sum())
    return np.matmul(X, X.T)


def vec2R(vec, N):
    """
    convert vector to correaltion matrix
    :param vec: 1d array with length N(N-1)/2
    :param N: scalar
    :return: 2D array with size N, N
    """
    if len(vec) != N*(N-1)/2:
        raise ValueError("Please check dimensional size!")
    theta_vec = R2pi(vec)
    Theta = theta_vec2Theta(theta_vec, N-1)
    return Theta2R(Theta)

if __name__ == "__main__":
    vec = np.random.randn(6)
    theta_vec = R2pi(vec)
    # print(theta_vec)
    Theta = theta_vec2Theta(theta_vec, 3)
    # print(Theta)
    R = Theta2R(Theta)
    print(R)
    print(vec2R(vec, 4))
