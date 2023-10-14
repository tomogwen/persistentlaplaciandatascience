
# Persistent Laplacian - perslap.py
# (C) 2023 - T Davies, Z Wan, R Sanchez-Garcia
# Made available under the MIT license

# Imports
import numpy as np
import pandas as pd
from itertools import combinations
from ismember import ismember
import random
from datetime import datetime

# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.datasets import fetch_openml
# from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt

# import dionysus as dion
from gtda.images import Binarizer, RadialFiltration, HeightFiltration
from gtda.homology import CubicalPersistence
from utils import cubical_complex, min_nonzero


def boundary(SC, k, sort=False, complex_type='simplicial'):
    # Boundary - compute the boundary of a simplicial complex sc at dim k>=0
    if complex_type == 'cubical':
        assert (k < 3), 'Requested Laplacian of dim >2, can only compute up to 2.'

    # init boundary matrix
    n_k = len(SC[k])

    if k == 0:
        return np.zeros((n_k, n_k))
    if n_k == 0:
        n_km1 = len(SC[k-1])
        return np.zeros((n_km1, n_km1))

    Sk = SC[k]      # k-simplices
    Skm1 = SC[k-1]  # (k-1)-simplices

    if sort:
        Sk = np.sort(Sk, axis=1)
        Skm1 = np.sort(Skm1, axis=1)

    n_km1 = len(SC[k - 1])
    B = np.zeros((n_km1, n_k))

    # compute boundary matrix
    # for each column in Sk

    if complex_type == 'cubical':

        num_nodes = [1, 2, 4, 8]
        num_remove = [0, 1, 2, 4]
        correct_error_count = [0, 0, 2]

    elif complex_type == 'simplicial':
        num_nodes = [1, 2, 3, 4]
        num_remove = [0, 1, 1, 1]
        correct_error_count = [0, 0, 0, 0]

    else:
        raise ValueError('Unknown complex type specified')

    error_count = np.zeros(n_k)
    sign_count = np.zeros(n_k)

    for comb in combinations(range(num_nodes[k]), num_remove[k]):
        comb = list(comb)

        # remove ith column from Sk
        remove_col_ind = list(range(num_nodes[k]))
        for c in comb:
            remove_col_ind.remove(c)

        B_aux = Sk[:, remove_col_ind]

        # find rows of B_aux in S_(k-1)
        truth_array, ind = ismember(B_aux, Skm1, 'rows')

        for j in range(n_k):
            if truth_array[j]:
                if len(ind) != len(truth_array):
                    print(Sk)
                    print(B_aux, Skm1)
                    raise ValueError('Incorrectly specified simplicial complex')

                B[ind[j], j] = (-1)**(sign_count[j]-1)
                sign_count[j] += 1
                # B[ind[j], j] += (-1)**(sign_count[j]-1)  # for delta complexes
            else:
                error_count[j] += 1

    for err in error_count:
        if err != correct_error_count[k]:
            raise ValueError('Incorrectly specified' + complex_type + 'complex')

    return B


def hodge_laplacian(SC, k):
    # computes the non-persistent Hodge Laplacian

    dim = len(SC) - 1
    n = len(SC)

    if k >= dim or k < 0 or not isinstance(k, int):
        raise ValueError("k should be an integer s.t. 0 <= k < len(SC)-1")

    # check dimensionality is correct
    else:
        B_up = boundary(SC, k+1)
        L_up = np.matmul(B_up, B_up.T)

    if k > 0:
        B_down = boundary(SC, k)
        L_down = np.matmul(B_down.T, B_down)
    else:
        L_down = np.zeros((len(SC[0]), len(SC[0])))

    L = L_up + L_down
    return L, L_up, L_down


def spectrum_hodgelap(SC, verbose=False):
    # Computes the spectrums of the non-persistent Hodge Laplacian
    L_arr = []
    spectra = []
    for i in range(len(SC)-1):
        L, L_up, L_down = hodge_laplacian(SC, i)
        L_arr.append(L)
        eigvals, eigvec = np.linalg.eig(L)
        spectra.append(eigvals)

        if verbose:
            print(str(i) +'-Hodge Laplacian')
            print(L)
            print('with spectrum:')
            print(np.round(eigvals, 2))
            print()

    return L_arr, spectra


def schur_complement(M, m, n, method='least_squares'):
    # require 0 < m < n

    if len(M.shape) != 2 and M.shape[0] != M.shape[1]:
        raise ValueError('Schur Complement error: matrix not square')

    if method == 'compute_inverse':
        inv = np.linalg.pinv(M[m:n, m:n])
        return M[0:m, 0:m] - M[0:m, m:n] @ inv @ M[m:n, 0:m]

    elif method == 'least_squares':
        # min (A*B) = A^{-1} * B
        inv_product = np.linalg.lstsq(M[m:n, m:n], M[m:n, 0:m], rcond=None)[0]
        return M[0:m, 0:m] - M[0:m, m:n] @ inv_product

    else:
        raise ValueError('Invalid Schur complement method')


def pers_lap_pair(K, L, q, complex_type='simplicial'):
    # check that the ordered basis is correct - and the same for K/L
    # the ones up to n_K_q are right
    # proper if it lines up, otherwise rearrange
    n_K_q = len(K[q])
    n_L_q = len(L[q])

    if n_K_q > n_L_q:
        raise ValueError('Your simplicial pair is not well-defined')

    B_K_q = boundary(K, q, complex_type=complex_type)
    B_L_q1 = boundary(L, q + 1, complex_type=complex_type)

    Lap_K_q_down = B_K_q.T @ B_K_q
    Lap_L_q_up = B_L_q1 @ B_L_q1.T

    if n_K_q == n_L_q:
        return Lap_L_q_up + Lap_K_q_down

    else:
        Lap_KL_q_up = schur_complement(Lap_L_q_up, n_K_q, n_L_q)
        return Lap_KL_q_up + Lap_K_q_down


def compute_pers_lap_pair(K, L, verbose=False, sort=True, complex_type='simplicial'):

    if sort:
        for i, SK in enumerate(K):
            if SK != []:
                K[i] = np.sort(SK, axis=1)
        for i, SL in enumerate(L):
            if SL != []:
                L[i] = np.sort(SL, axis=1)

    spectra = []
    PL = []

    for i in range(np.min([len(K), len(L)])-1):
        pers_lap = pers_lap_pair(K, L, i, complex_type=complex_type)
        PL.append(pers_lap)
        eigvals, eigvec = np.linalg.eig(pers_lap)
        spectra.append(eigvals)

        if verbose:
            print(str(i) + '-Persistent Laplacian')
            print(np.round(pers_lap, 2))
            print('with spectrum:')
            print(np.round(eigvals, 2))
            print()

    return PL, spectra


def features_perslap(filtration, resolution=5, complex_type='cubical', aggregate_function=None):

    max_val = np.max(filtration)
    min_val = np.min(filtration)
    increment = (max_val - min_val)/resolution

    vals = [min_val + i*increment for i in range(resolution)]
    vals[-1] = max_val  # undo floating point errors - ensures whole filtration isn't in filtration(vals[-1])

    if aggregate_function is not None:
        features = np.zeros(len(vals), len(vals))
    # prev_K_len, prev_L_len = None, None

    all_spectra = []
    for i, k in enumerate(vals):
        K = cubical_complex(filtration, k)
        for j, l in enumerate(vals[i:]):
            L = cubical_complex(filtration, l)
            pers_lap, spectra = compute_pers_lap_pair(K, L, verbose=False, complex_type=complex_type)
            spectra = np.real(spectra)
            all_spectra.append(spectra)
            if aggregate_function is not None:
                features[i, j] = aggregate_function(spectra)

    if aggregate_function is not None:
        return all_spectra, features, [min_val, max_val, increment]
    else:
        return all_spectra, [min_val, max_val, increment]


if __name__ == '__main__':
    # Full dataset is 60,000 train, 10,000 test
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    # train_size, test_size = 6000, 1000

    # Reshape to (n_samples, n_pixels_x, n_pixels_y)
    X = X.reshape((-1, 28, 28))
    random.seed(42)
    idx = random.sample(range(len(X)), 7000)
    X = X[idx]
    y = y[idx]

    df = pd.DataFrame(columns=['id', 'center', 'direction', 'start_val', 'end_val', 'eigs', 'label'])
