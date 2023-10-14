
# Persistent Laplacian - utils.py
# (C) 2023 - T Davies, Z Wan, R Sanchez-Garcia
# Made available under the MIT license

import numpy as np
import pandas as pd
from itertools import combinations
from ismember import ismember
# import random
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


def cubical_complex(filtration, filter_val, sort=True):
    # Builds cubical complexes from a filtration on an array

    # add vertices (pixels), edges (two adjacent pixels), cubes (unit cubes)
    f = np.array(filtration[0] < filter_val)
    nodes, edges, squares = [], [], []
    # add vertices
    for i, pix in enumerate(f.flatten()):
        if pix:
            nodes.append([i])

    # add edges
    for i in range(28):
        for j in range(14):
            if (i % 2) == 0:
                col = j * 2
            else:
                col = j * 2 + 1

            test_idx = [[i + 1, col], [i - 1, col], [i, col + 1], [i, col - 1]]
            for idx in test_idx:
                if f[i, col] and (0 <= idx[0] < 28) and (0 <= idx[1] < 28) and f[idx[0], idx[1]]:
                    edges.append([i * 28 + col, idx[0] * 28 + idx[1]])

    # add squares
    for i in range(0, 28, 2):
        for j in range(0, 28, 2):

            test_idx = []

            if i > 0 and j > 0:
                test_idx.append([[i, j - 1], [i - 1, j], [i - 1, j - 1]])

            if i < 27 and j > 0:
                test_idx.append([[i, j - 1], [i + 1, j], [i + 1, j - 1]])

            if i > 0 and j < 27:
                test_idx.append([[i - 1, j], [i, j + 1], [i - 1, j + 1]])

            if i < 27 and j < 27:
                test_idx.append([[i, j + 1], [i + 1, j], [i + 1, j + 1]])

            for idx in test_idx:
                # if verify_bounds(i, j, idx):
                if f[i, j] and f[idx[0][0], idx[0][1]] and f[idx[1][0], idx[1][1]] and f[idx[2][0], idx[2][1]]:
                    squares.append([i * 28 + j, idx[0][0] * 28 + idx[0][1],
                            idx[1][0] * 28 + idx[1][1], idx[2][0] * 28 + idx[2][1]])

    if sort:
        if len(nodes) != 0:
            nodes = np.sort(np.array(nodes), axis=1)
        if len(edges) != 0:
            edges = np.sort(np.array(edges), axis=1)
        else:
            return [nodes]
        if len(squares) != 0:
            squares = np.sort(np.array(squares), axis=1)
        else:
            return [nodes, edges]

    return [nodes, edges, squares]


def cubes_and_fill(filtration, filter_val):
    # Adds cubes then fills in edges and points

    f = np.array(filtration[0] < filter_val)
    nodes, edges, squares = [], [], []

    # add squares
    for i in range(0, 28, 2):
        for j in range(0, 28, 2):
            test_idx = [[[i, j - 1], [i - 1, j], [i - 1, j - 1]],
                        [[i, j - 1], [i + 1, j], [i + 1, j - 1]],
                        [[i - 1, j], [i, j + 1], [i - 1, j + 1]],
                        [[i, j + 1], [i + 1, j + 1], [i + 1, j + 1]]]

            for idx in test_idx:
                try:
                    if f[i, j] and f[idx[0][0], idx[0][1]] and f[idx[1][0], idx[1][1]] and f[idx[2][0], idx[2][1]]:
                        squares.append([i * 28 + j, idx[0][0] * 28 + idx[0][1],
                                        idx[1][0] * 28 + idx[1][1], idx[2][0] * 28 + idx[2][1]])
                    # edges.append([idx[0][0]*28+idx[0][1], idx[1][0]*28+idx[1][1]])
                except IndexError:
                    print('guess you should fix this edge case')
                    pass

    squares = np.sort(np.array(squares), axis=1)

    for square in squares:
        c = square[0]
        for edge in [[c, c+1], [c, c+28], [c+28, c+29], [c+1, c+29]]:
            edges.append(edge)

        for add in [0, 1, 28, 29]:
            nodes.append([c+add])

    nodes = np.sort(np.array(nodes), axis=1)
    edges = np.sort(np.array(edges), axis=1)

    return [nodes, edges, squares]


def fill_complex(SC, verbose=False):
    # Given the top-level simplices, add the induced lower-level simplices to the complex

    if verbose:
        print("FILLING SIMPLICIAL COMPLEX")

    for i, Sk in enumerate(SC):
        if len(Sk) > 0:
            SC[i] = np.sort(Sk, axis=1)

    # for each dimension k
    for k in range(len(SC)-1, 0, -1):
        if verbose:
            print('\nDIM ' + str(k))
            print('original SC_(k-1)')
            print(SC[k-1])
        # for each k-simplex in Sk
        for k_simplex in SC[k]:
            if verbose:
                print(str(k) + '-simplex is ' + str(k_simplex) + ', '
                      + str(k-1) + '-faces are:')
            # check that its constituent (k-1)-simplices are in S_(k-1)
            for face in combinations(k_simplex, k):
                face = list(face)
                if verbose:
                    print(face)

                # TODO: feels like the axis=(1) will break when k>2
                # comes from: https://stackoverflow.com/questions/39452843/in-operator-for-numpy-arrays
                # if face not in SC[k-1]:

                if len(SC[k-1]) < 2 or (not ((face == SC[k-1]).all(axis=(1))).any()):
                    sorted_face = sorted(face)
                    if verbose:
                        print('ADDED simplex ' + str(sorted_face) + ' to SC_' + str(k-1))
                    if len(SC[k-1]) == 0:
                        SC[k-1] = [sorted_face]
                    else:
                            SC[k-1] = np.append(SC[k-1], [sorted_face], axis=0)

        if verbose:
            print('new SC_(k-1)')
            print(SC[k-1])

    # should sort at end too but errors?
    for i, Sk in enumerate(SC):
        SC[i] = np.sort(Sk, axis=1)

    if verbose:
        print("\nDONE FILLING\n")

    return SC


def print_complex(K):
    for i, sc in enumerate(K):
        print(str(i) + '-simplices\n' + str(sc))
    print()


def min_nonzero(list_):
    # returns minimum non-zero value
    return np.min(list_[np.nonzero(list_)])


if __name__ == '__main__':
    # run tests
    exit(0)
