
# Persistent Laplacian - qm_experiments.py
# (C) 2023 - T Davies, Z Wan, R Sanchez-Garcia
# Made available under the MIT license

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.ticker import FuncFormatter
import seaborn as sns

from math import log as log
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import VarianceThreshold

from scipy.spatial import distance_matrix

from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical

from gtda.diagrams import HeatKernel, PersistenceEntropy, Amplitude, Scaler, NumberOfPoints, PersistenceImage
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_point_cloud, plot_diagram
import dionysus as dion

# local imports
from perslap import compute_pers_lap_pair, features_perslap


# Coloumb matrices (7165 x 23 x 23)
qm_X = np.genfromtxt('../data/qm7/qm7_X.csv', delimiter=',')
qm_X = qm_X.reshape((7165, 23, 23))

# labels (atomization energy) (7165 x 1)
qm_T = np.genfromtxt('../data/qm7/qm7_T.csv', delimiter=',')

# Cartestian coordinate of atoms (7165 x  23 x 3)
qm_R = np.genfromtxt('../data/qm7/qm7_R.csv', delimiter=',')
qm_R = qm_R.reshape((7165, 23, 3))

# Z is atomic charge (7165 x 1)
# P is cross-val splist (5 x 1433)

def dion_max(filtration):
    max_ = 0
    for simplex in filtration:
        if simplex.data > max_:
            max_ = simplex.data
    return max_


def dion_min(filtration):
    min_ = 99999999
    for simplex in filtration:
        if simplex.data < min_:
            min_ = simplex.data
    return min_


def dion_filt_to_filtration(filt):
    complex_ = [[], [], [], []]
    for simplex in filt:
        temp_simplex = [simplex.data]
        for vertex in simplex:
            temp_simplex.append(vertex)
        complex_[len(simplex)-1].append(temp_simplex)
    return complex_


def filter_cutoff(filt, cutoff):
    complex_ = [[], [], [], []]
    for k, complex_k in enumerate(filt):
        for simplex in complex_k:
            if simplex[0] < cutoff:
                complex_[k].append(simplex[1:])
    return complex_


def print_time(message):
    now = datetime.now()
    current_time = now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    print(f'{message} at {current_time}')


if __name__ == '__main__':
    df = pd.DataFrame(columns=['id', 'start_val', 'end_val', 'start_val_idx', 'end_val_idx', 'eigs', 'label'])
    for count, molecule in enumerate(qm_R):
        if count % 1 == 0:
            print_time('Starting molecule ' + str(count))

        resolution = 4
        filtration = dion.fill_rips(molecule, 3, 20)

        max_val = dion_max(filtration)
        min_val = dion_min(filtration)
        increment = (max_val - min_val) / resolution
        filtration = dion_filt_to_filtration(filtration)
        vals = [min_val + i * increment for i in range(resolution)]
        vals[-1] = max_val  # undo floating point errors - ensures whole filtration isn't in filtration(vals[-1])
        vals = vals[1:]

        print(vals)

        for i, k in enumerate(vals):
            # print_time(1)
            K = filter_cutoff(filtration, k)
            # print_time(2)
            for j, l in enumerate(vals[i:]):
                # print_time(3)
                L = filter_cutoff(filtration, l)
                # print_time(4)
                pers_lap, spectra = compute_pers_lap_pair(K, L, verbose=False, complex_type='simplicial')
                # print_time(5)
                for m, spectra_k in enumerate(spectra):
                    real_spectra_k = np.real(spectra_k)
                    spectra[m] = np.round(real_spectra_k, 3)
                # print_time(6)
                df = df.append({'id': count,
                                'start_val': k,
                                'end_val': l,
                                'start_val_idx': i,
                                'end_val_idx': j,
                                'eigs': spectra,
                                'label': qm_T[count]}, ignore_index=True)
                # print_time(7)
    df.to_pickle('../results/spectra_qm7.pkl')