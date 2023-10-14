
# Persistent Laplacian - mnist_experiments.py
# (C) 2023 - T Davies, Z Wan, R Sanchez-Garcia
# Made available under the MIT license

import numpy as np
import pandas as pd

import random
from datetime import datetime

from sklearn.datasets import fetch_openml
from gtda.images import Binarizer, RadialFiltration, HeightFiltration
from gtda.homology import CubicalPersistence

import warnings

import warnings

from ..src.utils import cubical_complex
from ..src.perslap import compute_pers_lap_pair
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def mnist_pers_lap(image, min=5, max=30, inc=5):

    vals = np.zeros((int(max/inc), int(max/inc)))

    for i, k in enumerate(range(min, max+inc, inc)):
        for j, l in enumerate(range(k, max+inc, inc)):
            K = cubical_complex(image, k)
            L = cubical_complex(image, l)
            pers_lap, spectra = compute_pers_lap_pair(K, L, verbose=False, complex_type='cubical')
            spectra = np.real(spectra)
            rounded0 = np.round(spectra[0], 3)
            nonzero = rounded0[np.nonzero(rounded0)]
            vals[i, j] = np.min(nonzero)

    return vals


def run_all_directions(df, min, max, inc, direction_list=None, save=True):

    if direction_list is None:
        direction_list = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]

    for counter, direction in enumerate(direction_list):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f'Starting direction {counter}/8 at {current_time}')

        binarizer = Binarizer(threshold=0.4)
        height_filtration = HeightFiltration(direction=np.array(direction))

        for id, (image, label) in enumerate(zip(X, y)):

            image_binarised = binarizer.fit_transform(image[None, :, :])
            image_filtration = height_filtration.fit_transform(image_binarised)

            for i, k in enumerate(range(min, max + inc, inc)):
                for j, l in enumerate(range(k, max + inc, inc)):
                    K = cubical_complex(image_filtration, k)
                    L = cubical_complex(image_filtration, l)
                    pers_lap, spectra = compute_pers_lap_pair(K, L, verbose=False, complex_type='cubical')
                    for m, spectra_k in enumerate(spectra):
                        real_spectra_k = np.real(spectra_k)
                        spectra[m] = np.round(real_spectra_k, 3)

                    df = df.append({'id': id,
                                    'start_val': k,
                                    'end_val': l,
                                    'direction': direction,
                                    'center': None,
                                    'eigs': spectra,
                                    'label': label}, ignore_index=True)

            # if id % 500 == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(f'Completed {id}/7000 at {current_time}')

    if save:
        df.to_pickle('results_direction_10_mnist.pkl')

    return df


def run_all_centers(df, min, max, inc, center_list=None, save=True):

    if center_list is None:
        center_list = [[13, 6],
                        [6, 13],
                        [13, 13],
                        [20, 13],
                        [13, 20],
                        [6, 6],
                        [6, 20],
                        [20, 6],
                        [20, 20]]

    for counter, center in enumerate(center_list):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f'Starting center {counter}/9 at {current_time}')

        binarizer = Binarizer(threshold=0.4)
        radial_filtration = RadialFiltration(center=np.array(center))

        for id, (image, label) in enumerate(zip(X, y)):

            image_binarised = binarizer.fit_transform(image[None, :, :])
            image_filtration = radial_filtration.fit_transform(image_binarised)

            for i, k in enumerate(range(min, max + inc, inc)):
                for j, l in enumerate(range(k, max + inc, inc)):
                    K = cubical_complex(image_filtration, k)
                    L = cubical_complex(image_filtration, l)
                    pers_lap, spectra = compute_pers_lap_pair(K, L, verbose=False, complex_type='cubical')
                    for m, spectra_k in enumerate(spectra):
                        real_spectra_k = np.real(spectra_k)
                        spectra[m] = np.round(real_spectra_k, 3)

                    df = df.append({'id': id,
                                    'start_val': k,
                                    'end_val': l,
                                    'direction': None,
                                    'center': center,
                                    'eigs': spectra,
                                    'label': label}, ignore_index=True)

            # if id % 500 == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(f'Completed {id}/7000 at {current_time}')

    if save:
        df.to_pickle('results_centers_mnist.pkl')

    return df


def run_experiments(X, y, center_list=None, direction_list=None, save=False, resolution=5):

    for type_, cent_or_dir_list in [['center', center_list], ['direction', direction_list]]:

        df = pd.DataFrame(columns=['id', 'start_val', 'end_val', 'start_val_idx', 'end_val_idx', 'type_of_filtration',
                                   'filtration_parameter', 'eigs', 'label'])

        for counter, center_or_dir in enumerate(cent_or_dir_list):

            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(f'Starting {type_} {center_or_dir} at {current_time}')

            binarizer = Binarizer(threshold=0.4)
            if type_ == 'center':
                filtration_fitter = RadialFiltration(center=np.array(center_or_dir))
            else:
                filtration_fitter = HeightFiltration(direction=np.array(center_or_dir))

            for id, (image, label) in enumerate(zip(X, y)):

                image_binarised = binarizer.fit_transform(image[None, :, :])
                filtration = filtration_fitter.fit_transform(image_binarised)

                max_val = np.max(filtration)
                min_val = np.min(filtration)
                increment = (max_val - min_val) / resolution

                vals = [min_val + i * increment for i in range(resolution)]
                vals[-1] = max_val  # ensures whole filtration isn't in filtration(vals[-1])

                for i, k in enumerate(vals):
                    K = cubical_complex(filtration, k)
                    for j, l in enumerate(vals[i:]):
                        L = cubical_complex(filtration, l)
                        pers_lap, spectra = compute_pers_lap_pair(K, L, verbose=False, complex_type='cubical')

                        for m, spectra_k in enumerate(spectra):
                            real_spectra_k = np.real(spectra_k)
                            spectra[m] = np.round(real_spectra_k, 3)

                        df = df.append({'id': id,
                                        'start_val': k,
                                        'end_val': l,
                                        'start_val_idx': i,
                                        'end_val_idx': j,
                                        'type_of_filtration': type_,
                                        'filtration_parameter': center_or_dir,
                                        'eigs': spectra,
                                        'label': label}, ignore_index=True)

                if id % 1000 == 0:
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    print(f'Completed {id}/7000 at {current_time}')

        df.to_pickle('results_overnight_' + type_ + '.pkl')

    return df


def run_ph_experiments(X, y, center_list, direction_list):

    filtrations = []
    df = pd.DataFrame(columns=['id', 'label', 'diagrams'])

    for type_, cent_or_dir_list in [['center', center_list], ['direction', direction_list]]:
        for counter, center_or_dir in enumerate(cent_or_dir_list):

            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(f'Starting {type_} {center_or_dir} at {current_time}')

            binarizer = Binarizer(threshold=0.4)
            if type_ == 'center':
                filtration_fitter = RadialFiltration(center=np.array(center_or_dir))
            else:
                filtration_fitter = HeightFiltration(direction=np.array(center_or_dir))

            for id, (image, label) in enumerate(zip(X, y)):

                image_binarised = binarizer.fit_transform(image[None, :, :])
                filtration = filtration_fitter.fit_transform(image_binarised)

                filtrations.append(filtration)
                df = df.append({'id': id, 'label': label}, ignore_index=True)

    cubical_persistence = CubicalPersistence(n_jobs=8)
    diagrams = cubical_persistence.fit_transform(filtrations)

    df['diagrams'] = list(diagrams)
    df.to_pickle('../results/diagrams_df.pkl')
    np.save('../results/diagrams.npy', diagrams)

    return df, diagrams


def load_data():
    # Full dataset is 60,000 train, 10,000 test
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    # train_size, test_size = 6000, 1000

    # Reshape to (n_samples, n_pixels_x, n_pixels_y)
    X = X.reshape((-1, 28, 28))
    random.seed(42)
    idx = random.sample(range(len(X)), 7000)
    X = X[idx]
    y = y[idx]

    return X, y


def filter_complex(complex_, filt_vals, filt):
    complex_filtered = [[], [], []]
    for simplex, filt_val in zip(complex_, filt_vals):
        if filt_val < filt:
            dim = len(simplex[0]) - 1
            complex_filtered[dim].append(simplex[0])
    return complex_filtered


if __name__ == '__main__':

    # X, y = load_data()

    # center_list = [[13, 6], [6, 13], [13, 13], [20, 13], [13, 20], [6, 6], [6, 20], [20, 6], [20, 20]]
    # direction_list = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]

    # run_experiments(X, y, center_list, direction_list)
    # run_ph_experiments(X, y, center_list=[], direction_list=[[1, 0]])

    complexes = np.load('../results/3d_objects_complexes.npy', allow_pickle=True)

    labels = np.zeros(40)
    labels[10:20] = 1
    labels[20:30] = 2
    labels[30:] = 3

    df = pd.DataFrame(columns=['id', 'start_val', 'end_val', 'start_val_idx', 'end_val_idx', 'eigs', 'label'])

    for idx, item in enumerate(complexes):
        print(idx)
        c0 = np.array(item)

        resolution = 5
        filt_vals = c0[:, 0]
        complex_ = c0[:, 1:]
        min_val, max_val = np.min(filt_vals), np.max(filt_vals)
        increment = (max_val - min_val)/resolution

        vals = [min_val + i*increment for i in range(resolution)]
        vals[-1] = max_val

        # K = filter_complex(complex_, filt_vals, vals[1])
        # L = filter_complex(complex_, filt_vals, vals[3])
        vals = vals[1:]

        for i, k in enumerate(vals):
            K = filter_complex(complex_, filt_vals, k)
            for j, l in enumerate(vals[i:]):
                L = filter_complex(complex_, filt_vals, l)

                pers_lap, spectra = compute_pers_lap_pair(K, L, verbose=False, complex_type='simplicial')

                for m, spectra_k in enumerate(spectra):
                    real_spectra_k = np.real(spectra_k)
                    spectra[m] = np.round(real_spectra_k, 3)

                df = df.append({'id': idx,
                                'start_val': k,
                                'end_val': l,
                                'start_val_idx': i,
                                'end_val_idx': j,
                                'eigs': spectra,
                                'label': labels[idx]}, ignore_index=True)

    df.to_pickle('../results/results_3d.pkl')
