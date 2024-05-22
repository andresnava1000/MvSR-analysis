import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
import random
import math
import itertools
from numpy.random import default_rng
import shutil


global_rng = default_rng(seed=0)


def load_data_from_file(filepath):
    data = np.loadtxt(filepath)
    return data.T


def zero_function(x, *args):
    # print("shape of x",x.shape, np.zeros(x.shape[0]), len(x[:, 0]), int(*args))
    return int(*args) * np.ones(x.shape[0])  # I am hoping that we will be able to learn an interesting function?


def identity_function(x, *args):
    return np.array(args)  # float(*args) * np.ones(x.shape[0])


def label_function(index):  # we can adjust the labels to be anything as long as they are unequal!
    if index == 0:
        return math.e
    elif index == 1:
        return math.pi**2


def func_chirp_mass(x, c):
    try:
        y = ((2 / 3) ** (1 / 3) * c) / (
            np.sqrt(3) * np.sqrt(27 * c**2 * x**14 - 4 * c**3 * x**9) + 9 * c * x**7
        ) ** (1 / 3) + (np.sqrt(3) * np.sqrt(27 * c**2 * x**14 - 4 * c**3 * x**9) + 9 * c * x**7) ** (
            1 / 3
        ) / (
            2 ** (1 / 3) * 3 ** (2 / 3) * x**3
        )
        return y
    except ValueError:  # This will catch domain errors from the sqrt function
        return np.nan


def func_poly3(X, A, B, C, D):
    return A + B * X + C * X**2 + D * X**3


def func_fried1(X, A, B, C, D):
    # Original functional form comes from here :
    # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman1.html
    # The function was modified to incorporate 4 free parameters (and remove constants)
    return np.sin(A * X[:, 0] * X[:, 1]) + B * (X[:, 2] - C) ** 2 + D * X[:, 3] + X[:, 4]


def func_fried2(X, A, B, C, D):
    # Original functional form comes from here :
    # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman2.html
    # The function was modified to incorporate 4 free parameters
    return (A * X[:, 0] ** 2 + (B * X[:, 1] * X[:, 2] - C / (D * X[:, 1] * X[:, 3] + 1)) ** 2) ** 0.5


def gaussian_noise(y, rng, noise_ratio):
    sigma = np.std(y) * np.sqrt(noise_ratio / (1.0 - noise_ratio))
    return rng.normal(loc=0.0, scale=sigma, size=len(y))


def create_folders(name, noises):
    if not os.path.exists("toy_data"):
        os.makedirs("toy_data")

    # Delete previous data if it exists
    if os.path.isdir(f"toy_data/{name}"):
        shutil.rmtree(f"toy_data/{name}")

    if not os.path.exists(f"toy_data/{name}"):
        os.makedirs(f"toy_data/{name}")

    if not os.path.exists(f"toy_data/{name}/perfect"):
        os.makedirs(f"toy_data/{name}/perfect")

    for noise in noises:
        if not os.path.exists(f"toy_data/{name}/noisy_{noise}"):
            os.makedirs(f"toy_data/{name}/noisy_{noise}")


def generate_data(func, name, Xs, nXs, params, noises, oversample=False):

    if nXs == 1:
        header = ["Xaxis0", "yaxis"]
    else:
        header = []
        for i in range(nXs):
            header.append(f"Xaxis{i}")
        header.append("yaxis")

    create_folders(name, noises)

    for idx, param in enumerate(params):
        x = Xs[idx]
        y = func(x.T, *param)
        # y = (
        #     10 * y / (max([abs(k) for k in y])) if np.abs(max([abs(k) for k in y])) != 0 else y
        # )  # normalization that probably we can remove. (probably stability reasons)
        if type(oversample) == list:
            x = np.concatenate([list(x) for _k in range(oversample[idx])])
            y = np.concatenate([list(y) for _k in range(oversample[idx])])

        if len(np.shape(x.T)) == 1:
            example = np.vstack((x.T, y)).T
        elif len(np.shape(x.T)) == 2:
            example = np.array([list(x.T[idx2]) + [y[idx2]] for idx2 in range(len(x.T))])

        # print("Shape of x:", x.shape)
        # print("Example data point: len ", len(x[0]))
        # print("Shape of y:", y.shape)
        # print("Example output:", y[0])

        with open(
            f"toy_data/{name}/perfect/example{idx}.csv",
            "w",
            encoding="UTF8",
            newline="",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(example)

        for noise in noises:
            # y_noisy = y + gaussian_noise(y, global_rng, noise)
            x = Xs[idx]
            x[0, :] += gaussian_noise(x[0, :], global_rng, noise)
            x[1, :] += gaussian_noise(x[1, :], global_rng, noise)
            # print('noise x:',noise,x.shape,len(x[:,0]),len(x[1,:]), gaussian_noise(x[1,:], global_rng, noise), np.mean(np.abs(gaussian_noise(x[1,:], global_rng, noise)) ))

            if type(oversample) == list:  # not us.
                y_noisy = np.concatenate(
                    [list(y_noisy[: int(len(y) / oversample[idx])]) for _k in range(oversample[idx])]
                )
            example = np.vstack((x, y)).T

            with open(
                f"toy_data/{name}/noisy_{noise}/example{idx}.csv",
                "w",
                encoding="UTF8",
                newline="",
            ) as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(example)


if __name__ == "__main__":

    noises = []  # can this be empty?
    data_directory = "/home/anava/projects/symbolic_regression_examples/data/paths/2D/90"
    file_list = [f for f in os.listdir(data_directory) if f.endswith(".txt")]
    n_files = len(file_list)
    file_combinations = list(itertools.combinations(file_list, 2))
    n_combinations = len(file_combinations)

    # c_values = [
    #     math.sqrt(random.uniform(1, 100)) for _ in range(n_combinations)
    # ]  # set contour label as random irrational number between 0 and 10
    # print(c_values)
    Xs = []
    c_values = []
    # for file in file_list:
    #     filepath = os.path.join(data_directory, file)
    #     Xs.append(load_data_from_file(filepath))

    for file_pair in file_combinations:
        data_combined = []
        c_value = []
        for idx, file in enumerate(file_pair):
            filepath = os.path.join(data_directory, file)
            data = load_data_from_file(filepath)
            data_combined.append(data)
            label = label_function(idx) * np.ones(data.shape[1])
            c_value.extend([label_function(idx)] * data.shape[1])
        combined_data = np.hstack(data_combined)  # Combine the two data arrays horizontally
        Xs.append(combined_data)
        c_values.append(c_value)

    generate_data(
        identity_function,
        "degeneracy_paths_2D_90",
        Xs,
        2,  # You now have 2 input dimensions
        c_values,  # Pass parameters
        noises,
    )

    generate_chirp_data = False
    if generate_chirp_data:
        noises = [0.033, 0.066, 0.1]

        # ____________________________________________________________________
        # Generate chirp mass data (2d input and output is zero.) :
        n_samples = 2809
        Xs_lim = [1.25, 1.55]
        c_values = [1.1, 1.215**5]
        # Xs = [np.linspace(Xs_lim[0], Xs_lim[1], n_samples) for _ in c_values]
        Xs = []
        for c in c_values:
            x0 = np.linspace(Xs_lim[0], Xs_lim[1], n_samples)
            x1 = func_chirp_mass(x0, c)
            Xs.append(np.vstack((x0, x1)))

        # generate_data(
        #     func_chirp_mass,
        #     "chirp_mass",
        #     Xs,
        #     1,
        #     [[c] for c in c_values],
        #     noises,
        # )

        generate_data(
            zero_function,  # This function returns zeros, aligning with your desired output
            "chirp_mass",
            Xs,
            2,  # You now have 2 input dimensions
            [
                [c] for c in c_values
            ],  # Pass parameters. (I have choosen them so that their floor is 1 and 2 which we will fit our data to.)
            noises,
        )

    # ____________________________________________________________________
    # Generate polynomial data with parameters to 0 :
    # step = 0.2
    # Xs, Xs_lim = [], [-2, 2]

    # for i in range(4):
    #     Xs.append(np.arange(Xs_lim[0], Xs_lim[1], step))

    # generate_data(
    #     func_poly3,
    #     "polynomial0",
    #     Xs,
    #     1,
    #     [[2, 2, 0, 0], [0, 2, 2, 0], [0, 0, 2, 2], [2, 0, 0, 2]],
    #     noises,
    # )

    # # ____________________________________________________________________
    # # Generate polynomial data with partial view :
    # step = 0.05
    # Xs, Xs_lim = [], [[-2, -1], [-1, 0], [0, 1], [1, 2]]

    # for idx, lim in enumerate(Xs_lim):
    #     Xs.append(np.arange(lim[0], lim[1], step))

    # generate_data(
    #     func_poly3,
    #     "polynomial_partial",
    #     Xs,
    #     1,
    #     [[2, -2, 2, 2], [2, -2, 2, 2], [2, -2, 2, 2], [2, -2, 2, 2]],
    #     noises,
    # )

    # # ____________________________________________________________________
    # # Generate friedman1 data :
    # npoints = 100
    # Xs, nXs = [], 5

    # # Loop through each example
    # for _ in range(4):
    #     loop = []
    #     # Loop through each X
    #     for i in range(nXs):
    #         loop.append(np.random.random_sample(npoints))
    #     Xs.append(np.array(loop))

    # generate_data(
    #     func_fried1,
    #     "friedman1",
    #     Xs,
    #     nXs,
    #     [[2, 2, 0, 0], [0, 2, 2, 0], [0, 0, 2, 2], [2, 0, 0, 2]],
    #     noises,
    # )

    # # ____________________________________________________________________
    # # Generate friedman2 data :
    # npoints = 100
    # Xs, nXs = [], 4

    # # Loop through each example
    # for _ in range(4):
    #     loop = []
    #     loop.append(np.random.uniform(low=0, high=100, size=npoints))
    #     loop.append(np.random.uniform(low=40 * np.pi, high=560 * np.pi, size=npoints))
    #     loop.append(np.random.uniform(low=0, high=1, size=npoints))
    #     loop.append(np.random.uniform(low=1, high=11, size=npoints))
    #     Xs.append(np.array(loop))

    # generate_data(
    #     func_fried2,
    #     "friedman2",
    #     Xs,
    #     nXs,
    #     [[2, 2, 0, 0], [0, 2, 2, 0], [0, 0, 2, 2], [2, 0, 0, 2]],
    #     noises,
    # )
