# R00090111 Simone Rodigari

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn import model_selection
import matplotlib.pyplot as plt


class Model:
    degree: int
    p0: []


def pre_processing():
    data = pd.read_csv("./diamonds.csv")
    combo = np.array(data[["cut", "color", "clarity"]])

    # dictionary to hold features and targets for each combination
    data_points = defaultdict(lambda: (list(), list()))
    # unique combinations
    combo_set = set(tuple(da_combo) for da_combo in combo)

    # For each combination of these cut, colour and clarity grades
    # extract the corresponding data-points [1 point]
    for combination_subset in combo_set:
        #  Going grade-by-grade split the data-points into features [1 point] and targets [1 point]
        features_ = np.array(data[(data["cut"] == combination_subset[0]) &
                                  (data["color"] == combination_subset[1]) &
                                  (data["clarity"] == combination_subset[2])
                                  ][["carat", "depth", "table"]])
        targets_ = np.array(data[(data["cut"] == combination_subset[0]) &
                                 (data["color"] == combination_subset[1]) &
                                 (data["clarity"] == combination_subset[2])
                                 ][["price"]])
        # Select only the datasets containing more than 800 data-points for further processing [1 point]
        if len(features_) > 800:
            data_points[combination_subset][0].append(features_)
            data_points[combination_subset][1].append(targets_)

    return data_points


def eval_poly_3(degree, parameter_vector, x, y, z):
    r = 0
    t = 0
    for n in range(degree + 1):
        for i in range(n + 1):
            for j in range(n + 1):
                for k in range(n + 1):
                    if i + j + k == n:
                        r = r + parameter_vector[t] * (x ** i) * (y ** j) * (z ** k)
                        t = t + 1
    return r


def num_3_coefficients(d):
    t = 0
    for n in range(d + 1):
        for i in range(n + 1):
            for j in range(n + 1):
                for k in range(n + 1):
                    if i + j + k == n:
                        t = t + 1
    return t


def linearize(deg, p0, x, y, z):
    f0 = eval_poly_3(deg, p0, x, y, z)
    J = np.zeros((1, len(p0)))
    epsilon = 1e-6
    for i in range(len(p0)):
        p0[i] += epsilon
        fi = eval_poly_3(deg, p0, x, y, z)
        p0[i] -= epsilon
        di = (fi - f0) / epsilon
        J[:, i] = di
    return f0, J


def calculate_update(y, f0, J):
    l = 1e-2
    # normal equation matrix
    N = np.matmul(J.T, J) + l * np.eye(J.shape[1])
    # residual
    r = y - f0
    # normal equation system
    n = np.matmul(J.T, r)
    # optimal parameter update
    dp = np.linalg.solve(N, n)
    return dp


def regression(pol_degree, train_data_features, train_data_targets):
    # initialise the parameter vector of coefficients with zeros
    p0_ = np.zeros(num_3_coefficients(pol_degree))
    max_iterations = 10
    f0 = np.zeros(len(train_data_features))
    J = np.zeros((len(train_data_features), len(p0_)))
    # targets = np.zeros(len(f0))
    for ii in range(max_iterations):
        i = 0
        for (feature, target) in zip(train_data_features, train_data_targets):
            # alternates linear and parameter update
            x, y, z = feature[0][0], feature[0][1], feature[0][2]
            f0i, Ji = linearize(pol_degree, p0_, x, y, z)
            f0[i] = f0i
            J[i, :] = Ji
            # targets[i] = target[0]
            i += 1
        dp = calculate_update(train_data_targets[0][0], f0, J)
        p0_ += dp
    return p0_


def k_fold(feats, targs):
    feats = feats[0]
    targs = targs[0]
    differences, predictive_model = [], Model()
    # initialize  best model
    predictive_model.degree, predictive_model.p0 = 0, []

    # Setup a k-fold cross-validation procedure for all datasets extracted in task 1 [1 point]
    kf = model_selection.KFold(n_splits=8, shuffle=True)
    for train_index, test_index in kf.split(feats, targs):
        # Compare different model functions by selection
        # different polynomial degrees ranging from 0 to 3 [1 point]
        for degree in range(4):
            p0 = regression(degree, [feats[train_index]], [targs[train_index]])
            prediction = eval_poly_3(degree, p0,
                                     feats[ls][0][0],
                                     feats[test_index][0][1],
                                     feats[test_index][0][2])
            # Calculate the difference between the predicted prices and the
            # actual sale prices for the test set in each fold [1 point]
            difference = abs(targs[test_index][0] - prediction)
            differences.append(difference)
            if difference == np.min(differences):
                predictive_model.degree = degree
                predictive_model.p0 = p0

    # Use the mean absolute price difference as measure of
    # quality for the different model functions [1 point] --> this is done above
    print('mean difference: ', np.mean(differences))
    # Determine the best polynomial degree for each dataset [1 point]
    print('best polynomial degree: ', predictive_model.degree)

    return predictive_model


def visualization(feats, targs, p0, deg):
    feats = feats[0]
    targs = targs[0]
    # Estimate the model parameters for each dataset [1 point]
    # using the selected optimal model function as determined in task 6. ---> this happens in the calling method loop
    predictions = []
    for feature, target in zip(feats, targs):
        # Calculate the estimated price for each diamond in the
        # dataset using the estimated model parameters [1 point]
        predictions.append(eval_poly_3(deg, p0, feature[0], feature[1], feature[2]))

    # Plot the estimated prices against the true sale prices [1 point]
    plt.close("all")
    fig, ax1t = plt.subplots()
    # price
    ax1t.set_yticks([100, 1000, 9000, 18000])
    # sample number (within subset)
    ax1t.set_xticks([100, 200, 800, 1100])
    ax1t.plot(targs, color="grey")
    ax1t.plot(predictions, color="brown")
    plt.show()


''' main execution '''
dp = pre_processing()

# for features, targets in dp.values():
#     p0 = regression(3, features, targets)
#     print(p0)

for features, targets in dp.values():
    model = k_fold(features, targets)
    visualization(features, targets, model.p0, model.degree)
    print()

