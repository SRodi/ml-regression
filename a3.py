import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn import model_selection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def pre_processing():
    # extracts what types of cut qualities [1 point]
    # colour grades [1 point] clarity grades [1 point]
    data = pd.read_csv("./diamonds.csv")
    combo = np.array(data[["cut", "color", "clarity"]])

    # dictionary to hold features and targets for each combination
    data_points = defaultdict(lambda: (list(), list()))
    # unique combinations
    combo_set = set(tuple(da_combo) for da_combo in combo)

    # For each combination of these cut, colour and clarity grades
    # extract the corresponding data-points [1 point]
    features, targets = [], []
    for combination_subset in combo_set:
        features = list(np.array(data[(data["cut"] == combination_subset[0]) &
                                       (data["color"] == combination_subset[1]) &
                                       (data["clarity"] == combination_subset[2])
                                       ][["carat", "depth", "table"]]))
        targets = list(np.array(data[(data["cut"] == combination_subset[0]) &
                                      (data["color"] == combination_subset[1]) &
                                      (data["clarity"] == combination_subset[2])
                                      ][["price"]]))
        if len(features) > 800:
            data_points[combination_subset][0].append(features)
            data_points[combination_subset][1].append(targets)

    return data_points


# polynomial
def eval_poly_3(degree, features_vector, parameter_vector):
    r = np.zeros(features_vector.shape[0])
    t = 0
    for feature in features_vector:
        x, y, z = feature[0], feature[1], feature[2]
        for n in range(degree + 1):
            for i in range(n + 1):
                for j in range(n + 1):
                    for k in range(n + 1):
                        if i + j + k == n:
                            # if t < len(feature):
                            r = r + parameter_vector[t] * (x ** i) * (y ** j) * (z ** k)
                            t = t + 1
        return r


# parameter vector
def model_function_3_coefficients(d):
    t = 0
    for n in range(d + 1):
        for i in range(n + 1):
            for j in range(n + 1):
                for k in range(n + 1):
                    if i + j + k == n:
                        t = t + 1
    return t


def linearize(deg, data, p0):
    f0 = eval_poly_3(deg, data, p0)
    J = np.zeros((len(f0), len(p0)))
    epsilon = 1e-6
    for i in range(len(p0)):
        p0[i] += epsilon
        fi = eval_poly_3(deg, data, p0)
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
    p0 = np.zeros(model_function_3_coefficients(pol_degree))

    # alternates linearization and parameter update
    f0, J = linearize(pol_degree, np.array(train_data_features), p0)
    for train_target in train_data_targets:
        dp = calculate_update(train_target[0], f0, J)
        p0 += dp

    return p0


def k_fold(features, targets):
    features = pd.Series(features)
    targets = pd.Series(targets)
    print(len(features))
    print(len(targets))
    # Calculate the difference between the predicted prices and the
    # actual sale prices for the test set in each fold [1 point]
    kf = model_selection.KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(features, targets):
        for degree in range(3):
            p0 = regression(degree, list(features.iloc[train_index].as_matrix(columns=None)),
                            list(targets.iloc[train_index].as_matrix(columns=None)))
            print(p0)
            print(targets[train_index])
    # Compare different model functions by selection different polynomial
    # degrees ranging from 0 to 3 [1 point]

    # Use the mean absolute price difference as measure of quality for the different model functions [1 point]

    # Determine the best polynomial degree for each dataset [1 point]


def visualization(dp):
    # Estimate the model parameters for each dataset [1 point]
    # using the selected optimal model function as determined in task 6.
    for features, targets in dp.values():
        print(type(features))
        print(type(targets))

        # Calculate the estimated price for each diamond in the dataset using the estimated model parameters [1 point]
        p0 = regression(3, features, targets)
        print(type(p0))

        # Plot the estimated prices against the true sale prices [1 point]
        # targets = np.asarray(targets)
        # print(type(targets))
        # print(type(targets))

        plt.close("all")
        fig, ax1t = plt.subplots()
        # plt.suptitle(classifier)
        # ax1t = ax1.twinx()
        # plot times
        # ax1.plot(num_samples, avg_train_times, color="grey")
        # ax1.plot(num_samples, avg_predict_times, color="brown")
        # ax1.set_ylim([0, 1.5])
        ax1t.set_yticks(p0)
        ax1t.set_xticks(targets[:0])
        # ax1.spines['top'].set_visible(False)
        # ax1.spines['left'].set_visible(False)
        # ax1.spines['right'].set_visible(False)
        # plot accuracies
        ax1t.plot(p0, p0, color="orange")
        # ax1t.set_ylim([0.8, 1.1])
        # ax1t.set_yticks([0.8, 0.9, 0.93, 0.95])
        # ax1t.spines['top'].set_visible(False)
        # ax1t.spines['left'].set_visible(False)
        # ax1t.spines['right'].set_visible(False)
        # print for report purpose
        # filename = 'comparison_' + classifier + '.png'
        # fig.savefig(filename)
        plt.show()


'''
    main execution
'''

dp = pre_processing()


# for features, targets in dp.values():
#     # print(features)
#     # print(targets)
#     p0 = regression(3, features, targets)
#     print(p0)
#
# for features, targets in dp.values():
#     k_fold(features[0], targets[0])

visualization(dp)













    # max_iter = 10
    # for deg in range(3):
    #     p0 = np.zeros(model_function_3_coefficients(deg))
    #     for i in range(max_iter):
    #         f0, J = linearize(deg, np.array(features), p0)
    #         for target in targets:
    #             dp = calculate_update(target[0], f0, J)  # todo:  pass target
    #             p0 += dp
    #     print(p0)

# task 1- 5 completed apart in _raise_linalgerror_singular
#     raise LinAlgError("Singular matrix")
# numpy.linalg.LinAlgError: Singular matrix
#
# -->  maybe relates to regularisation term to prevent the normal equation system from being singular
