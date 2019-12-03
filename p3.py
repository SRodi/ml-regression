import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


def pre_processing():
    # Create a function that loads the file and
    data = pd.read_csv("./diamonds.csv")

    # extracts what types of cut qualities [1 point]
    # colour grades [1 point] clarity grades [1 point]
    combo = np.array(data[["cut", "color", "clarity"]])

    features = data[["carat", "depth", "table", "price"]]
    targets = data[["price"]]

    # unique combinations
    combo_set = set(tuple(da_combo) for da_combo in combo)
    print('number of ["cut", "color", "clarity"] combinations: ', len(combo_set))

    # create a dictionary which will hold the data points for each combination
    data_dictionary = dict.fromkeys(combo_set, np.zeros(shape=(len(combo_set))))
    dictio = defaultdict(lambda: (list(), list()))
    # create list for features and targets
    features, targets = [], []

    # For each combination of these cut, colour and clarity grades
    # extract the corresponding data-points [1 point]
    for combination_subset in combo_set:

        # add entry to dictionary as np array containing all data points for the grade's combination
        data_dictionary[combination_subset] = np.array(data[(data["cut"] == combination_subset[0]) &
                                                            (data["color"] == combination_subset[1]) &
                                                            (data["clarity"] == combination_subset[2])
                                                            ][["carat", "depth", "table", "price"]])

        # count the number of data-points within each subset [1 point]
        number_data_points = len(data_dictionary[combination_subset])
        print('Number of data-points for ', combination_subset, ': ', number_data_points)

        # Select only the datasets containing more than 800 data-points for further processing [1 point]
        if number_data_points < 800:
            del data_dictionary[combination_subset]
        else:
            for data_points in data_dictionary[combination_subset]:
                # # split the data-points into features [1 point]
                # features.append([data_points[0],
                #                  data_points[1],
                #                  data_points[2]])
                # # and targets [1 point]
                # targets.append(data_points[3])

                dictio[combination_subset][0].append([data_points[0],
                                 data_points[1],
                                 data_points[2]])
                dictio[combination_subset][1].append(data_points[3])

    print('features: ', len(features))
    print('targets: ', len(targets))
    print(features)
    print(targets)

    return dictio


# def calculate_model_function(deg, data, p):
#     array = data
#     result = np.zeros(array.shape[0])
#     t=0
#     for n in range(deg+1):
#         for i in range(n+1):
#             for j in range(n+1):
#                 for k in range(n+1):
#                     if i + j + k == n:
#                         result += p[t]*(array[:,0]**i)*(array[:,1]**j)*(array[:,2]**k)
#                         t+=1
#     return result

#  polynomial model function that takes as input parameters the degree of the polynomial,
#  a list of feature vectors as extracted in task 1, and a parameter vector of coefficients
def calculate_model_function(deg, data, p):
    result = np.zeros(data.shape[0])
    k = 0
    for n in range(deg + 1):
        for i in range(n + 1):
            for j in range(n + 1):
                for o in range(n + 1):
                    if i + j + o == n:
                        # calculates the estimated target vector using a multi-variate
                        # polynomial of the specified degree [3 points].
                        result += p[k] * (data[:, 0] ** i) * (data[:, 1] ** j) * (data[:, 2] ** o)
                        k += 1
    return result


def linearize(deg, data, p0):
    f0 = calculate_model_function(deg, data, p0)
    J = np.zeros((len(f0), len(p0)))
    epsilon = 1e-6
    for i in range(len(p0)):
        p0[i] += epsilon
        fi = calculate_model_function(deg, data, p0)
        p0[i] -= epsilon
        di = (fi - f0)/epsilon
        J[:, i] = di
    return f0, J


def calculate_update(y, f0, J):
    l = 1e-2
    N = np.matmul(J.T, J) + l*np.eye(J.shape[1])
    r = y-f0
    n = np.matmul(J.T, r)
    dp = np.linalg.solve(N, n)
    return dp


# determines the correct size for the parameter vector
# from the degree of the multi-variate polynomial [1 point]
def num_coefficients_3(d):
    t = 0
    for n in range(d+1):
        for i in range(n+1):
            for j in range(n+1):
                for k in range(n+1):
                    if i+j+k == n:
                        t = t+1
    return t


# for deg in range(3):
#     p0 = np.zeros(num_coefficients_3(deg))
#     print(model_function(deg, np.array(features_), p0))

data = pre_processing()


max_iter = 10
for subset in data.values():
    feat_data = np.array(subset)
    for deg in range(5):
        p0 = np.zeros(num_coefficients_3(deg))
        for i in range(max_iter):
            f0, J = linearize(deg, feat_data, p0)
            dp = calculate_update(feat_data, f0, J)
            p0 += dp

        x, y = np.meshgrid(np.arange(np.min(feat_data[:, 0]), np.max(feat_data[:, 0]), 0.1),
                           np.arange(np.min(feat_data[:, 1]), np.max(feat_data[:, 1]), 0.1))
        test_data = np.array([x.flatten(), y.flatten()]).transpose()
        test_target = calculate_model_function(deg, test_data, p0)
