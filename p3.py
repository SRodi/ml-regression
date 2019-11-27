import numpy as np
import pandas as pd

# Create a function that loads the file and
data = pd.read_csv("./diamonds.csv")
# extracts what types of cut qualities [1 point]
# colour grades [1 point] clarity grades [1 point]
combo = np.array(data[["cut", "color", "clarity"]])
# For each combination of these cut, colour and clarity grades
# extract the corresponding data-points [1 point]
combo_set = set(tuple(da_combo) for da_combo in combo)  # unique combinations
print('number of combinations: ', len(combo_set))
data_dictionary = dict.fromkeys(combo_set, [])
print(len(data_dictionary.keys()))
# Create a loop going over all combinations of cut, colour, and clarity [1 point]
for combination_subset in combo_set:
    data_dictionary[combination_subset] = list(np.array(data[(data["cut"] == combination_subset[0]) &
                                           (data["color"] == combination_subset[1]) &
                                           (data["clarity"] == combination_subset[2])
                                           ][["carat", "depth", "table", "price", "x", "y", "z"]]))
    # count the number of data-points within each subset [1 point]
    number_data_points = len(data_dictionary[combination_subset])
    # print('Number of data-points for ', combination_subset, ': ', number_data_points)
    # Select only the datasets containing more than 800 data-points for further processing [1 point]
    if number_data_points < 800:
        del data_dictionary[combination_subset]
print(len(data_dictionary.keys()))
# Going grade-by-grade
features, targets = [], []
for combination_subset in combo_set:
    try:
        # split the data-points into features [1 point]
        features.append([data_dictionary[combination_subset][0],
                        data_dictionary[combination_subset][1],
                        data_dictionary[combination_subset][2]])
        # and targets [1 point]
        targets.append(data_dictionary[combination_subset][3])
    except KeyError:
        # some handling here
        val = 0
print(features)
print(targets)
print('data_points: ', len(data_dictionary.keys()))
print('features: ', len(features))
print('targets: ', len(targets))



