import numpy as np
import pandas as pd


def pre_processing():
    # Create a function that loads the file and
    data = pd.read_csv("./diamonds.csv")

    # extracts what types of cut qualities [1 point]
    # colour grades [1 point] clarity grades [1 point]
    combo = np.array(data[["cut", "color", "clarity"]])

    # unique combinations
    combo_set = set(tuple(da_combo) for da_combo in combo)
    print('number of ["cut", "color", "clarity"] combinations: ', len(combo_set))

    # create a dictionary which will hold the data points for each combination
    data_dictionary = dict.fromkeys(combo_set, np.zeros(shape=(len(combo_set))))

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
                # split the data-points into features [1 point]
                features.append([data_points[0],
                                 data_points[1],
                                 data_points[2]])
                # and targets [1 point]
                targets.append(data_points[3])

    print('features: ', len(features))
    print('targets: ', len(targets))
    print(features)
    print(targets)

    return features, targets


feat, tar = pre_processing()
