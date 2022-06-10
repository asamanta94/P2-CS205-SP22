import time

import numpy as np

from os import path

from nn import NN
from selection import Selection

DATASETS_DIR = "dataset"
DATA_SMALL_FILE = "CS205_SP_2022_SMALLtestdata__64.txt"
DATA_LARGE_FILE = "CS205_SP_2022_Largetestdata__51.txt"
CALIBRATION_FILE_1 = "CS205_CalibrationData__1.txt"
CALIBRATION_FILE_2 = "CS205_CalibrationData__2.txt"
CALIBRATION_FILE_3 = "CS205_CalibrationData__3.txt"

SMALL_DATASET_PATH = path.join(DATASETS_DIR, DATA_SMALL_FILE)
LARGE_DATASET_PATH = path.join(DATASETS_DIR, DATA_LARGE_FILE)
CALIBRATION_FILE_1_PATH = path.join(DATASETS_DIR, CALIBRATION_FILE_1)
CALIBRATION_FILE_2_PATH = path.join(DATASETS_DIR, CALIBRATION_FILE_2)
CALIBRATION_FILE_3_PATH = path.join(DATASETS_DIR, CALIBRATION_FILE_3)

FILE_PATHS = [SMALL_DATASET_PATH, LARGE_DATASET_PATH, CALIBRATION_FILE_1_PATH, CALIBRATION_FILE_2_PATH, CALIBRATION_FILE_3_PATH]


def get_user_input():
    """

    :return:
    """
    print("Welcome to Aritra Samanta's Feature Selection Algorithm.")
    f_path = input("Type in the name of the file to test: ")

    algo = int(input("Type the number of the algorithm you want to run. "
                     "(1 - Forward Selection, 2 - Backward Elimination): "))

    return f_path, algo


def get_data(algorithm, file_path):
    """
    Function to run the algorithm on the file path.

    :param algorithm:
    :param file_path:
    :return:
    """
    # Get the dataset from file
    dataset = np.genfromtxt(file_path)
    n_features = len(dataset[0][1:])
    n_instances = len(dataset)

    print("\nThis dataset has {} features (not including the class attribute), with {} instances.\n"
          .format(n_features, n_instances))

    # Run the dataset against the classifier for predicting accuracy
    classifier = NN(dataset)
    accuracy = classifier.evaluate()

    print("Running nearest neighbor with all {0} features, using \"leaving-one-out\" evaluation, I get"
          " an accuracy of {1:.2f}%.\n".format(n_features, accuracy * 100))

    # Run feature selection algorithm
    print("Beginning Search.\n")
    s = Selection(algorithm, dataset, classifier)
    start_time = time.time()
    accuracies, features = s.search()
    end_time = time.time()
    duration = end_time - start_time
    if duration < 60.00:
        print("{0:.2f} seconds".format(duration))
    else:
        print("{0:.2f} minutes".format(duration / 60))
    return accuracies, features


if __name__ == '__main__':
    f, a = get_user_input()
    get_data(a, f)
