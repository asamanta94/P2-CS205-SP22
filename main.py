import numpy as np

from os import path

from knn import KNN
from selection import Selection

DATASETS_DIR = "dataset"
DATA_SMALL_FILE = "CS205_SP_2022_SMALLtestdata__64.txt"
DATA_LARGE_FILE = "CS205_SP_2022_Largetestdata__51.txt"

SMALL_DATASET_PATH = path.join(DATASETS_DIR, DATA_SMALL_FILE)
LARGE_DATASET_PATH = path.join(DATASETS_DIR, DATA_LARGE_FILE)


if __name__ == '__main__':
    print("Welcome to Aritra Samanta's Feature Selection Algorithm.")
    file_path = input("Type in the name of the file to test: ")

    algorithm = int(input("Type the number of the algorithm you want to run. "
                      "(1 - Forward Selection, 2 - Backward Elimination): "))

    # Get the dataset from file
    dataset = np.genfromtxt(file_path)
    n_features = len(dataset[0][1:])
    n_instances = len(dataset)

    print("\nThis dataset has {} features (not including the class attribute), with {} instances.\n"
          .format(n_features, n_instances))

    # Run the dataset against the classifier for predicting accuracy
    classifier = KNN(1, dataset)
    accuracy = classifier.evaluate()

    print("Running nearest neighbor with all {0} features, using \"leaving-one-out\" evaluation, I get"
          " an accuracy of {1:.2f}%.\n".format(n_features, accuracy * 100))

    # Run feature selection algorithm
    print("Beginning Search.\n")
    s = Selection(algorithm, dataset, classifier)
    s.search()
