import matplotlib.pyplot as plt

from main import get_data, FILE_PATHS
from selection import FEATURE_SELECTION_ALGORITHMS, ALGORITHMS_STRING


def plot_data(accuracies, features, figure_no, algo_str, file_str):
    """
    Function to plot the data.

    :param accuracies: The accuracies for the features that were considered during feature selection.
    :param features: Features that were considered for feature selection.
    :param figure_no: For plotting data into a particular figure.
    :param algo_str: String representation of the algorithm being used.
    :param file_str: Path name of the file being used.
    """
    # Plot depth vs. max queue size
    plt.figure("Figure {}".format(figure_no))
    plt.bar(features, accuracies)
    plt.xlabel("Features")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Number of features - {}, {}".format(algo_str, file_str))

    # Non-blocking draw
    plt.draw()


if __name__ == "__main__":

    k = 1
    # For each of the algorithms and for all the data sets, run the algorithm and plot the data
    for file_path in FILE_PATHS:
        for algorithm in FEATURE_SELECTION_ALGORITHMS:
            # Run the feature selection algorithm on a dataset.
            a, f = get_data(algorithm, file_path)

            # Accuracy needs to be multiplied by a 100 because they're returned as ratios from get_data
            a_per = [i * 100 for i in a]
            f_str = [len(i) for i in f]
            plot_data(a_per, f_str, k, ALGORITHMS_STRING[algorithm - 1], file_path)
            k = k + 1

    # Show figures
    plt.show()
