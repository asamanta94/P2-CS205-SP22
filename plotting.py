import matplotlib.pyplot as plt

from main import get_data, FILE_PATHS
from selection import FEATURE_SELECTION_ALGORITHMS, ALGORITHMS_STRING


def plot_data(accuracies, features, figure_no, algo_str, file_str="Empty"):
    # Plot depth vs. max queue size
    plt.figure("Figure {}".format(figure_no))
    plt.bar(features, accuracies)
    plt.xlabel("Features")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Number of features - {}, {}".format(algo_str, file_str))
    plt.draw()


if __name__ == "__main__":

    k = 1
    for file_path in FILE_PATHS:
        for algorithm in FEATURE_SELECTION_ALGORITHMS:
            a, f = get_data(algorithm, file_path)
            a_per = [i * 100 for i in a]
            f_str = [len(i) for i in f]
            plot_data(a_per, f_str, k, ALGORITHMS_STRING[algorithm - 1], file_path)
            k = k + 1

    plt.show()
