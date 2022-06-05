import matplotlib.pyplot as plt

from main import get_data, FILE_PATHS
from selection import FEATURE_SELECTION_ALGORITHMS


def plot_data(accuracies, features):
    # Plot depth vs. max queue size
    plt.figure("Figure 1")
    plt.plot(features, accuracies)
    plt.xlabel("Features")
    plt.ylabel("Accuracy")
    plt.title("")
    plt.show()


if __name__ == "__main__":

    for file_path in FILE_PATHS:
        for algorithm in FEATURE_SELECTION_ALGORITHMS:
            a, f = get_data(algorithm, file_path)
            a_per = [i * 100 for i in a]
            f_str = [len(i) for i in f]
            plot_data(a_per, f_str)
