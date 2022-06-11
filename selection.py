from nn import CLASS_INDEX

FORWARD_SELECTION = 1
BACKWARD_ELIMINATION = 2

FEATURE_SELECTION_ALGORITHMS = [FORWARD_SELECTION, BACKWARD_ELIMINATION]
ALGORITHMS_STRING = ["Forward Selection", "Backward Elimination"]


class Selection(object):
    """
    Class for running the feature selection algorithms. The algorithms that this class
    can run are:
        1. Forward Selection: The algorithm is a greedy algorithm. The algorithm starts with an empty set and keeps
        adding features in a greedy manner by selecting the feature that, in addition to the other selected features,
        gives the best possible accuracy.
        2. Backward Elimination: The algorithm is a greedy algorithm. The algorithm starts with a set that contains all
        the features and keeps removing features in a greedy manner by selecting the feature that, in addition to the
        other selected features, gives the worst possible accuracy.
    """

    def __init__(self, way, data_set, classifier):
        self.way = way
        self.data_set = data_set
        self.len_data_set = len(self.data_set)
        self.default_rate = self.get_default_rate()
        self.n_features = len(self.data_set[0][1:])
        self.classifier = classifier

    def features_add_one_set(self, features):
        """
        Get the next set of features for the given set of features.

        :param features: The current set of features.
        :return: The next set of possible features.
        """
        children = []

        for i in range(self.n_features):
            # If feature is already in the feature set no need to add.
            if (i + 1) in features:
                continue

            # Create a new feature and add it to the set.
            child = set(features)
            child.add(i + 1)
            child = tuple(child)
            children.append(child)

        return children

    def features_remove_one_set(self, features):
        """
        Return a list of features where each feature in the list is a subset of the original set of features
        barring one element.

        :param features: Original set of features.
        :return: List of new set of features.
        """
        children = []
        for i in features:

            # Remove one feature and add it to the set of children
            child = set(features)
            child.remove(i)
            child = tuple(child)
            if child == ():
                continue

            children.append(child)

        return children

    def get_features(self, features):
        """
        Get the next possible set of features based on the algorithm.

        :param features: Current set of features.
        :return: Next possible set of features.
        """
        if self.way == FORWARD_SELECTION:
            return self.features_add_one_set(features)
        elif self.way == BACKWARD_ELIMINATION:
            return self.features_remove_one_set(features)

    def get_default_rate(self):
        """
        Function to get the default rate of a given dataset.

        :return: The length of a dataset.
        """
        classes = [data_point[CLASS_INDEX] for data_point in self.data_set]
        count_class_1_0 = classes.count(1.0)
        count_class_2_0 = classes.count(2.0)
        return max(count_class_1_0, count_class_2_0) / self.len_data_set

    def search(self):
        """
        Use greedy search for figuring out the best possible set of features. Two searches are
        possible. Forward selection and backward elimination. Both searches are greedy in nature.
        :return:
        """

        # For statistics
        accuracies = []
        features = []

        # Create the initial feature based on the algorithm
        # For backward elimination, start with all the features
        best_accuracy = -1.0
        best_features = ()
        feature = ()
        if self.way == BACKWARD_ELIMINATION:
            feature = tuple([i + 1 for i in range(self.n_features)])
            best_features = feature
            best_accuracy = self.classifier.evaluate()

        while True:

            # Expand the possible set of features.
            children = self.get_features(feature)

            # If no more features can be added, then search is done.
            if len(children) == 0:
                break

            # Reset local parameters
            current_best_accuracy = -1.0
            current_best_features = ()

            # For each child feature, pick the feature set with the best accuracy and add it to
            # the features list.
            for child in children:
                self.classifier.feature_list = child
                accuracy = self.classifier.evaluate()
                if accuracy >= current_best_accuracy:
                    current_best_accuracy = accuracy
                    current_best_features = child

                print("\tUsing feature(s) {0} accuracy is {1:.2f}%.".format(set(child), accuracy * 100))

            if best_accuracy >= current_best_accuracy:
                print("\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
            print("\nFeature set {0} was best, accuracy is {1:.2f}%.\n".format(set(current_best_features),
                                                                               current_best_accuracy * 100))
            # Set the next best set of features for consideration
            feature = current_best_features

            accuracies.append(current_best_accuracy)
            features.append(current_best_features)

            # Update the global best accuracy
            if best_accuracy <= current_best_accuracy:
                best_accuracy = current_best_accuracy
                best_features = current_best_features

        print("\nFinished search!! The best feature subset is {0} which has an accuracy of {1:.2f}%.\n"
              .format(set(best_features), best_accuracy * 100))

        return accuracies, features
