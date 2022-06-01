FORWARD_SELECTION = 1
BACKWARD_ELIMINATION = 2


class Selection(object):

    def __init__(self, way, data_set, classifier):
        self.way = way
        self.data_set = data_set
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
            if i in features:
                continue

            # Create a new feature and add it to the set.
            child = set(features)
            child.add(i)
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

        :param features:
        :return:
        """
        if self.way == FORWARD_SELECTION:
            return self.features_add_one_set(features)
        elif self.way == BACKWARD_ELIMINATION:
            return self.features_remove_one_set(features)

    def search(self):
        """
        Use greedy search for figuring out the best possible set of features. Two searches are
        possible. Forward selection and backward elimination. Both searches are greedy in nature.

        FORWARD SELECTION

        BACKWARD ELIMINATION

        :return:
        """
        # TODO: Say something about Forward Selection and Backward Elimination

        # For statistics
        accuracies = []
        features = []

        # Create the initial feature based on the algorithm
        # For backward elimination, start with all the features
        best_accuracy = -1.0
        best_features = ()
        feature = ()
        if self.way == BACKWARD_ELIMINATION:
            feature = tuple([i for i in range(10)])
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
                if accuracy > current_best_accuracy:
                    current_best_accuracy = accuracy
                    current_best_features = child

                print("\tUsing feature(s) {0} accuracy is {1:.2f}%.".format(child, accuracy * 100))

            print("\nFeature set {0} was best, accuracy is {1:.2f}%.\n".format(current_best_features,
                                                                               current_best_accuracy * 100))
            feature = current_best_features

            accuracies.append(current_best_accuracy)
            features.append(current_best_features)

            if best_accuracy < current_best_accuracy:
                best_accuracy = current_best_accuracy
                best_features = current_best_features

            # TODO: ACCURACY DESCREASED CALCULATION AND PRINT

        print("\nFinished search!! The best feature subset is {0} which has an accuracy of {1:.2f}%.\n"
              .format(best_features, best_accuracy * 100))
