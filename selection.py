FORWARD_SELECTION = 1
BACKWARD_ELIMINATION = 2


class Selection(object):

    def __init__(self, way, data_set, classifier):
        self.way = way
        self.data_set = data_set
        self.n_features = len(self.data_set[0][1:])
        self.classifier = classifier
        self.visited = set()

    def get_features(self, features):
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
            if child in self.visited:
                continue
            children.append(child)

        return children

    def search(self):
        accuracies = []
        features = []

        best_accuracy = -1.0
        best_features = ()

        if self.way == FORWARD_SELECTION:
            features.append(())
            self.visited.add(())

            while True:
                # Get the best feature
                top = features[0]
                features.remove(top)

                # Expand the possible set of features.
                children = self.get_features(top)
                self.visited.add(top)

                # If no more features can be added, then search is done.
                if len(children) == 0:
                    break

                # Reset local parameters
                current_best_accuracy = -1.0
                current_best_features = ()

                # For each child feature, pick the feature set with the best accuracy and add it to
                # the features list.
                for child in children:
                    if child not in features:
                        self.classifier.feature_list = child
                        accuracy = self.classifier.evaluate()
                        if accuracy > current_best_accuracy:
                            current_best_accuracy = accuracy
                            current_best_features = child

                        print("\tUsing feature(s) {0} accuracy is {1:.2f}%.".format(child, accuracy * 100))

                print("\nFeature set {0} was best, accuracy is {1:.2f}%.\n".format(current_best_features,
                                                                                   current_best_accuracy * 100))
                features.append(current_best_features)

                if best_accuracy < current_best_accuracy:
                    best_accuracy = current_best_accuracy
                    best_features = current_best_features

                # TODO: ACCURACY DESCREASED CALCULATION AND PRINT

            print("\nFinished search!! The best feature subset is {0} which has an accuracy of {1:.2f}%.\n"
                  .format(best_features, best_accuracy * 100))
