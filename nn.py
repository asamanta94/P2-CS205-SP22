import numpy as np

from math import sqrt

CLASS_INDEX = 0
DATA_INDEX = 1


class NN(object):

    def __init__(self, data_set):
        self.data_set = data_set
        self.data_set_len = len(self.data_set)
        self.n_features = len(self.data_set[0][DATA_INDEX:])
        self._feature_list = tuple([i for i in range(self.n_features)])

    def euclidean_distance(self, v1, v2):
        """
        Function to calculate the Euclidean distance between 2 vectors.

        :param v1: The first vector.
        :param v2: The second vector.
        :return: A floating number denoting the Euclidean distance between 2 vectors.
        """
        distance = 0

        # Calculate distance (a1 - b1) ^ 2 + (a2 -b2) ^ 2 + ... + (an - bn) ^ 2
        for i in self._feature_list:
            distance += pow(v1[i - 1] - v2[i - 1], 2)

        # Return the square root of that distance
        return sqrt(distance)

    def predict(self, trained_data, new_data_point):
        """
        Predict the class for a sample data, given some trained data.

        :param trained_data: Trained data.
        :param new_data_point: New data point for prediction using KNN classifier.
        :return: The predicted class.
        """
        # Get K nearest neighbors for the sample data point
        trained_data_len = len(trained_data)
        smallest_distance = 1000000
        smallest_distance_class = 1.0

        # Calculate the euclidean distances to all the neighbors and get the class for the data point
        # that has the smallest distance to the new data point.
        for i in range(trained_data_len):
            distance = self.euclidean_distance(new_data_point[DATA_INDEX:], trained_data[i][DATA_INDEX:])
            if smallest_distance > distance:
                smallest_distance = distance
                smallest_distance_class = trained_data[i][CLASS_INDEX]

        # Return the class found to have the smallest distance.
        return smallest_distance_class

    def evaluate(self):
        """
        Evaluate the accuracy of the model using the K-fold cross validation.

        :return: The accuracy as a floating point integer.
        """
        correct_count = 0
        for i in range(self.data_set_len):

            # Leave one piece of data out
            sample_data_for_testing = self.data_set[i]

            # Take the rest of the data for training
            trained_data = np.concatenate((self.data_set[:i], self.data_set[(i + 1):]))

            # Predict
            predicted_class = self.predict(trained_data, sample_data_for_testing)

            # Check the predicted class and adjust the correct count
            if predicted_class == sample_data_for_testing[CLASS_INDEX]:
                correct_count = correct_count + 1

        # Return the accuracy
        return correct_count / self.data_set_len

    @property
    def feature_list(self):
        return self._feature_list

    @feature_list.setter
    def feature_list(self, value):
        self._feature_list = value
