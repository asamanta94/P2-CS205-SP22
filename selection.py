FORWARD_SELECTION = 0
BACKWARD_SELECTION = 1


class Selection(object):

    def __init__(self, way, data_set):
        self.way = way
        self.data_set = data_set
        self.n_features = len(self.data_set[0][1:])
        self.visited = set()

    def get_features(self, features):
        children = []
        for i in range(self.n_features):
            if i in features:
                continue
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

        if self.way == FORWARD_SELECTION:
            features.append(())
            self.visited.add(())
            while len(features) > 0:
                top = features[0]
                print(top)
                features.remove(top)

                children = self.get_features(top)
                self.visited.add(top)
                # Get the greediest child

                for child in children:
                    if child not in features:
                        features.append(child)
