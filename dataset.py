
class Dataset(object):
    """
    Class to read a dataset file.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.file_object = None
        self.data = None

    def read_file(self):
        """


        :return:
        """
        if self.file_object is None:
            self.file_object = open(self.file_path, "r")
            lines = self.file_object.readlines()

            self.data = [line.split() for line in lines]
