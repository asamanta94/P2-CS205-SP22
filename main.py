from os import path

from dataset import Dataset

DATASETS_DIR = "dataset"
DATA_SMALL_FILE = "CS205_SP_2022_SMALLtestdata__64.txt"
DATA_LARGE_FILE = "CS205_SP_2022_Largetestdata__51.txt"

SMALL_DATASET_PATH = path.join(DATASETS_DIR, DATA_SMALL_FILE)
LARGE_DATASET_PATH = path.join(DATASETS_DIR, DATA_LARGE_FILE)


if __name__ == '__main__':
    data = Dataset(SMALL_DATASET_PATH)
    data.read_file()
