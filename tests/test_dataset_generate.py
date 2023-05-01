from mtcnn.datasets import MTCNNRawDataset


def test_raw_dataset_split():
    MTCNNRawDataset.make_dataset("dataset", (0.8, 0.1, 0.1))


if __name__ == "__main__":
    test_raw_dataset_split()
