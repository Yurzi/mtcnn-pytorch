from mtcnn.datasets import MTCNNRawDataset
from mtcnn.utils.dataset import gen_train_set_frow_raw


def test_raw_dataset_split():
    MTCNNRawDataset.make_dataset("dataset", (0.8, 0.1, 0.1))


def test_train_set_gen():
    raw_dataset = MTCNNRawDataset("dataset")
    gen_train_set_frow_raw(raw_dataset, "dataset/pnet", (12, 12))


if __name__ == "__main__":
    # test_raw_dataset_split()
    test_train_set_gen()
