from mtcnn.datasets import MTCNNRawDataset
from mtcnn.utils.dataset import generate_train_set_from_raw, get_mean_anchor_size
from mtcnn.utils.harverster import RandomHarvester


def test_raw_dataset_split():
    MTCNNRawDataset.make_dataset("dataset", (0.8, 0.1, 0.1))


def test_train_set_gen():
    raw_dataset = MTCNNRawDataset("dataset")
    generate_train_set_from_raw(raw_dataset, "dataset/pnet", (12, 12), get_mean_anchor_size)


if __name__ == "__main__":
    # test_raw_dataset_split()
    test_train_set_gen()
