from mtcnn.datasets import MTCNNRawDataset

dataset = MTCNNRawDataset("dataset", "train")

if __name__ == "__main__":
    img, bbox, landmark = dataset[0]
    print("[img]:", img)
    print("[bbox]:", bbox)
    print("[landmark]:", landmark)
