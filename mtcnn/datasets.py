import os
from typing import Any, Tuple

from PIL import Image
from torch.utils.data import Dataset
from utils.dataset import prase_anno_line, prase_raw_anno_line, write_anno_file
from utils.functional import random_picker, split_num
from utils.logger import ConsoleLogWriter, Logger

logger = Logger(ConsoleLogWriter())


class MTCNNRawDataset(Dataset):
    """
    Dataset for raw folder
    """

    dirname = "raw"
    image_dir_prefix = "images"

    def __init__(self, perfix: str, transform=None):
        super(MTCNNRawDataset, self).__init__()
        self.dir = os.path.join(perfix, self.dirname)
        self.image_dir = os.path.join(self.dir, self.image_dir_prefix)
        self.loader = self.default_loader
        self.transform = transform
        # read annotations from annotations.txt, line by line
        self.annotations = list()
        with open(os.path.join(self.dir, "annotations.txt"), "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line == "":
                    continue
                # prase a raw annotation line
                image_path, bbox, landmark = prase_raw_anno_line(line)
                self.annotations.append((image_path, bbox, landmark))

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index) -> Any:
        image_path, bbox, landmark = self.annotations[index]
        # read image from image_path to PILImage
        img = self.loader(os.path.join(self.image_dir, image_path))

        # do some transform
        if self.transform is not None:
            img = self.transform(img)

        return img, bbox, landmark

    def pil_loader(self, path: str) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def accimage_loader(self, path: str) -> Any:
        import accimage

        try:
            return accimage.Image(path)
        except OSError:
            return self.pil_loader(path)

    def default_loader(self, path: str) -> Any:
        from torchvision import get_image_backend

        if get_image_backend() == "accimage":
            return self.accimage_loader(path)
        else:
            return self.pil_loader(path)

    @staticmethod
    def make_dataset(perfix: str, ratio: Tuple[float, float, float]) -> None:
        """
        generate train, eval, test annotation file from raw

        Input:
            perfix: dataset folder perfix
            ratio: train, eval, test segment ratio
        """
        # check parms
        train_ratio, eval_ratio, test_ratio = ratio
        assert train_ratio >= 0 and eval_ratio >= 0 and test_ratio >= 0, "ratio must be positive"
        logger({"INFO": "make dataset for raw"})

        raw_perfix = os.path.join(perfix, MTCNNRawDataset.dirname)
        # read annotations from annotations.txt, line by line
        raw_annotations = list()
        logger({"INFO": "read annotations from " + raw_perfix + "/annotations.txt"})
        with open(os.path.join(raw_perfix, "annotations.txt"), "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line == "":
                    continue
                # prase a raw annotation line
                image_path, bbox, landmark = prase_raw_anno_line(line)
                raw_annotations.append((image_path, bbox, landmark))
        total_num = len(raw_annotations)
        logger({"INFO": "total annotations num: " + str(total_num)})

        # normalize ratio and get select num
        train_ratio = train_ratio / sum(ratio)
        eval_ratio = eval_ratio / sum(ratio)
        test_ratio = test_ratio / sum(ratio)
        train_num, eval_num, test_num = split_num(total_num, (train_ratio, eval_ratio, test_ratio))
        logger(
            {
                "INFO": "train num: "
                + str(train_num)
                + " eval num:"
                + str(eval_num)
                + " test num:"
                + str(test_num)
            }
        )

        # split raw_annotations into train, eval, test
        train_annotations, rest_annotations = random_picker(raw_annotations, train_num)
        eval_annotations, test_annotations = random_picker(rest_annotations, eval_num)

        # write annotations to train, eval, test annotation file
        write_anno_file(os.path.join(perfix, "train.txt"), train_annotations)
        write_anno_file(os.path.join(perfix, "eval.txt"), eval_annotations)
        write_anno_file(os.path.join(perfix, "test.txt"), test_annotations)

        logger({"INFO": "make dataset for raw done"})


class MTCNNDataset(Dataset):
    """
    Dataset for mtcnn train(pnet, rnet, onet), eval and test data
    """
