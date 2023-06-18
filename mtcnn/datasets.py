import os
from typing import Any, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from mtcnn.utils.dataset import construct_image_pyramid
from mtcnn.utils.functional import default_scale_step, random_picker, split_num
from mtcnn.utils.logger import DebugLogger, TqdmLogWriter
from mtcnn.utils.parser import parse_anno_line, parse_raw_anno_line, write_anno_file

logger = DebugLogger(__name__, TqdmLogWriter())


class MTCNNRawDataset(Dataset):
    """
    Dataset for raw folder
    """

    dirname = "raw"
    image_dir_prefix = "images"
    supported_type = ["cascade", "train", "eval", "test"]

    def __init__(self, perfix: str, dataset_type: str = "cascade", transform=None):
        super(MTCNNRawDataset, self).__init__()
        # check dataset_type
        if dataset_type not in self.supported_type:
            raise NotImplementedError("dataset_type must be one of " + str(self.supported_type))

        # set some properties
        self.dir = os.path.join(perfix, self.dirname)
        self.image_dir = os.path.join(self.dir, self.image_dir_prefix)
        self.loader = self.default_loader
        self.transform = (
            transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
            if transform is None
            else transform
        )

        self.annotation_basename = (
            "annotations.txt" if dataset_type == "cascade" else dataset_type + ".txt"
        )
        # read annotations from annotations.txt, line by line
        self.annotations = list()
        with open(os.path.join(self.dir, self.annotation_basename), "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line == "":
                    continue
                # prase a raw annotation line
                image_path, bbox, landmark = parse_raw_anno_line(line)
                self.annotations.append((image_path, bbox, landmark))

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx) -> Any:
        # make it iterabale
        if idx >= len(self):
            raise IndexError("index out of range")

        image_path, bbox, landmark = self.annotations[idx]
        # read image from image_path to PILImage
        img = self.loader(os.path.join(self.image_dir, image_path))

        # do some transform
        if self.transform is not None:
            img = self.transform(img)

        # on defualt transform , the img is torch.Tensor(C, H, W) not normalized
        return img, bbox, landmark

    def pil_loader(self, path: str) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def accimage_loader(self, path: str) -> Any:
        import accimage #type: ignore

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
        generate train, eval, test annotation file from general

        Input:
            perfix: dataset folder perfix
            ratio: train, eval, test segment ratio
        """
        # check parms
        train_ratio, eval_ratio, test_ratio = ratio
        assert train_ratio >= 0 and eval_ratio >= 0 and test_ratio >= 0, "ratio must be positive"
        logger.info("make dataset for raw")

        raw_perfix = os.path.join(perfix, MTCNNRawDataset.dirname)
        # read annotations from annotations.txt, line by line
        raw_annotations = list()
        logger.info("read annotations from" + raw_perfix + "/annotations.txt")
        with open(os.path.join(raw_perfix, "annotations.txt"), "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line == "":
                    continue
                # prase a raw annotation line
                image_path, bbox, landmark = parse_raw_anno_line(line)
                raw_annotations.append((image_path, bbox, landmark))
        total_num = len(raw_annotations)
        logger.info("total annotations num: " + str(total_num))

        # normalize ratio and get select num
        train_ratio = train_ratio / sum(ratio)
        eval_ratio = eval_ratio / sum(ratio)
        test_ratio = test_ratio / sum(ratio)
        train_num, eval_num, test_num = split_num(total_num, (train_ratio, eval_ratio, test_ratio))
        logger.info(
            "train num: "
            + str(train_num)
            + " eval num:"
            + str(eval_num)
            + " test num:"
            + str(test_num)
        )

        # split raw_annotations into train, eval, test
        train_annotations, rest_annotations = random_picker(raw_annotations, train_num)
        eval_annotations, test_annotations = random_picker(rest_annotations, eval_num)

        # write annotations to train, eval, test annotation file
        write_anno_file(os.path.join(raw_perfix, "train.txt"), train_annotations)
        write_anno_file(os.path.join(raw_perfix, "eval.txt"), eval_annotations)
        write_anno_file(os.path.join(raw_perfix, "test.txt"), test_annotations)

        logger.info("make dataset for raw done")


class MTCNNDataset(Dataset):
    """
    Dataset for mtcnn cascade, pnet, rnet, onet's train, eval and test data
    """

    net_type = ["cascade","pnet", "rnet", "onet"]
    task_type = ["train", "eval", "test"]

    image_dir_prefix = "images"

    def __init__(self, perfix: str, net_type: str, task_type: str, transform=None):
        super(MTCNNDataset, self).__init__()
        # check type
        if task_type not in self.task_type:
            raise NotImplementedError("not support task_type for " + task_type)
        if net_type not in self.net_type:
            raise NotImplementedError("not support net_type for " + net_type)

        self.net_type = net_type
        self.task_type = task_type
        self.transform = (
            transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
            if transform is None
            else transform
        )
        self.loader = self.default_loader

        # properties settings for different task
        if self.net_type == "cascade":
            if self.task_type == "train":
                # use pnet train as train entry
                self.dirname = "pnet"
                self.annotation_basename = self.task_type + ".txt"
            else:
                self.dirname = "raw"
                self.annotation_basename = self.task_type + ".txt"
        else:
            self.dirname = self.net_type
            self.annotation_basename = self.task_type + ".txt"

        self.dir = os.path.join(perfix, self.dirname)
        self.image_dir = os.path.join(self.dir, self.image_dir_prefix)

        # read annotations from annotations file, line by line
        self.annotations = list()
        with open(os.path.join(self.dir, self.annotation_basename), "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line == "":
                    continue
                # prase a annotation line
                if self.net_type == "cascade":
                    if self.task_type == "train":
                        image_path, cls_label, bbox, landmark = parse_anno_line(line)
                        self.annotations.append((image_path, cls_label, bbox, landmark))
                    else:
                        # use raw annotation format
                        image_path, bbox, landmark = parse_raw_anno_line(line)
                        self.annotations.append((image_path, bbox, landmark))
                else:
                    # use train type annotation format
                    image_path, cls_label, bbox, landmark = parse_anno_line(line)
                    self.annotations.append((image_path, cls_label, bbox, landmark))

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx) -> Any:
        # make it iterabale
        if idx >= len(self):
            raise IndexError("index out of range")

        if self.net_type == "cascade":
            if self.task_type == "train":
                return self.get_train_type_item(idx)
            else:
                return self.get_other_type_item(idx)
        else:
            return self.get_train_type_item(idx)

    def pil_loader(self, path: str) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def accimage_loader(self, path: str) -> Any:
        import accimage #type: ignore

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

    def get_train_type_item(self, idx: int):
        # if the dataset is not for cascade, some anno transform should be done
        image_path, cls_label, bbox, landmark = self.annotations[idx]
        # load img from image_path
        img = self.loader(os.path.join(self.image_dir, image_path))
        # do image transform
        if self.transform is not None:
            img = self.transform(img)
        # generate type indicator
        cls_type_indicator = torch.tensor([1])  # all data can be used to classification task
        # only pos and part data can be used to bbox task
        bbox_type_indicator = torch.tensor([1]) if cls_label in [1, 2] else torch.tensor([0])
        # only pos data can be used to landmark task
        landmark_type_indicator = torch.tensor([1]) if cls_label == 1 else torch.tensor([0])

        # change cls_label to binary class, only positive sample is target
        cls_label = 1 if cls_label == 1 else 0

        # return image, (cls_label, bbox, landmark), (cls_type_indicator, bbox_type_indicator, landmark_type_indicator)
        return (
            img,
            (cls_label, bbox, landmark),
            (cls_type_indicator, bbox_type_indicator, landmark_type_indicator),
        )
    def get_other_type_item(self, idx: int):
        # if the dataset  is eval or test type, perferom like raw dataset, but we need image pyramid
        image_path, bbox, landmark = self.annotations[idx]
        # load img from image_path
        img = self.loader(os.path.join(self.image_dir, image_path))
        # do image transform
        if self.transform is not None:
            img = self.transform(img)

        # generate image pyramid
        images_pyramid = construct_image_pyramid(img, default_scale_step(0.4, 5))

        # return images[], bbox, landmark
        return images_pyramid, bbox, landmark


