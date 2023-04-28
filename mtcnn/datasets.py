import os
from typing import Any

from PIL import Image
from torch.utils.data import Dataset
from utils.dataset import prase_raw_anno_line


class MTCNNRawDataset(Dataset):
    """
    Dataset for raw folder
    """

    dirname = "raw"
    image_dir_prefix = "imges"

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
