import os
from typing import List, Tuple

import torch


def prase_raw_anno_line(line: str):
    """
    parse a line of annotation under raw folder, return a tuple of (image_path, bbox, landmark)
    """
    items = line.split(" ")

    # image path
    image_path = ""
    bbox = torch.empty(4)
    landmark = torch.empty(10)

    if len(items) == 1:
        image_path = items[0]

    # bbox
    if len(items) == 5:
        image_path = items[0]
        bbox = torch.tensor([float(x) for x in items[1:5]])

    # landmark
    if len(items) == 15:
        image_path = items[0]
        bbox = torch.tensor([float(x) for x in items[1:5]])
        landmark = torch.tensor([float(x) for x in items[5:15]])

    return image_path, bbox, landmark


def prase_anno_line(line: str):
    """
    prase a line of annotation under other folder, return a tuple of (image_path, cls_label, bbox, landmark)
    """

    items = line.split(" ")

    assert len(items) == 15, "annotation line must have 15 parms"

    image_path = items[0]
    cls_label = int(items[1])
    bbox = torch.tensor([float(x) for x in items[2:6]])
    landmark = torch.tensor([float(x) for x in items[6:16]])

    return image_path, cls_label, bbox, landmark


def write_anno_file(path: os.PathLike | str, annotations: List[Tuple]) -> None:
    # string format is a file path
    if isinstance(path, str):
        path = os.fspath(path)

    with open(path, "w") as f:
        for anno in annotations:
            write_buf = list()
            for item in anno:
                if isinstance(item, torch.Tensor):
                    item = item.tolist()
                    write_buf.extend(item)
                elif isinstance(item, str):
                    write_buf.append(item)
                else:
                    raise TypeError("bad annotations type")

            line = " ".join([str(x) for x in write_buf])
            f.write(line + "\n")
