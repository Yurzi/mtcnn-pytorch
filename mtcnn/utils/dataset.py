import os
from typing import Generator, List, Tuple

import torch
from torchvision import transforms

from mtcnn.utils.harverster import harverst_train_set_frow_raw


def prase_raw_anno_line(line: str):
    """
    parse a line of annotation under raw folder, return a tuple of (image_path, bbox, landmark)
    """
    items = line.split(" ")

    # image path
    image_path = ""
    bbox = None
    landmark = None

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
                if item is None:
                    continue
                if isinstance(item, torch.Tensor):
                    item = item.tolist()
                    write_buf.extend(item)
                elif isinstance(item, str):
                    write_buf.append(item)
                elif isinstance(item, int):
                    write_buf.append(str(item))
                else:
                    raise TypeError("bad annotations type")

            line = " ".join([str(x) for x in write_buf])
            f.write(line + "\n")


def construct_image_pyramid(
    original: torch.Tensor, step_fn: Generator[float, None, None]
) -> List[torch.Tensor]:
    """
    construct image pyramid from original image
    Input:
        original: original image
        step_fn: a generator that generate a scale factor

    Output:
        list of image [biggest size -> smallset size]
    """
    res: List[torch.Tensor] = list()
    ori_width = original.shape[2]
    ori_height = original.shape[1]

    for scale_factor in step_fn:
        transform = transforms.Compose(
            [
                transforms.Resize((ori_height * scale_factor, ori_width * scale_factor)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        res.append(transform(original))

    return res


def gen_train_set_frow_raw(raw_dataset, dir: str, crop_size: Tuple[int, int]):
    # check dir exist or create it
    image_path_perfix = "images"
    annotation_basename = "annotations.txt"

    image_dir = os.path.join(dir, image_path_perfix)
    annotation_path = os.path.join(dir, annotation_basename)

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    annotations = list()
    counter = 0
    for image, cls_label, bbox, landmark in harverst_train_set_frow_raw(raw_dataset, crop_size):
        image_path = str(counter) + ".jpg"
        counter += 1

        annotations.append((image_path, cls_label, bbox, landmark))
        pil_imge = transforms.ToPILImage()(image)

        pil_imge.save(os.path.join(image_dir, image_path))

    # write annotations file
    write_anno_file(annotation_path, annotations)
