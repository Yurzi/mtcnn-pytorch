import argparse

from torchvision import os

from mtcnn.datasets import MTCNNRawDataset
from mtcnn.utils.config import get_config
from mtcnn.utils.dataset import generate_train_set_from_raw, get_mean_anchor_size
from mtcnn.utils.logger import ConsoleLogWriter, DebugLogger

logger = DebugLogger(__name__, ConsoleLogWriter())


def check_raw_dataset(dir: str) -> bool:
    train_anno_path = os.path.join(dir, "train.txt")
    eval_anno_path = os.path.join(dir, "eval.txt")
    test_anno_path = os.path.join(dir, "test.txt")

    return (
        os.path.exists(train_anno_path)
        and os.path.exists(eval_anno_path)
        and os.path.exists(test_anno_path)
    )


def main(args):
    # get config
    cfg = get_config(args.config)
    if args.path is not None:
        cfg.dataset_dir = args.path
    else:
        if cfg.dataset_dir is None or len(cfg.dataset_dir) == 0:
            raise ValueError("dataset_dir is not specified")
        if cfg.net_name is None or len(cfg.net_name) == 0:
            raise ValueError("net_name is not specified")

    # generate dataset
    # check if raw dataset is complated
    raw_dataset_dir = os.path.join(cfg.dataset_dir, "raw")
    if not check_raw_dataset(raw_dataset_dir):
        logger.info("process raw dataset...")
        MTCNNRawDataset.make_dataset(cfg.dataset_dir, cfg.dataset_split)

    raw_dataset = MTCNNRawDataset(cfg.dataset_dir)
    anchor_size = int(1.15 * get_mean_anchor_size(raw_dataset))

    raw_train_dataset = MTCNNRawDataset(cfg.dataset_dir, dataset_type="train")
    raw_eval_dataset = MTCNNRawDataset(cfg.dataset_dir, dataset_type="eval")
    raw_test_dataset = MTCNNRawDataset(cfg.dataset_dir, dataset_type="test")
    # generate train set
    # for one net
    logger.info(f"process {cfg.net_name} dataset...")
    logger.info(f"[img_size]: {cfg.img_size}")
    net_perfix = os.path.join(cfg.dataset_dir, cfg.net_name)
    generate_train_set_from_raw(
        raw_train_dataset, net_perfix, "train" ,cfg.img_size, anchor_size, config=cfg, reset = True
    )
    generate_train_set_from_raw(
        raw_eval_dataset, net_perfix, "eval" ,cfg.img_size, anchor_size, config=cfg
    )
    generate_train_set_from_raw(
        raw_test_dataset, net_perfix, "test" ,cfg.img_size, anchor_size, config=cfg
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="a tool to generate dataset for mtcnn from raw")
    parser.add_argument("--config", required=True, type=str, help="config file name under configs folder")
    parser.add_argument("--path", type=str, help="path to dataset dir")
    args = parser.parse_args()
    if args.config is None:
        args.config = "default"

    main(args)
