import argparse
import torch
import os

from utils.config import get_config
from utils.trainer import Trainer
from utils.evaluation import MTCNNMultiTaskAcc
from schedulers import get_scheduler, get_optimizer
from torch.utils.data import DataLoader
from datasets import MTCNNDataset
from backbones.net import PNet
from backbones.functional import MTCNNMultiSourceSLoss


def main(args):
    # get config
    cfg = get_config(args.config)
    # net_name
    net_name = "pnet" if cfg.net_name is None else cfg.net_name
    # get dataloader
    dataset = MTCNNDataset(cfg.dataset_dir, net_name, "train")
    test_dataset = MTCNNDataset(cfg.dataset_dir, net_name, "test")
    train_dataloader = DataLoader(dataset, num_workers=cfg.num_workers, batch_size=cfg.batch_size, shuffle=cfg.shuffle)
    test_dataloader = DataLoader(test_dataset, num_workers=cfg.num_workers, batch_size=cfg.batch_size, shuffle=cfg.shuffle)
    # get model
    model = PNet()
    # get_device
    if cfg.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    # get optimizer
    optimizer = get_optimizer(cfg.optimizer_type, model, **cfg.optimizer_params)

    # get scheduler
    scheduler = get_scheduler(cfg.scheduler_type, optimizer, **cfg.scheduler_params)

    # get loss_fn
    cls_weight, bbox_weight, landmark_weight = cfg.task_weight
    loss_fn = MTCNNMultiSourceSLoss(cls_weight, bbox_weight, landmark_weight, cfg.ohem_rate)
    # get acc_fn
    acc_fn = MTCNNMultiTaskAcc(cls_weight, bbox_weight, landmark_weight, cfg.acc_iou_threshold, cfg.acc_ldmk_threshold)

    # get trainer
    trainer = Trainer(model, train_dataloader, optimizer, loss_fn)
    trainer.set_device(device).set_lr_schduler(scheduler).set_max_epoch(cfg.epoch)
    trainer.set_test_dataloader(test_dataloader).set_acc_fn(acc_fn)
    trainer.setup()

    if args.resume:
        weight_filename = net_name + "-" + cfg.weight
        trainer.load_state(os.path.join(cfg.weight_dir, weight_filename))
    
    epoch = trainer.epoch
    for epoch in range(epoch, cfg.epoch):
        trainer.train()
        trainer.test()
        if cfg.save_epoch != -1 and epoch % cfg.save_epoch == 0:
            trainer.save_state(cfg.output, net_name, is_all=cfg.save_all)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MTCNN PNet train script")
    parser.add_argument("--config", required=True,help="config file path")
    parser.add_argument("--resume", help="resume train task", action="store_true")

    args = parser.parse_args()
    if args.config is None:
        args.config = "default_p"

    main(args)

