from collections import OrderedDict
import torch.nn as nn
import torch
import os

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler, StepLR
from torch.optim import Optimizer
from tqdm import tqdm

from .logger import ConsoleLogWriter, DebugLogger, Logger
from typing import Callable

def get_lr(optimizer: Optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class Trainer():
    """
    A Class collects all properties for train
    """

    def __init__(self, model:nn.Module, train_loader: DataLoader,optimizer:Optimizer,loss_fn:Callable, logger: Logger | None = None) -> None:
        self.device = None # setup when call setup() or set_device()
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.lr_scheduler = None # setup when call setup() and set_lr_schduler()
        self.loss_fn = loss_fn
        self.acc_fn = None

        self.logger = logger if logger is not None else DebugLogger(__name__, ConsoleLogWriter())

        self.step = 0
        self.epoch = 0

        self.max_epoch = -1
        # for train
        self.total_loss = 0
        self.total_accuracy = 0
        self.total_cls_accuracy = 0 
        self.total_bbox_accuracy = 0
        self.total_ldmk_accuracy = 0
        # for test
        self.test_loader = None

        # a flag to check if trainer is all setup
        self.is_setup = False


    def train(self) -> None:
        """
        only do train for one epoch
        """
        if not self.is_setup:
            self.logger.error("trainer must be setup, call step first")
            return

        # setup a pragress bar
        self.logger.info("Training...")
        epoch_step = len(self.train_loader)
        pbar = tqdm(total=epoch_step,desc=f'Epoch {self.epoch + 1}/{self.max_epoch}',mininterval=0.3)

        # shift model to train mode
        self.model.train()
        for _, (image, target, type_indicator) in enumerate(self.train_loader):
            self.step += 1
            # shift data to device
            image = image.to(self.device)
            for item in target:
                item.to(self.device)
            for item in type_indicator:
                item.to(self.device)

            # forward
            output = self.model(image)
            loss = self.loss_fn(output, target, type_indicator)

            self.total_loss += loss.item()
            if self.acc_fn is not None:
                general_acc, cls_acc, bbox_acc, ldmk_acc = self.acc_fn(output, target, type_indicator)
                self.total_accuracy += general_acc
                self.total_cls_accuracy += cls_acc
                self.total_bbox_accuracy += bbox_acc
                self.total_ldmk_accuracy += ldmk_acc
            else:
                general_acc, cls_acc, bbox_acc, ldmk_acc = -1, -1, -1, -1


            # backward
            self.optimizer.zero_grad()
            loss.backward()

            # update
            self.optimizer.step()
            last_lr = get_lr(self.optimizer)
            self.lr_scheduler.step() #type: ignore

            # update progress bar
            avg_loss = self.total_loss / self.step
            postfix_dict = OrderedDict({'loss': loss.item(),'avg_loss': avg_loss,'lr': last_lr})
            if self.acc_fn is not None:
                avg_acc = self.total_accuracy / self.step
                avg_cls_acc = self.total_cls_accuracy / self.step
                avg_bbox_acc = self.total_bbox_accuracy / self.step
                avg_ldmk_acc = self.total_ldmk_accuracy / self.step
                postfix_dict = OrderedDict({
                    'loss':loss.item(),
                    'avg_loss': avg_loss,
                    'lr': last_lr,
                    'avg_acc': avg_acc,
                    'avg_cls_acc': avg_cls_acc,
                    'avg_bbox_acc': avg_bbox_acc,
                    'avg_ldmk_acc': avg_ldmk_acc
                })
            pbar.set_postfix(postfix_dict)
            pbar.update(1)


        # close progress bar
        self.epoch += 1
        pbar.close()
        print("\n")

    def test(self) -> None:
        """
        test model
        """
        if self.test_loader is None:
            self.logger.warn("test_loader is not defined, skip test")
            return
        if not self.is_setup:
            self.logger.error("trainer must be setup, call step first")
            return
        if self.acc_fn is None:
            self.logger.warn("acc_fn is not defined, accuracy will not be caculated")

        self.logger.info("Testing...")
        epoch_step = len(self.test_loader)
        pbar = tqdm(total=epoch_step,desc=f'Test {self.epoch}/{self.max_epoch}',mininterval=0.3)
        test_step = 0
        total_test_loss = 0
        total_test_accuracy = 0
        total_test_cls_accuracy = 0
        total_test_bbox_accuracy = 0
        total_test_ldmk_accuracy = 0
        # shift model to eval mode
        self.model.eval()

        with torch.no_grad():
            for _, (image, target, type_indicator) in enumerate(self.test_loader):
                test_step += 1
                # shift data to device
                image = image.to(self.device)
                for item in target:
                    item.to(self.device)
                for item in type_indicator:
                    item.to(self.device)

                # forward
                output = self.model(image)
                loss = self.loss_fn(output, target, type_indicator)

                total_test_loss += loss.item()
                if self.acc_fn is not None:
                    general_acc, cls_acc, bbox_acc, ldmk_acc = self.acc_fn(output, target, type_indicator)
                    total_test_accuracy += general_acc
                    total_test_cls_accuracy += cls_acc
                    total_test_bbox_accuracy += bbox_acc
                    total_test_ldmk_accuracy += ldmk_acc
                else:
                    general_acc, cls_acc, bbox_acc, ldmk_acc = -1, -1, -1, -1
                # update progress bar
                avg_loss = total_test_loss / test_step
                postfix_dict = OrderedDict({'loss': loss.item(),'avg_loss': avg_loss})
                if self.acc_fn is not None:
                    avg_acc = total_test_accuracy / test_step
                    avg_cls_acc = total_test_cls_accuracy / test_step
                    avg_bbox_acc = total_test_bbox_accuracy / test_step
                    avg_ldmk_acc = total_test_ldmk_accuracy / test_step
                    postfix_dict = OrderedDict({
                        'loss':loss.item(),
                        'avg_loss': avg_loss,
                        'avg_acc': avg_acc,
                        'avg_cls_acc': avg_cls_acc,
                        'avg_bbox_acc': avg_bbox_acc,
                        'avg_ldmk_acc': avg_ldmk_acc
                    })
                pbar.set_postfix(postfix_dict)
                pbar.update(1)
        # test end
        pbar.close() 

        avg_loss = total_test_loss / test_step
        avg_acc = total_test_accuracy / test_step
        avg_cls_acc = total_test_cls_accuracy / test_step
        avg_bbox_acc = total_test_bbox_accuracy / test_step
        avg_ldmk_acc = total_test_ldmk_accuracy / test_step
        self.logger.info(f"Test {self.epoch}/{self.max_epoch} loss: {avg_loss} acc: {avg_acc} cls_acc: {avg_cls_acc} bbox_acc: {avg_bbox_acc} ldmk_acc: {avg_ldmk_acc}\n")



    def setup(self):
        # if you do not want to set more things call this
        # check device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # check lr_scheduler
        if self.lr_scheduler is None:
            self.lr_scheduler = StepLR(self.optimizer, step_size=1, gamma=0.5)

        # train_loader batch size must greater than 1
        if self.train_loader.batch_size <= 1: # type: ignore
            self.logger.fatal("train_loader batch size must greater than 1, due to BatchNormalize")
            return self

        if not self.is_setup:
            self.is_setup = True

        return self

    def set_max_epoch(self, max_epoch: int):
        if max_epoch > 0:
            self.max_epoch = max_epoch
        return self

    def set_logger(self, logger:Logger):
        if logger is not None:
            self.logger = logger

        return self

    def set_device(self,device:torch.device):
        if device is not None:
            self.device = device

        return self

    def set_lr_schduler(self, lr_scheduler: LRScheduler):
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler

        return self

    def set_loss_fn(self, loss_fn: Callable):
        if loss_fn is not None:
            self.loss_fn = loss_fn

        return self

    def set_acc_fn(self, acc_fn: Callable):
        if acc_fn is not None:
            self.acc_fn = acc_fn
        return self

    def set_test_dataloader(self, test_loader: DataLoader):
        if test_loader is not None:
            self.test_loader = test_loader

        return self

    def load_state(self, path: str):
        # load state(weight) from path
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])

        is_all = checkpoint['is_all']
        if is_all and self.is_setup:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler']) # type:ignore
            self.step = checkpoint['step']
            self.epoch = checkpoint['epoch']

        else:
            self.logger.warn("Trainer is not setup fully, some state may not be loaded")
        return self
            
    def save_state(self, dir: str, net_name: str, is_all:bool = False):
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)

        # sava_state(weight) to path, if is_all will save lr_scheduler and other things
        checkpoint = {}
        checkpoint['model'] = self.model.state_dict()
        checkpoint['is_all'] = is_all
        if is_all and self.is_setup:
            checkpoint['optimizer'] = self.optimizer.state_dict()
            checkpoint['lr_scheduler'] = self.lr_scheduler.state_dict() # type:ignore
            checkpoint['step'] = self.step
            checkpoint['epoch'] = self.epoch
        else:
            self.logger .warn("Trainer is not setup fully, some state may not be saved")

        loss = 0
        if self.step > 0:
            loss = self.total_loss / self.step

        checkpoint_filename = f"{net_name}-step{self.step}-loss-{loss}.pt"
        checkpoint_path = os.path.join(dir, checkpoint_filename)

        torch.save(checkpoint, checkpoint_path)
        # save_last
        checkpoint_last = f"{net_name}-last.pt"
        checkpoint_last_path = os.path.join(dir, checkpoint_last)
        torch.save(checkpoint, checkpoint_last_path)

        return self
