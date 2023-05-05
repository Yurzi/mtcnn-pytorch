import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

def get_optimizer(type_name:str, model:nn.Module, **kwargs) -> Optimizer:
    if type_name == 'adam':
        optimizer = optim.Adam(model.parameters(), **kwargs)
    elif type_name == 'sdg':
        optimizer = optim.SGD(model.parameters(), **kwargs)
    else:
        raise NotImplementedError
    return optimizer


def get_scheduler(type_name:str, optimizer:Optimizer, **kwargs) -> LRScheduler:
    if type_name == 'steplr':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    else:
        raise NotImplementedError
    return lr_scheduler

