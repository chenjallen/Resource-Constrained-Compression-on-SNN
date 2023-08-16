import warnings
import math
import logging
import random
import os
import numpy as np
from termcolor import colored

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

import networks


def seed_all(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_logger(filename, rank=-1, verbosity=1, name='log'):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        fmt='[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )
    color_formatter = logging.Formatter(fmt=colored('[%(asctime)s]', 'green') +
                                            colored('[%(filename)s][line:%(lineno)d]', 'yellow') +
                                            '[%(levelname)s] %(message)s',
                                        datefmt='%Y-%m-%d %H:%M:%S'
                                        )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity] if rank in [-1, 0] else logging.WARNING)
    logger.propagate = False

    fh = logging.FileHandler(filename, 'w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(color_formatter)
    logger.addHandler(sh)

    return logger


def get_network(args):
    net_name = str.lower(args.net_name)
    if net_name == 'cifar10net':
        model = networks.Cifar10Net()
    else:
        raise NotImplementedError

    return model


def best_initialization(best_state_dict, model):
    model.load_state_dict(best_state_dict, strict=True)

    return model


EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)


class CosineLRFtScheduler(_LRScheduler):
    def __init__(self, optimizer, ft_epochs_max, eta_min=0, last_epoch=-1, verbose=False):
        self.ft_epochs_max = ft_epochs_max
        self.eta_min = eta_min
        self.mc = True
        self.last_mc_epoch = last_epoch
        super(CosineLRFtScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if not self.mc:
            if self.last_epoch == 0:
                return [base_lr * 10 for base_lr in self.base_lrs]
            elif (self.last_epoch - 1 - self.ft_epochs_max) % (2 * self.ft_epochs_max) == 0:
                return [group['lr'] + (base_lr * 10 - self.eta_min) *
                        (1 - math.cos(math.pi / self.ft_epochs_max)) / 2
                        for base_lr, group in
                        zip(self.base_lrs, self.optimizer.param_groups)]
            return [(1 + math.cos(math.pi * self.last_epoch / self.ft_epochs_max)) /
                    (1 + math.cos(math.pi * (self.last_epoch - 1) / self.ft_epochs_max)) *
                    (group['lr'] - self.eta_min) + self.eta_min
                    for group in self.optimizer.param_groups]
        else:
            return self.base_lrs

    def step(self, mc=True, epoch=None):
        if self.mc != mc:
            print(mc)
            self.mc = mc
            if not self.mc:
                self.last_mc_epoch = self.last_epoch
                self.last_epoch = -1
            else:
                self.last_epoch = self.last_mc_epoch

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
                print(values)
            else:
                warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values = self._get_closed_form_lr()
                else:
                    values = self.get_lr()

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group['lr'] = lr
            self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
