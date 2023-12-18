# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


import math
import torch


class LinearWarmupLRScheduler(torch.optim.lr_scheduler.LRScheduler):
    """Follows the LoRA paper"""

    def __init__(self, optimizer, init_lr, max_lr, warmup_steps, last_epoch=-1, verbose=False) -> None:
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch <= self.warmup_steps:
            # Warm-up phase
            return [self.init_lr + (self.max_lr - self.init_lr) * self.last_epoch / self.warmup_steps] * len(
                self.optimizer.param_groups
            )
        else:
            return [self.max_lr] * len(self.optimizer.param_groups)


class CosineDecayWithWarmupLRScheduler(torch.optim.lr_scheduler.LRScheduler):
    """Follows the GPT-3 paper"""

    def __init__(
        self, optimizer, init_lr, max_lr, min_lr, warmup_steps, max_decay_steps, last_epoch=-1, verbose=False
    ) -> None:
        """
        Args:
            init_lr: initial learning rate
            max_lr: maximum learning rate at the end of the linear warm up phase
            min_lr: minimum learning rate at the end of the cosine annealing phase phase
            warmup_steps: number of steps to linear warm the learning rate from init_lr to max_lr
            max_decay_steps: number of steps to apply cosine annealing to the learning rate from max_lr to min_lr
        """

        self.init_lr = init_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.max_decay_steps = max_decay_steps

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warm-up phase
            return [self.init_lr + (self.max_lr - self.init_lr) * self.last_epoch / self.warmup_steps] * len(
                self.optimizer.param_groups
            )
        elif self.last_epoch >= self.warmup_steps and self.last_epoch < self.max_decay_steps:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_steps) / (self.max_decay_steps - self.warmup_steps)
            return [self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))] * len(
                self.optimizer.param_groups
            )
        else:
            return [self.min_lr] * len(self.optimizer.param_groups)
