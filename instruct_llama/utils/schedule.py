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

    def __init__(self, optimizer, min_lr, max_lr, warmup_steps, max_decay_steps, last_epoch=-1, verbose=False) -> None:
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.max_decay_steps = max_decay_steps

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warm-up phase
            return [self.min_lr + (self.max_lr - self.min_lr) * self.last_epoch / self.warmup_steps] * len(
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
