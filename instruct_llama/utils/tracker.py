# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


from typing import Dict
import numpy as np

import torch
import torch.distributed as dist


class StatsTracker:
    """Tracker for LLM model during fine-tune"""

    def __init__(self, distributed: bool = False, rank=0):
        self.distributed = distributed
        self.rank = rank

        self.reset()

    def update(self, loss: torch.Tensor, num_accurate: int, num_samples: int):
        metrics = self.metrics

        metrics[0] += loss.item()  # sum up batch loss
        metrics[1] += np.exp(loss.item())  # sum up perplexity
        metrics[2] += 1  # increase number of micro batches
        metrics[3] += num_accurate  # sum up number of accurate prediction tokens
        metrics[4] += num_samples  # sum up number of tokens

        self.c += 1

    def reset(self) -> None:
        self.metrics = torch.zeros(5).to(f'cuda:{self.rank}' if self.distributed else 'cuda')
        self.c = 0

    def get_dict(self) -> Dict:
        if self.c == 0:
            return {}

        metrics = self.metrics

        if self.distributed:
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

        loss = metrics[0] / metrics[2]
        perplexity = metrics[1] / metrics[2]
        accuracy = 100 * metrics[3] / metrics[4]

        return {'loss': loss.item(), 'accuracy': accuracy.item(), 'perplexity': perplexity.item()}


class RMStatsTracker:
    """Tracker for reward model"""

    def __init__(self, distributed: bool = False, rank=0):
        self.distributed = distributed
        self.rank = rank

        self.reset()

    def update(self, loss: torch.Tensor, num_accurate: int, num_samples: int, reward_best: float, reward_worst: float):
        metrics = self.metrics

        metrics[0] += loss.item()  # sum up batch loss
        metrics[1] += 1  # increase number of samples
        metrics[2] += num_accurate  # sum up number of accurate prediction tokens
        metrics[3] += num_samples  # sum up number of responses or combinations
        metrics[4] += reward_best  # sum up best reward
        metrics[5] += reward_worst  # sum up worst reward

        self.c += 1

    def reset(self) -> None:
        self.metrics = torch.zeros(6).to(f'cuda:{self.rank}' if self.distributed else 'cuda')
        self.c = 0

    def get_dict(self) -> Dict:
        if self.c == 0:
            return {}

        metrics = self.metrics

        if self.distributed:
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

        loss_mean = metrics[0] / metrics[1]
        accuracy = 100 * metrics[2] / metrics[3]
        preferred_reward_mean = metrics[4] / metrics[1]
        rejected_reward_mean = metrics[5] / metrics[1]

        return {
            'loss': loss_mean.item(),
            'accuracy': accuracy.item(),
            'preferred_reward': preferred_reward_mean.item(),
            'rejected_reward': rejected_reward_mean.item(),
            'reward_gap': (preferred_reward_mean - rejected_reward_mean).item(),
        }
