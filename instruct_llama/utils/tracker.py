# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


from typing import Dict

import torch
import torch.distributed as dist


class BaseTracker:
    def __init__(
        self,
        distributed: bool = False,
        rank: int = 0,
    ):
        assert rank >= 0
        self.distributed = distributed
        self.rank = rank
        self.reset()

    def reset(self) -> None:
        ...

    def update(self, **args) -> None:
        ...

    def get_dict(self, reset: bool) -> Dict:
        return {}

    def to_tensor(self, data) -> torch.Tensor:
        return torch.tensor(data).to(f'cuda:{self.rank}' if self.distributed else 'cpu')

    def reduce_tensor(self, data) -> None:
        if self.distributed:
            dist.all_reduce(data, op=dist.ReduceOp.SUM)


class StatsTracker(BaseTracker):
    """Tracker for LLM model during pre-training or fine-tuning stages"""

    def reset(self) -> None:
        self.losses = []
        self.num_accurate = 0
        self.num_samples = 0

    def update(self, losses: torch.Tensor, num_accurate: int, num_samples: int) -> None:
        assert len(losses.shape) == 1
        self.losses.extend(losses.tolist())
        self.num_accurate += num_accurate
        self.num_samples += num_samples

    def get_dict(self, reset: bool = False) -> Dict:
        if len(self.losses) == 0:
            return {}

        losses = self.to_tensor(self.losses)
        num_accurate = self.to_tensor(self.num_accurate)
        num_samples = self.to_tensor(self.num_samples)

        self.reduce_tensor(losses)
        self.reduce_tensor(num_accurate)
        self.reduce_tensor(num_samples)

        if reset:
            self.reset()

        return {
            'loss': losses.mean().item(),
            'accuracy': (num_accurate / num_samples).item(),
            'perplexity': torch.exp(losses).mean().item(),
        }


class RMStatsTracker(BaseTracker):
    """Tracker for reward model"""

    def reset(self) -> None:
        self.losses = []
        self.chosen_rewards = []
        self.rejected_rewards = []
        self.num_accurate = 0
        self.num_samples = 0

    def update(self, losses: torch.Tensor, chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> None:
        assert len(losses.shape) == 1
        assert len(chosen_rewards.shape) == 1
        assert len(rejected_rewards.shape) == 1

        self.losses.extend(losses.tolist())
        self.chosen_rewards.extend(chosen_rewards.tolist())
        self.rejected_rewards.extend(rejected_rewards.tolist())
        self.num_accurate += (chosen_rewards > rejected_rewards).sum().item()
        self.num_samples += len(chosen_rewards)

    def get_dict(self, reset: bool = False) -> Dict:
        if len(self.losses) == 0:
            return {}

        losses = self.to_tensor(self.losses)
        chosen_rewards = self.to_tensor(self.chosen_rewards)
        rejected_rewards = self.to_tensor(self.rejected_rewards)
        num_accurate = self.to_tensor(self.num_accurate)
        num_samples = self.to_tensor(self.num_samples)

        self.reduce_tensor(losses)
        self.reduce_tensor(chosen_rewards)
        self.reduce_tensor(rejected_rewards)
        self.reduce_tensor(num_accurate)
        self.reduce_tensor(num_samples)

        if reset:
            self.reset()

        return {
            'loss': losses.mean().item(),
            'accuracy': (num_accurate / num_samples).item(),
            'chosen_reward': chosen_rewards.mean().item(),
            'rejected_reward': rejected_rewards.mean().item(),
            'reward_gap': (chosen_rewards.mean() - rejected_rewards.mean()).item(),
        }
