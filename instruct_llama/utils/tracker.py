# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


from typing import Dict

import torch
import torch.distributed as dist


class BaseTracker:
    def __init__(
        self,
        local_rank: int = 0,
        world_size: int = 0,
    ):
        assert local_rank >= 0
        assert world_size >= 0
        self.world_size = world_size
        self.local_rank = local_rank

        self.distributed = self.world_size > 1
        self.device = f'cuda:{self.local_rank}' if self.distributed else 'cpu'
        self.reset()

    def reset(self) -> None:  # noqa: E704
        ...

    def update(self, **args) -> None:  # noqa: E704
        ...

    def get_dict(self, reset: bool) -> Dict:
        return {}

    def to_tensor(self, data) -> torch.Tensor:
        return torch.tensor(data).to(self.device)

    def gather_tensor(self, data: torch.Tensor) -> torch.Tensor:
        if self.distributed:
            all_values = [torch.empty_like(data).to(self.local_rank) for _ in range(self.world_size)]
            dist.all_gather(all_values, data)
            cat_function = torch.cat if data.dim() > 0 else torch.stack
            return cat_function(all_values, dim=0)
        else:
            return data


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

        losses = self.gather_tensor(losses)
        num_accurate = self.gather_tensor(num_accurate)
        num_samples = self.gather_tensor(num_samples)

        if reset:
            self.reset()

        return {
            'loss': losses.mean().item(),
            'accuracy': (num_accurate.sum() / num_samples.sum()).item(),
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

        losses = self.gather_tensor(losses)
        chosen_rewards = self.gather_tensor(chosen_rewards)
        rejected_rewards = self.gather_tensor(rejected_rewards)
        num_accurate = self.gather_tensor(num_accurate)
        num_samples = self.gather_tensor(num_samples)

        if reset:
            self.reset()

        return {
            'loss': losses.mean().item(),
            'accuracy': (num_accurate.sum() / num_samples.sum()).item(),
            'chosen_reward_mean': chosen_rewards.mean().item(),
            'chosen_reward_std': chosen_rewards.std().item(),
            'rejected_reward_mean': rejected_rewards.mean().item(),
            'rejected_reward_std': rejected_rewards.std().item(),
            'reward_gap': (chosen_rewards.mean() - rejected_rewards.mean()).item(),
        }
