# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.

from typing import List, Dict, Text
import torch
import torch.distributed as dist


class BaseNormalizer:
    def __init__(
        self,
        shape=(),
        local_rank: int = 0,
        world_size: int = 0,
    ):
        assert local_rank >= 0
        assert world_size >= 0
        self.world_size = world_size
        self.local_rank = local_rank
        self.shape = shape
        self.distributed = self.world_size > 1
        self.device = f'cuda:{self.local_rank}' if self.distributed else 'cpu'
        self.reset()

    def reset(self) -> None:  # noqa: E704
        ...

    def update(self, **args) -> None:  # noqa: E704
        ...

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


class Normalizer(BaseNormalizer):
    """Normalize data to have targeted mean and STD.

    Note: This tends to produce more outliers than the running mean and STD method.
    """

    def __init__(
        self,
        shape=(),
        target_mean: float = 0.0,  # default zero mean
        target_std: float = 0.5,  # default unit variance
        window_size: int = 10000,
        local_rank: int = 0,
        world_size: int = 0,
    ):
        assert window_size >= 1000
        assert target_std > 0
        super().__init__(shape=shape, local_rank=local_rank, world_size=world_size)

        self.window_size = window_size
        self.target_mean = target_mean
        self.target_std = target_std

    def reset(self) -> None:
        self.gain = torch.ones(self.shape, dtype=torch.float32, device=self.device)
        self.bias = torch.zeros(self.shape, dtype=torch.float32, device=self.device)

        # store values internally
        self.delta = []

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        self.delta.extend(x.tolist())

        if self.window_size > 0 and len(self.delta) > self.window_size:
            self.delta = self.delta[-self.window_size :]

        self._update_gain_and_bias()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return self.gain.detach().to(x.device) * x + self.bias.detach().to(x.device)

    def state_dict(self) -> Dict[Text, torch.Tensor]:
        gains = self.gather_tensor(self.gain)
        bias = self.gather_tensor(self.bias)

        if self.distributed:
            gains = gains.mean(0)
            bias = bias.mean(0)

        return {'gain': gains, 'bias': bias}

    def load_state_dict(self, state_dict: Dict[Text, torch.Tensor]):
        self.gain = state_dict['gain']
        self.bias = state_dict['bias']

    def _update_gain_and_bias(self) -> None:
        if len(self.delta) < 1:
            return

        delta = torch.tensor(self.delta).to(self.device)
        mean = delta.mean(dim=0)
        std = delta.std(dim=0)

        self.gain = self.target_std / std
        self.bias = self.target_mean - self.gain * mean


class RunningNormalizer(BaseNormalizer):
    """Cumulate running mean and variance to normalize data to have zero mean and unit STD."""

    def reset(self):
        self.mean = torch.zeros(self.shape, dtype=torch.float32, device=self.device)
        self.var = torch.ones(self.shape, dtype=torch.float32, device=self.device)
        self.count = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        n = x.shape[0]

        if n > 1:
            # update count and moments
            x = x.to(self.device)
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0)

            self.count += n
            delta = batch_mean - self.mean
            self.mean += delta * n / self.count
            m_a = self.var * (self.count - n)
            m_b = batch_var * n
            M2 = m_a + m_b + torch.square(delta) * self.count * n / self.count
            self.var = M2 / self.count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.count == 0:
            return x

        var_nozero = torch.where(self.var <= 0, 1e-8, self.var)
        return (x - self.mean.to(x.device).detach()) / torch.sqrt(var_nozero.to(x.device).detach())

    def state_dict(self) -> Dict[Text, torch.Tensor]:
        return {'mean': self.mean, 'var': self.var, 'count': self.count}

    def load_state_dict(self, state_dict: Dict[Text, torch.Tensor]) -> None:
        self.mean = state_dict['mean']
        self.var = state_dict['var']
        self.count = state_dict['count']
