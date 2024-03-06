# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.

from typing import Dict, Text
import torch
import torch.distributed as dist


class BaseNormalizer:
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


class RunningNormalizer(BaseNormalizer):
    """Cumulate running mean and variance to normalize data to have zero mean and unit STD."""

    def reset(self):
        self.mean = torch.zeros((1), dtype=torch.float32, device=self.device)
        self.var = torch.ones((1), dtype=torch.float32, device=self.device)
        self.count = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        flat_x = x.flatten()
        n = flat_x.shape[0]

        if n > 1:
            # update count and moments
            flat_x = flat_x.to(self.device)
            batch_mean = flat_x.mean()
            batch_var = flat_x.var()

            self.count += n
            delta = batch_mean - self.mean
            self.mean += delta * n / self.count
            m_a = self.var * (self.count - n)
            m_b = batch_var * n
            M2 = m_a + m_b + torch.square(delta) * self.count * n / self.count
            self.var = M2 / self.count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.count <= 4:
            return x

        var_nozero = torch.where(self.var <= 0, 1e-8, self.var)
        return (x - self.mean.to(x.device).detach()) / torch.sqrt(var_nozero.to(x.device).detach())

    def state_dict(self) -> Dict[Text, torch.Tensor]:
        return {'mean': self.mean, 'var': self.var, 'count': self.count}

    def load_state_dict(self, state_dict: Dict[Text, torch.Tensor]) -> None:
        self.mean = state_dict['mean']
        self.var = state_dict['var']
        self.count = state_dict['count']


class Normalizer(BaseNormalizer):
    """Normalize data to have targeted mean and STD.

    Note: This tends to produce more outliers than the running mean and STD method.
    """

    def __init__(
        self,
        target_mean: float = 0.0,  # default zero mean
        target_std: float = 1.0,  # default unit STD
        window_size: int = 10000,
        local_rank: int = 0,
        world_size: int = 0,
    ):
        assert window_size >= 1000
        assert target_std > 0
        super().__init__(local_rank=local_rank, world_size=world_size)

        self.window_size = window_size
        self.target_mean = target_mean
        self.target_std = target_std

    def reset(self) -> None:
        self.gain = torch.ones((1), dtype=torch.float32, device=self.device)
        self.bias = torch.zeros((1), dtype=torch.float32, device=self.device)

        # store values internally
        self.delta = []

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        self.delta.extend(x.flatten().tolist())

        if self.window_size > 0 and len(self.delta) > self.window_size:
            self.delta = self.delta[-self.window_size :]

        self._update_gain_and_bias()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return self.gain.detach().to(x.device) * x + self.bias.detach().to(x.device)

    def state_dict(self) -> Dict[Text, torch.Tensor]:
        gains = self.gather_tensor(self.gain)
        bias = self.gather_tensor(self.bias)

        if self.distributed:
            gains = gains.mean()
            bias = bias.mean()

        return {'gain': gains, 'bias': bias}

    def load_state_dict(self, state_dict: Dict[Text, torch.Tensor]):
        self.gain = state_dict['gain']
        self.bias = state_dict['bias']

    def _update_gain_and_bias(self) -> None:
        if len(self.delta) < 4:
            return

        delta = torch.tensor(self.delta).to(self.device)
        mean = delta.mean()
        std = delta.std()

        self.gain = self.target_std / std
        self.bias = self.target_mean - self.gain * mean
