"""A simple prompt environment that RL agent can interact with."""
from typing import Iterable, Tuple, List
import random
import math
import itertools
import pickle
import numpy as np
import torch


# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from instruct_llama.model import Transformer
from instruct_llama.tokenizer import Tokenizer

from instruct_llama.utils import build_prompt_completion


class PromptEnv:
    def __init__(
        self,
        data_sources: Iterable[str],
        max_seq_len: 1024,
        reward_model: Transformer,
        runtime_dtype: torch.dtype = torch.bfloat16,
        runtime_device: torch.device = 'cpu',
    ) -> None:
        assert 0 < max_seq_len <= reward_model.params.max_seq_len

        # same logic as the fine-tune datasets
        self.max_seq_len = max_seq_len

        self.data = []
        seq_length_stats = []  # track statistics

        # Load datasets
        for source in data_sources:
            samples = pickle.load(open(source, 'rb'))
            for sample in samples:
                # we only need prompt for the environment, as the completion should be given by the RL agent
                x = sample['prompt_tokens']
                seq_length = len(x)
                if seq_length <= self.max_seq_len:
                    self.data.append(x)
                    seq_length_stats.append(seq_length)

        assert len(self.data) > 0

        self.total_num_tokens = sum(seq_length_stats)
        self.seq_length_stats = {
            'min': int(np.min(seq_length_stats)),
            'max': int(np.max(seq_length_stats)),
            'mean': int(np.mean(seq_length_stats)),
            'std': int(np.std(seq_length_stats)),
        }
        random.shuffle(self.data)

        print(f'Number of sample prompts: {len(self.data)}')
        print(f'Sample prompt length statistics: {self.seq_length_stats}')

        # freeze all layers
        for n, p in reward_model.named_parameters():
            p.requires_grad = False

        self.device = runtime_device
        self.reward_model = reward_model.to(dtype=runtime_dtype, device=self.device)
        self.reward_model.eval()

        self.prompt_tokens = None
        self.terminal_steps = None
        self.done = True
        self.c_episodes = 0

    def reset(self) -> List[int]:
        """Returns a random prompt."""
        self.done = False
        prompt_tokens = random.choice(self.data)

        self.prompt_tokens = prompt_tokens

        return self.prompt_tokens

    def step(self, completion_tokens: List[int]) -> float:
        """Takes in a completion tokens, returns the estimated reward."""
        if self.done:
            raise RuntimeError('Call reset() before make_action()')

        self.c_episodes += 1
        self.done = True

        rewards = self.compute_rewards(completion_tokens)
        return rewards

    @torch.inference_mode()
    def compute_rewards(self, completion_tokens: List[int]) -> float:
        tokens = self.prompt_tokens + completion_tokens
        tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)

        # [1, sequence_length, 1]
        outputs = self.reward_model(tokens.unsqueeze(0))

        # get reward for terminal time step
        rewards = outputs.squeeze()[-1]

        return rewards.cpu().item()


class PromptBatchedEnv:
    def __init__(
        self,
        data_sources: Iterable[str],
        max_seq_len: 1024,
        batch_size: int,
        pad_id: int,
        reward_model: Transformer,
        runtime_dtype: torch.dtype = torch.bfloat16,
        runtime_device: torch.device = 'cpu',
    ) -> None:
        assert 0 < max_seq_len <= reward_model.params.max_seq_len
        assert batch_size >= 1

        # same logic as the fine-tune datasets
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

        self.data = []
        seq_length_stats = []  # track statistics

        # Load datasets
        for source in data_sources:
            samples = pickle.load(open(source, 'rb'))
            for sample in samples:
                # we only need prompt for the environment, as the completion should be given by the RL agent
                x = sample['prompt_tokens']
                seq_length = len(x)
                if seq_length <= self.max_seq_len:
                    self.data.append(x)
                    seq_length_stats.append(seq_length)

        assert len(self.data) > 0

        self.total_num_tokens = sum(seq_length_stats)
        self.seq_length_stats = {
            'min': int(np.min(seq_length_stats)),
            'max': int(np.max(seq_length_stats)),
            'mean': int(np.mean(seq_length_stats)),
            'std': int(np.std(seq_length_stats)),
        }
        random.shuffle(self.data)

        print(f'Number of sample prompts: {len(self.data)}')
        print(f'Sample prompt length statistics: {self.seq_length_stats}')

        # freeze all layers
        for n, p in reward_model.named_parameters():
            p.requires_grad = False

        self.device = runtime_device
        self.reward_model = reward_model.to(dtype=runtime_dtype, device=self.device)
        self.reward_model.eval()
        self.batch_size = batch_size

        self.prompt_tokens = None
        self.terminal_steps = None
        self.done = True
        self.c_episodes = 0

    def set_batch_size(self, v: int) -> None:
        assert v >= 1
        self.batch_size = v

        self.prompt_tokens = None
        self.terminal_steps = None
        self.done = True

    def reset(self) -> List[List[int]]:
        """Returns a list of random prompts."""
        self.done = False
        prompt_tokens = random.choices(self.data, k=self.batch_size)

        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= self.max_seq_len

        self.prompt_tokens = prompt_tokens

        return self.prompt_tokens

    def step(self, completion_tokens: List[List[int]]) -> Tuple[List[float], bool]:
        """Takes in a list of completion tokens, returns the estimated reward and a boolean flag indicate episode is terminated."""
        if self.done:
            raise RuntimeError('Call reset() before step()')

        assert len(completion_tokens) == len(self.prompt_tokens)

        batch_size = len(self.prompt_tokens)

        self.c_episodes += len(self.prompt_tokens)
        self.done = True

        max_seq_len = max(len(prompt) + len(completion) for prompt, completion in zip(self.prompt_tokens, completion_tokens))

        tokens = torch.full((batch_size, max_seq_len), self.pad_id, dtype=torch.long)

        # record the terminal index of the completion, often referred to as the terminal time step in RL
        terminal_steps = torch.zeros((batch_size, 1), dtype=torch.long)

        for i, p_tk, c_tk in enumerate(zip(self.prompt_tokens, completion_tokens)):
            tokens[i, : len(p_tk) + len(c_tk)] = torch.concat((p_tk, c_tk), dim=0).type(torch.long)
            terminal_steps[i] = len(p_tk) + len(c_tk) - 1  # minus 1 because indexing starts from zero

        rewards = self.compute_rewardS(tokens, terminal_steps)

        return rewards, self.done

    @torch.inference_mode()
    def compute_rewardS(self, batched_tokens: torch.Tensor, terminal_steps: torch.Tensor) -> List[float]:
        # [batch_size, sequence_length]
        outputs = self.reward_model(batched_tokens).squeeze(-1)

        # get reward for terminal time step, where the sequence ends without counting the padding tokens
        rewards = torch.gather(outputs, dim=1, index=terminal_steps).squeeze(1)  # [batch_size]

        return rewards.cpu().list()
