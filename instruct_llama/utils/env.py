from typing import Tuple, List, Callable
import os
import logging
import torch
from torch.distributions import Categorical

from instruct_llama.models.model_lora import Transformer
from instruct_llama.utils.custom_dataset import PromptOnlyDataset
from instruct_llama.utils.normalizer import Normalizer

logger = logging.getLogger(__name__)


class PromptEnv:
    """A simple prompt-completion RL environment.

    It's like a bandit environment, where the agent only take a single step and the episode is over.
    However, unlike traditional RL environment, instead of taking a single action in the step(), here we take the full completion tokens as a step.
    """

    def __init__(
        self,
        prompt_dataset: PromptOnlyDataset,
        reward_model: Transformer,
        sft_model: Transformer,
        normalize_reward: bool = True,
        normalizer_ckpt: str = None,
        clip_reward_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        device: torch.device = 'cpu',
    ):
        self.device = device
        self.dataset = prompt_dataset
        self.reward_model = reward_model.eval().to(device)
        self.sft_model = sft_model.eval().to(device)
        self.reward_model.disable_cache()
        self.sft_model.disable_cache()

        self.normalize_reward = normalize_reward
        self.clip_reward_fn = clip_reward_fn

        self.normalizer = Normalizer()
        if normalizer_ckpt and os.path.exists(normalizer_ckpt):
            rm_norm_state = torch.load(normalizer_ckpt)
            self.normalizer.load_state_dict(rm_norm_state)
            logger.info(f'Loaded RM statistics normalizer checkpoint from {normalizer_ckpt!r}')
            del rm_norm_state

    def reset(self, batch_size: int = 1) -> List[List[int]]:
        """Returns a batch of randomly sampled prompts"""
        return self.dataset.sample(batch_size)

    @torch.no_grad()
    def step(self, prompt_completions: torch.Tensor, terminal_steps: torch.Tensor) -> torch.Tensor:
        """Returns the normalized and clipped environment rewards for the given prompt-completion pairs"""

        assert len(prompt_completions) == len(terminal_steps)
        assert len(prompt_completions.shape) == 2
        assert len(terminal_steps.shape) == 1

        # compute the environment reward for the given completions
        prompt_completions = prompt_completions.to(self.device)
        terminal_steps = terminal_steps.to(self.device)
        outputs = self.reward_model(prompt_completions).squeeze(-1)  # [batch_size, seq_length]

        # get rewards for terminal step, where sequence ends with EOS token, or reached maximum seq_length
        env_rewards = torch.gather(outputs, dim=1, index=terminal_steps.unsqueeze(-1)).squeeze(1)  # [batch_size]
        if self.normalize_reward:
            self.normalizer.update(env_rewards)
            env_rewards = self.normalizer.normalize(env_rewards)

        if self.clip_reward_fn is not None:
            env_rewards = self.clip_reward_fn(env_rewards)

        return env_rewards

    @torch.no_grad()
    def compute_kl_penalties(self, tokens: torch.Tensor, actions: torch.Tensor, logprobs: torch.Tensor) -> torch.Tensor:
        """Returns the pre-token (completion only) KL penalties for the given prompt-completion pairs"""

        assert len(tokens) == len(logprobs) == len(actions)
        assert len(tokens.shape) == len(logprobs.shape) == len(actions.shape) == 2

        tokens = tokens.to(self.device)
        actions = actions.to(self.device)
        logprobs = logprobs.to(self.device)

        outputs = self.sft_model(tokens)  # [batch_size, seq_len, vocab_size]

        sft_logprobs = torch.gather(torch.log_softmax(outputs, dim=-1), dim=2, index=actions.unsqueeze(2)).squeeze(2)
        kl = logprobs - sft_logprobs

        return kl
