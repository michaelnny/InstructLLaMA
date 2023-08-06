"""A simple prompt environment module that the RL agent can interact with."""
import torch


# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from instruct_llama.model import Transformer
from instruct_llama.tokenizer import Tokenizer

from instruct_llama.utils import build_prompt_completion


DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and correct. If you don't know the answer to a question, please don't share false information."


class PromptEnv:
    def __init__(
        self,
        tokenizer: Tokenizer,
        max_steps: 1024,
        reward_model: Transformer,
        runtime_dtype: torch.dtype = torch.bfloat16,
        runtime_device: torch.device = 'cpu',
        sys_prompt: str = None,
    ) -> None:
        assert 0 < max_steps < reward_model.params.max_seq_len

        # freeze all layers
        for n, p in reward_model.named_parameters():
            p.requires_grad = False

        self.device = runtime_device
        self.reward_model = reward_model.to(device=self.device, dtype=runtime_dtype)
        self.reward_model.eval()

        self.max_steps = max_steps
        self.tokenizer = tokenizer
        self.action_dim = tokenizer.vocab_size
        self.terminal_action = tokenizer.eos_id
        self.token_ids = None
        self.done = True
        self.t = 0

        self.default_prompt = [
            {
                'role': 'system',
                'content': sys_prompt if sys_prompt is not None and isinstance(sys_prompt, str) else DEFAULT_SYSTEM_PROMPT,
            }
        ]

    def observation(self) -> torch.Tensor:
        assert self.token_ids is not None and len(self.token_ids) > 0

        return torch.tensor(self.token_ids, dtype=torch.long)

    def reset(self, prompt: str):
        assert prompt is not None and isinstance(prompt, str) and len(prompt) > 0

        dialog = self.default_prompt + [{'role': 'user', 'content': prompt}]

        prompt_tokens, _ = build_prompt_completion(dialog, self.tokenizer)
        assert prompt_tokens is not None

        self.token_ids = prompt_tokens
        self.done = False
        self.t = 0

        return self.observation()

    def step(self, action: int):
        if not isinstance(action, int):
            raise ValueError(f'Expect action to be int type, got {type(action)}')
        if action < 0 or action > self.action_dim - 1:
            raise ValueError(f'Action out of range, expect in the range of [0, {self.action_dim -1}], got {action}')
        if self.done:
            raise RuntimeError('Call reset() before step()')

        reward = 0.0
        self.token_ids.append(action)
        self.t += 1
        if action == self.terminal_action:
            self.done = True
            reward = self.compute_estimated_reward()

        return self.observation(), reward, self.done

    @torch.inference_mode()
    def compute_estimated_reward(self) -> float:
        if self.token_ids is None or len(self.token_ids) <= 1:
            return 0.0
        elif self.token_ids[-1] != self.terminal_action:
            return 0.0

        # [1, sequence_length]
        tokens = self.observation().to(device=self.device).unsqueeze(0)

        # [1, sequence_length, 1]
        output = self.reward_model(tokens)

        # [1]
        reward = output.squeeze()[-1]

        return reward.cpu().item()
