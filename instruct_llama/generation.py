# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import os

import time
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F

# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from instruct_llama.model import ModelArgs, Transformer
from instruct_llama.tokenizer import Tokenizer
from instruct_llama.utils.prompt_builder import (
    Message,
    Dialog,
    ChatPrediction,
    CompletionPrediction,
    build_prompt_completion,
)


class Llama:
    @staticmethod
    def build(
        ckpt_path: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        device: str,
        seed: int = 1,
    ) -> 'Llama':
        if not os.path.exists(ckpt_path):
            raise ValueError(f'Checkpoint file {ckpt_path!r} does not exist, aborting ...')
        ckpt_dir = os.path.dirname(ckpt_path)

        params_path = os.path.join(ckpt_dir, 'params.json')
        if not os.path.exists(params_path):
            raise ValueError(f'Can not find model metadata file {params_path!r}, aborting ...')

        print(f'Starting to load model checkpoints {ckpt_path!r} ...')

        torch.manual_seed(seed)
        torch.set_default_device(device)
        torch.set_default_dtype(torch.float16)

        t0 = time.time()
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        with open(params_path, 'r') as f:
            params = json.loads(f.read())

        try:
            del params['max_seq_len']
            del params['max_batch_size']
            del params['use_cache']
        except Exception:
            pass

        model_args: ModelArgs = ModelArgs(
            **params,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            use_cache=True,
        )

        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)

        print(f'Model checkpoint loaded in {time.time() - t0:.2f} seconds')

        print(f'Starting to load tokenizer checkpoint {tokenizer_path!r} ...')
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size

        for params in model.parameters():
            params.requires_grad = False
        model = model.eval()

        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device='cuda')
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device='cuda')
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device='cuda')
        input_text_mask = tokens != pad_id
        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction='none',
                    ignore_index=pad_id,
                )
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            eos_reached |= (~input_text_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id)
            prev_pos = cur_pos
            if all(eos_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        if logprobs:
            return [
                {
                    'generation': self.tokenizer.decode(t),
                    'tokens': [self.tokenizer.decode(x) for x in t],
                    'logprobs': logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [{'generation': self.tokenizer.decode(t)} for t in generation_tokens]

    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = []
        for dialog in dialogs:
            current_prompt_tokens, _ = build_prompt_completion(dialog, self.tokenizer)

            prompt_tokens.append(current_prompt_tokens)

        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        if logprobs:
            return [
                {
                    'generation': {
                        'role': 'assistant',
                        'content': self.tokenizer.decode(t),
                    },
                    'tokens': [self.tokenizer.decode(x) for x in t],
                    'logprobs': logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [{'generation': {'role': 'assistant', 'content': self.tokenizer.decode(t)}} for t in generation_tokens]


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
