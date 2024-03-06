# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""Merge LoRA fine-tunned checkpoint and pretrained checkpoint into a single checkpoint file"""

import os
import shutil
import json
import torch
import torch.nn as nn


# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from instruct_llama.models.model_lora import Transformer, LoraModelArgs


def get_clean_state_dict(model: nn.Module):
    """Clean up lora weights and return cleaned state dict."""
    model_dict = model.state_dict()
    key_to_delete = [k for k in model_dict if 'lora_' in k]
    for del_key in key_to_delete:
        del model_dict[del_key]
    return model_dict


def convert_model_to_dtype(model: torch.nn.Module, dtype) -> None:
    for name, module in model.named_modules():
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'token_embeddings' in name:
            if hasattr(module, 'weight'):
                if module.weight.dtype != dtype:
                    module = module.to(dtype)
        else:
            module = module.to(dtype)


def merge_lora_checkpoint(
    base_ckpt_path: str,
    lora_ckpt_path: str,
    save_path: str,
    dtype=torch.bfloat16,
) -> None:
    """Merges LoRA weights with pretrained base model.

    Args:
        base_ckpt_path: The base checkpoint (like pre-trained or fine-tuned) used for training with lora.
        lora_ckpt_path: Path to the checkpoint with trained LoRA weights, which are the output of
            `finetune_lora.py`.
        save_path: target path to save the merged stat_dict.
        dtype: save model weights and biases in the given target data type, default torch.bfloat16.
    """

    if not os.path.exists(lora_ckpt_path):
        raise ValueError(f'LoRA checkpoint file {lora_ckpt_path!r} does not exist, aborting ...')
    if not os.path.exists(base_ckpt_path):
        raise ValueError(f'Pretrained checkpoint dir {base_ckpt_path!r} does not exist, aborting ...')

    if os.path.exists(save_path):
        print(f'The checkpoint file {save_path!r} already exists, aborting ...')
        return

    # try to get lora_params.json file based on the lora_ckpt_path
    lora_dir = os.path.dirname(lora_ckpt_path)

    # Create the path to the JSON file based on the directory
    params_path = os.path.join(lora_dir, 'params.json')
    if not os.path.exists(params_path):
        print(f'Can not find model params file {params_path!r}, aborting ...')
        return

    output_dir = os.path.dirname(save_path)
    if not os.path.exists(output_dir):
        # Create the output directory if necessary
        os.makedirs(output_dir, mode=0o777, exist_ok=True)

    print(f'Loading base model checkpoints {base_ckpt_path!r}...')
    base_checkpoint = torch.load(base_ckpt_path)

    print(f'Loading LoRA model checkpoints {lora_ckpt_path!r}...')
    lora_checkpoint = torch.load(lora_ckpt_path)

    with open(params_path, 'r') as f:
        meta_params = json.load(f)

        del meta_params['quant_4bit']
        del meta_params['quant_lora_4bit']

    model_args = LoraModelArgs(
        **meta_params,
        # No quantization during merge weights
        quant_4bit=False,
        quant_lora_4bit=False,
    )

    model = Transformer(model_args)

    # 1. Load the pretrained weights
    model.load_state_dict(base_checkpoint, strict=False)

    # 2. Load the fine-tuned lora weights
    model.load_state_dict(lora_checkpoint, strict=False)

    # 3. merge LoRA weights, which was handled inside the LoRALinear.train() method
    model.eval()

    # 4. optional, convert to bfloat16
    convert_model_to_dtype(model, dtype)

    # 5. Remove LoRA parameters from the model state
    state_dict = get_clean_state_dict(model)

    print(f'Saving merged model weights to {save_path!r} ...')
    torch.save(state_dict, save_path)

    meta_file = os.path.join(output_dir, 'params.json')
    if not os.path.exists(meta_file):
        del_keys = ('lora', 'quant', 'dropout', 'use_cache', 'gradient_checkpointing')
        meta = model.params.dict()
        meta = {k: v for k, v in meta.items() if all([n not in k for n in del_keys])}

        print(f'Saving model metadata to {meta_file!r} ...')
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)


if __name__ == '__main__':
    # fine-tuned model
    merge_lora_checkpoint(
        base_ckpt_path='/home/michael/models/meta_llama2/llama-2-7b/consolidated.pth',
        lora_ckpt_path='./checkpoints/sft_lora/lora_7B-steps-6000.pth',
        save_path='./checkpoints/sft/7B-steps-6000.pth',
    )

    # PPO models
    merge_lora_checkpoint(
        base_ckpt_path='./checkpoints/sft/7B-steps-6000.pth',
        lora_ckpt_path='./checkpoints/rlhf_lora/lora_7B-epoch-60.pth',
        save_path='./checkpoints/rlhf/7B-epoch-60.pth',
    )
