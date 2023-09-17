"""Merge LoRA fine-tunned checkpoint and pretrained checkpoint into a single checkpoint file"""

import os
import shutil
import torch
import torch.nn as nn


# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from instruct_llama.model import Transformer, RewardModel, ModelArgs, supported_model_types
from instruct_llama.utils import lora


def get_clean_state_dict(model: nn.Module):
    """Clean up lora weights and return cleaned state dict."""
    model_dict = model.state_dict()
    key_to_delete = [k for k in model_dict if 'lora_' in k]
    for del_key in key_to_delete:
        del model_dict[del_key]
    return model_dict


def lora_model_lookup(checkpoint: dict) -> int:
    """Returns the LoRA rank from the adapter checkpoint."""
    return checkpoint['layers.0.attention.wq.lora_B'].shape[1]


def merge_lora_checkpoint(
    model_type: str,
    lora_ckpt_path: str,
    base_ckpt_dir: str,
    save_path: str,
    lora_alpha: int = 32,
    dtype: torch.dtype = torch.bfloat16,
    is_reward_model: bool = False,
) -> None:
    """Merges LoRA weights with pretrained base model.

    Args:
        model_type: The llama-2 model type, supports 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'.
        lora_ckpt_path: Path to the checkpoint with trained LoRA weights, which are the output of
            `finetune_lora.py`.
        base_ckpt_dir: The base checkpoint (like pre-trained or fine-tuned) used for training with lora.
        save_path: target path to save the merged stat_dict.
        lora_alpha: the lora alpha value used during fine-tuning, default 32.
        dtype: save model weights and biases in the given target data type, default torch.bfloat16.
    """

    assert model_type in supported_model_types

    if not os.path.exists(lora_ckpt_path):
        raise ValueError(f'LoRA checkpoint file {lora_ckpt_path} does not exist, aborting...')
    if not os.path.exists(base_ckpt_dir):
        raise ValueError(f'Pretrained checkpoint dir {base_ckpt_dir} does not exist, aborting...')

    if os.path.exists(save_path):
        print(f'The checkpoint file {save_path} already exists, aborting...')
        return

    output_dir = os.path.dirname(save_path)
    if not os.path.exists(output_dir):
        # Create the output directory if necessary
        os.makedirs(output_dir, mode=0o777, exist_ok=True)

    print('Loading model checkpoints ...')

    # try to find and load pre-trained and lora checkpoints
    checkpoints = sorted(Path(base_ckpt_dir).glob('*.pth'))
    assert len(checkpoints) == 1, f'no checkpoint files found in {base_ckpt_dir}'
    pretrained_ckpt_file = checkpoints[0]

    pretrained_checkpoint = torch.load(pretrained_ckpt_file)
    lora_checkpoint = torch.load(lora_ckpt_path)

    # find the rank from LoRA checkpoint
    rank = lora_model_lookup(lora_checkpoint)

    with lora(r=rank, alpha=lora_alpha, dropout=0.0, enabled=True):
        model_args = ModelArgs.from_model_type(model_type)

        if is_reward_model:
            model = RewardModel(model_args)
        else:
            model = Transformer(model_args)

        # 1. Load the pretrained weights
        model.load_state_dict(pretrained_checkpoint, strict=False)
        # 2. Load the fine-tuned lora weights
        model.load_state_dict(lora_checkpoint, strict=False)

    # convert to target dtype
    model.to(dtype=dtype)
    model.eval()

    state_dict = get_clean_state_dict(model)

    print(f'Saving merged model weights to {save_path} ...')
    torch.save(state_dict, save_path)

    print(f'Copying params.json to {output_dir}...')
    shutil.copy(os.path.join(base_ckpt_dir, 'params.json'), output_dir)


if __name__ == '__main__':
    # # fine-tuned model
    # merge_lora_checkpoint(
    #     model_type='7B',
    #     lora_ckpt_path='./checkpoints/train_sft_lora/lora_7B-iter-2000.pth',
    #     base_ckpt_dir='./meta_checkpoints/llama-2/llama-2-7b/',
    #     save_path='./checkpoints/7b-sft/iter-2000-merged.pth',
    # )

    # # RM model
    # merge_lora_checkpoint(
    #     model_type='7B',
    #     lora_ckpt_path='./checkpoints/train_rm_lora/lora_7B-iter-2500.pth',
    #     base_ckpt_dir='./checkpoints/7b-sft/',
    #     save_path='./checkpoints/7b-rm/iter-2500-merged.pth',
    #     is_reward_model=True
    # )

    # PPO model
    merge_lora_checkpoint(
        model_type='7B',
        lora_ckpt_path='./checkpoints/train_ppo_lora/lora_7B_policy-epoch-60.pth',
        base_ckpt_dir='./checkpoints/7b-sft/',
        save_path='./checkpoints/7b-ppo/policy-epoch-60-merged.pth',
    )
