"""Merge LoRA fine-tunned checkpoint and pretrained checkpoint into a single checkpoint file"""

import os
import torch
import torch.nn as nn


# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from instruct_llama.model import Transformer, ModelArgs, supported_model_types
from instruct_llama.lora import lora


def del_lora_state_dict(model: nn.Module):
    base_model_dict = model.state_dict()
    key_to_delete = [k for k in base_model_dict if "lora_" in k]
    for del_key in key_to_delete:
        del base_model_dict[del_key]
    return base_model_dict


def lora_model_lookup(checkpoint: dict) -> int:
    """Returns the LoRA rank from the adapter checkpoint."""
    return checkpoint["layers.0.attention.wq.lora_B"].shape[1]


def merge_lora_checkpoint(
    model_type: str,
    lora_ckpt_path: str,
    pretrained_ckpt_path: str,
    save_path: str,
) -> None:
    """Merges LoRA weights with pretrained base model.

    Args:
        model_type: The llama-2 model type, supports 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'.
        lora_ckpt_path: Path to the checkpoint with trained LoRA weights, which are the output of
            `finetune_lora.py`.
        pretrained_ckpt_path: The pretrained checkpoint used in side the `finetune_lora.py` when start the fine-tuning.
        save_path: target path to save the merged stat_dict.
    """

    assert model_type in supported_model_types

    if not os.path.exists(lora_ckpt_path):
        raise ValueError(f"LoRA checkpoint file {lora_ckpt_path} does not exist, aborting...")
    if not os.path.exists(pretrained_ckpt_path):
        raise ValueError(f"Pretrained checkpoint file {pretrained_ckpt_path} does not exist, aborting...")

    if os.path.exists(save_path):
        print(f"The checkpoint file {save_path} already exists, aborting...")
        return

    print("Loading model checkpoints ...")

    pretrained_checkpoint = torch.load(pretrained_ckpt_path)
    lora_checkpoint = torch.load(lora_ckpt_path)

    # find the rank from LoRA checkpoint
    rank = lora_model_lookup(lora_checkpoint)

    assert rank == 32

    with lora(r=rank, alpha=16, dropout=0.0, enabled=True):
        model_args = ModelArgs.from_model_type(model_type)

        model = Transformer(model_args)

        # 1. Load the pretrained weights
        model.load_state_dict(pretrained_checkpoint, strict=False)
        # 2. Load the fine-tuned lora weights
        model.load_state_dict(lora_checkpoint, strict=False)

    model.eval()
    merged_model_dict = del_lora_state_dict(model)
    print("Saving LoRA to base model weights ...")
    torch.save(merged_model_dict, save_path)
    print(f"Merged model state dict saved at {save_path}")


if __name__ == "__main__":
    merge_lora_checkpoint(
        model_type="7B",
        lora_ckpt_path="./checkpoints/finetune_lora/lora_7B-iter-1600.pth",
        pretrained_ckpt_path="./checkpoints/llama-2/llama-2-7b/consolidated.pth",
        save_path="./checkpoints/7b-finetune/iter-1600-merged.pth",
    )
