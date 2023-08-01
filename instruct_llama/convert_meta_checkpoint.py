import json
import gc
import shutil
from typing import Dict, Any

import torch
from tqdm import tqdm


# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from instruct_llama.model import Transformer, ModelArgs

"""
Sample usage:

```bash
python -m scripts.convert_checkpoint -h

python -m scripts.convert_checkpoint converted
```
"""


def convert_state_dict(state_dict: Dict[str, torch.Tensor], dtype: torch.dtype = torch.bfloat16) -> Dict[str, torch.Tensor]:
    """
    Swap some keys from Meta's weights to support our model.
    """
    converted = {}

    for k in state_dict.keys():
        # skip this as we don't it in our model
        if k == 'rope.freqs':
            continue

        our_k = k

        # map Meta's key to our key
        if k == 'tok_embeddings.weight':
            our_k = 'token_embeddings.weight'
        elif k == 'norm.weight':
            our_k = 'post_norm.weight'
        elif k == 'output.weight':
            our_k = 'lm_head.weight'

        converted[our_k] = state_dict[k].to(dtype)

    return converted


params_key_mapping = {
    'dim': 'hidden_size',
    'n_heads': 'num_attn_heads',
    'n_layers': 'num_layers',
}


shard_dims = {
    'lm_head.weight': 0,
    'token_embeddings.weight': 1,
    'attention.wq.weight': 0,
    'attention.wk.weight': 0,
    'attention.wv.weight': 0,
    'attention.wo.weight': 1,
    'feed_forward.w1.weight': 0,
    'feed_forward.w3.weight': 0,
    'feed_forward.w2.weight': 1,
}


MODEL_TYPE = {
    '7B': 'llama-2-7b',
    '7B-chat': 'llama-2-7b-chat',
    '13B': 'llama-2-13b',
    '13B-chat': 'llama-2-13b-chat',
    '70B': 'llama-2-70b',
    '70B-chat': 'llama-2-70b-chat',
}

supported_model_types = list(MODEL_TYPE.keys())


def convert_meta_weights(
    meta_root_ckpt_dir: Path,  # root dir to meta checkpoints, should not include model types in the path
    output_dir: Path,
    model_type: str = '7B',
    dtype: str = 'bfloat16',
    verify: bool = False,
) -> None:
    assert model_type in supported_model_types

    dt = getattr(torch, dtype, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f'{dtype} is not a valid dtype.')

    device = None
    if verify:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dtype = dt

    full_model_type = MODEL_TYPE[model_type]
    model_ckpt_dir = meta_root_ckpt_dir / full_model_type
    output_dir = output_dir / full_model_type
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Starting to convert LLaMA 2 model weights for model type {model_type}, this may take few minutes...')

    checkpoint_files = sorted(model_ckpt_dir.glob('*.pth'))
    checkpoint_files.sort()
    n_checkpoints = len(checkpoint_files)

    if n_checkpoints == 0:
        raise RuntimeError(
            f'No checkpoints were found at checkpoint root dir {model_ckpt_dir}. `consolidated.0*.pth` files expected at that location.'
        )

    print(f'Found {n_checkpoints} checkpoint shards, will merge these into a single checkpoint...')

    # the tokenizer is the same for all model sizes, so we store it in the output parent dir
    if not (output_dir.parent / 'tokenizer.model').exists():
        print(f'Copying tokenizer model to {output_dir.parent}...')
        shutil.copy(meta_root_ckpt_dir / 'tokenizer.model', output_dir.parent)

    # for the bigger models, there are multiple model-parallel checkpoints
    # and we combine them into one single file
    combined = None
    for file in tqdm(checkpoint_files, total=n_checkpoints):
        checkpoint = torch.load(file, map_location='cpu')
        converted = convert_state_dict(checkpoint, dtype=dtype)
        if combined is None:
            combined = converted
            continue
        for name, param in converted.items():
            dim = None
            for k, d in shard_dims.items():
                if k in name:
                    dim = d
                    break
            if dim is None:
                # Extra check: assert that tensors are the same if not sharded
                # assert torch.allclose(combined[name], param)
                continue
            combined[name] = torch.cat((combined[name], param), dim=dim)

        del checkpoint
        del converted
        gc.collect()

    if verify:
        # note with bf16, 7B model will consume 14GB GPU RAM
        print('Verifying the merged checkpoint state by calling model.load_state_dict on our model...')

        if device == 'cuda':
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

        model_args = ModelArgs.from_model_type(model_type)
        model_args.vocab_size = 32000
        model_args.head_type = 'lm_head'
        model = Transformer(model_args)
        model.load_state_dict(combined, strict=True)
        model = model.to(device)
        del model

    # save merged checkpoint
    ckpt_save_path = output_dir / 'consolidated.pth'
    print(f'Saving merged checkpoint to {ckpt_save_path} ...')
    torch.save(combined, ckpt_save_path)

    print(f'Copying params.json to {output_dir}...')
    shutil.copy(model_ckpt_dir / 'params.json', output_dir)


if __name__ == '__main__':
    convert_meta_weights(
        meta_root_ckpt_dir=Path('/home/michael/llama-2'),
        output_dir=Path('./checkpoints/llama-2'),
        model_type='7B',
        verify=True,
    )
    convert_meta_weights(
        meta_root_ckpt_dir=Path('/home/michael/llama-2'),
        output_dir=Path('./checkpoints/llama-2'),
        model_type='7B-chat',
        verify=True,
    )
