# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""Module to build prompt only dataset for RL"""
from typing import Tuple, List, Mapping, Text, Any
import functools
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import shutil
import json
import random
import pickle
import torch

# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from instruct_llama.models.tokenizer import Tokenizer
from instruct_llama.utils.logger import create_logger
from instruct_llama.utils.file_helper import (
    find_certain_files_under_dir,
    read_json_file,
    read_jsonl_file,
    read_zipped_jsonl_file,
)
from instruct_llama.utils.prompt_builder import build_prompt_completion, Dialog


logger = create_logger()

Metadata = Mapping[Text, Text]

DEFAULT_SYSTEM_PROMPT = {
    'role': 'system',
    'content': '',
}

# this will be inserted into the training data as the first system prompt
DEFAULT_DIALOG = [DEFAULT_SYSTEM_PROMPT]


# ----------------------------------- helper functions -----------------------------------

KEYWORDS_TO_SKIP = ['photo', 'video', 'movie', 'youtube', 'YouTube']


Answers = List[Mapping[Text, Any]]


def _split_and_save_datasets(
    datasets: List[dict],
    validation_ratio: float,
    train_output_file: str,
    val_output_file: str,
    meta_output_file: str,
    meta: dict,
) -> None:
    # split into train and validation datasets as dolly only have one single .json file
    random.shuffle(datasets)

    val_size = int(len(datasets) * validation_ratio)
    train_size = len(datasets) - val_size

    train_set, val_set = torch.utils.data.random_split(datasets, [train_size, val_size])

    for data, out_file in zip((train_set, val_set), (train_output_file, val_output_file)):
        if len(data) > 0:
            logger.info(f'Saving {len(data)} processed data to {out_file!r} ...')
            pickle.dump(data, open(out_file, 'wb'))

    meta = {
        **meta,
        'num_train_samples': len(train_set),
        'num_validation_samples': len(val_set),
    }

    logger.info(f'Saving metadata to {meta_output_file!r} ...')

    with open(meta_output_file, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def _convert_to_llama_chat_format(raw_text) -> Dialog:
    dialog = []
    conversations = raw_text.split('\n\nHuman: ')[1:]

    for pair in conversations:
        # standardize some punctuation
        pair = pair.replace(',  ', ', ').replace('.  ', '. ').replace('?  ', '? ').replace('!  ', '! ')
        contents = pair.split('\n\nAssistant: ')
        # skip some bad samples
        if len(contents) != 2:
            return dialog

        dialog.append({'role': 'user', 'content': contents[0]})
        dialog.append({'role': 'assistant', 'content': contents[1]})

    return dialog


def _process_single_red_team_jsonl_file(
    file_path: str,
    tokenizer: Tokenizer,
    max_seq_len: int,
) -> List[Tuple[int]]:
    """
    Red teaming data has different structures than other files.
    Read one single .jsonl file and go over each row to build the dataset samples,
    where we only need the prompt tokens for RL.
    """

    # needs to tread it as if it is .json
    with open(file_path, 'r', encoding='utf-8') as file:
        rows = json.loads(file.read())

    samples = []

    for row in rows:
        dialog = _convert_to_llama_chat_format(row['transcript'])

        if len(dialog) == 0:
            continue

        # build prompt tokens, for RL, we don't need the completion tokens,
        # as that's the job of the RL agent
        dialog = DEFAULT_DIALOG + dialog
        prompt_tokens, _ = build_prompt_completion(dialog, tokenizer)

        assert prompt_tokens is not None

        if len(prompt_tokens) > max_seq_len:
            continue

        samples.append({'prompt_tokens': prompt_tokens})

    return samples


def _process_single_jsonl_file(
    file_path: str,
    tokenizer: Tokenizer,
    max_seq_len: int,
) -> List[Tuple[int]]:
    """
    Read one single .jsonl file and go over each row to build the dataset samples,
    where we only need the prompt tokens for RL.
    """

    samples = []

    for row in read_zipped_jsonl_file(file_path):
        dialog = _convert_to_llama_chat_format(row['chosen'])

        if len(dialog) == 0:
            continue

        # build prompt tokens, for RL, we don't need the completion tokens,
        # as that's the job of the RL agent
        dialog = DEFAULT_DIALOG + dialog
        prompt_tokens, _ = build_prompt_completion(dialog, tokenizer)

        assert prompt_tokens is not None

        if len(prompt_tokens) > max_seq_len:
            continue

        samples.append({'prompt_tokens': prompt_tokens})

    return samples


# ----------------------------------- high quality datasets -----------------------------------


def process_hh_rlhf_dataset(
    src_dir: str,
    output_dir: str,
    tokenizer: Tokenizer,
    num_workers=4,
    validation_ratio: float = 0.05,
    max_seq_length: int = 2048,  # prompt lengths greater than this are discarded
    overwrite_output: bool = False,
    metadata: Metadata = {
        'name': 'Human preference data',
        'language': 'English',
        'home_page': 'https://github.com/anthropics/hh-rlhf',
    },
) -> None:
    """Process Human preference dataset and save the tokenized prompt to .pkl format."""

    assert os.path.exists(src_dir) and os.path.isdir(src_dir)
    assert 0 <= validation_ratio <= 0.2

    train_output_file = os.path.join(output_dir, 'train.pkl')
    val_output_file = os.path.join(output_dir, 'validation.pkl')
    meta_output_file = os.path.join(output_dir, 'meta.json')

    if any(os.path.exists(f) for f in (train_output_file, val_output_file, meta_output_file)) and not overwrite_output:
        logger.error(
            f'The output files "{train_output_file}", "{val_output_file}", "{meta_output_file}" already exists, aborting ...'
        )
        return

    # Create the output directory if necessary
    if os.path.exists(output_dir) and overwrite_output:
        logger.info(f'Cleanup output folder {output_dir!r}')
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    if metadata is None:
        metadata = {}

    working_files = find_certain_files_under_dir(src_dir, '.jsonl.gz')

    num_files = len(working_files)

    if num_files == 0:
        logger.warning('Found no .jsonl.gz file')
        return

    if num_files < num_workers:
        num_workers = num_files

    logger.info(f'Processing {num_files} .jsonl.gz files using {num_workers} workers ...')
    process_file_func = functools.partial(
        _process_single_jsonl_file,
        max_seq_len=max_seq_length,
        tokenizer=tokenizer,
    )

    datasets = []

    # Create a ProcessPoolExecutor with maximum N processes
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_file_func, file) for file in working_files]

        for future in as_completed(futures):
            samples = future.result()

            datasets.extend(samples)

    metadata['vocab_size'] = tokenizer.vocab_size
    metadata[
        'data_structure'
    ] = 'A list of dictionary object with a single key "prompt_tokens", which contains the tokenized user prompt'

    logger.info('Saving processed Human preference dataset ...')
    _split_and_save_datasets(
        datasets,
        validation_ratio,
        train_output_file,
        val_output_file,
        meta_output_file,
        metadata,
    )


if __name__ == '__main__':
    seed = 1
    torch.manual_seed(seed)
    random.seed(seed)

    tokenizer = Tokenizer(model_path='./meta_checkpoints/tokenizer.model')

    process_hh_rlhf_dataset(
        src_dir='/home/michael/datasets/hh-rlhf',
        output_dir='./datasets/hh-rlhf_prompt_only',
        tokenizer=tokenizer,
        num_workers=8,
    )
