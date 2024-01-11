# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""
Module for build pretraining datasets.
"""

from typing import Tuple, List, Mapping, Text
import functools
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import json
import random
import pickle
import copy
import numpy as np


# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from instruct_llama.models.tokenizer import Tokenizer
from instruct_llama.utils.logger import create_logger
from instruct_llama.utils.file_helper import find_certain_files_under_dir, read_zipped_jsonl_file


logger = create_logger()

Metadata = Mapping[Text, Text]


# ----------------------------------- helper functions -----------------------------------


def _save_dataset_to_disk(metadata, output_dir, data_type, dataset_prefix, num_tokens, temp_files, logger, shuffle=True):
    if shuffle:
        random.shuffle(temp_files)

    save_fname = os.path.join(output_dir, f'{dataset_prefix}.npy')
    logger.info(f'Merging and saving dataset to {save_fname!r} ...')
    _merge_and_write_to_disk(temp_files, data_type, num_tokens, save_fname)

    metadata['num_tokens'] = num_tokens

    meta_file_json = os.path.join(output_dir, f'{dataset_prefix}_meta.json')
    logger.info(f'Saving metadata to {meta_file_json!r} ...')

    with open(meta_file_json, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def _merge_and_write_to_disk(temp_files, data_type, max_items, output_file, delete_temp_files=True):
    """
    Combining different temp files together and save the content into a single numpy memmap array.
    """
    assert data_type in [np.uint16, np.uint32]
    assert not os.path.exists(output_file)
    assert max_items > 0

    if len(temp_files) == 0:
        return

    num_added = 0

    memmap_array = np.memmap(output_file, dtype=data_type, mode='w+', shape=(max_items,))

    # Load each of temp files and write to the memmap array
    for f in temp_files:
        data = pickle.load(open(f, 'rb'))
        start = num_added
        end = min(start + len(data), max_items)

        memmap_array[start:end] = data[:end]
        num_added += len(data)
        if num_added >= max_items or end >= max_items:
            break

    # Explicitly flush the file buffer to ensure data is written to disk
    memmap_array.flush()

    # Delete temp files
    if delete_temp_files:
        for f in temp_files:
            os.remove(f)


def process_single_file(
    input_file: str, tokenizer: Tokenizer, validation_ratio: float, output_dir: str, max_chunk_size: int = 2
):
    base_name = os.path.basename(input_file)

    tokens = {'train': [], 'validation': []}
    temp_files = {'train': [], 'validation': []}
    num_tokens = {'train': 0, 'validation': 0}

    def write_chunk_to_disk(ds_prefix):
        assert tokens[ds_prefix][0] == tokenizer.bos_id
        assert tokens[ds_prefix][-1] == tokenizer.eos_id
        temp_f = os.path.join(output_dir, f'{ds_prefix}_{base_name}_{num_tokens[ds_prefix]}.tmp')
        with open(temp_f, 'wb') as f:
            pickle.dump(tokens[ds_prefix], f)
        temp_files[ds_prefix].append(temp_f)
        tokens[ds_prefix] = []

    for row in read_zipped_jsonl_file(input_file):
        text = row['raw_content']
        text_ids = tokenizer.encode(text, bos=True, eos=True)

        ds_prefix = 'validation' if random.random() < validation_ratio else 'train'
        tokens[ds_prefix].extend(text_ids)
        num_tokens[ds_prefix] += len(text_ids)

        # Write current chunk to disk to free up RAM.
        # one integer takes 28 bytes, so 1GB RAM = 1e9 bytes
        if len(tokens[ds_prefix]) * 28 >= max_chunk_size * 1e9:
            write_chunk_to_disk(ds_prefix)

    # Handle remaining chunk
    for ds_prefix in tokens.keys():
        if len(tokens[ds_prefix]) > 0:
            write_chunk_to_disk(ds_prefix)

    return temp_files, num_tokens


def process_redparjama_dataset(
    src_dir: str,
    output_dir: str,
    tokenizer: Tokenizer,
    validation_ratio: float = 0.05,
    max_chunk_size=2,  # RAM GB
    num_workers=8,
    overwrite_output: bool = False,
    metadata: Metadata = {
        'name': 'Red Pajama v2',
        'language': 'English',
        'home_page': 'https://github.com/togethercomputer/RedPajama-Data',
    },
) -> None:
    """Process Red Pajama dataset and save the tokenized text to .pkl format.

    Here's an example format:

    <s>doc 1</s><s>doc 2</s><s>doc 3</s>

    """

    assert os.path.exists(src_dir) and os.path.isdir(src_dir)

    train_output_file = os.path.join(output_dir, 'train.npy')
    val_output_file = os.path.join(output_dir, 'validation.npy')

    if any(os.path.exists(f) for f in (train_output_file, val_output_file)) and not overwrite_output:
        logger.error(f'The output files "{train_output_file}", "{val_output_file}" already exists, aborting ...')
        return

    if metadata is None:
        metadata = {}

    # the Red Pajama v2 dataset comes with .json.gz, but are actually .jsonl format
    working_files = find_certain_files_under_dir(src_dir, '.json.gz')

    num_files = len(working_files)

    if num_files == 0:
        logger.warning('Found no file')
        return

    if num_files < num_workers:
        num_workers = num_files

    # Create the output directory if necessary
    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    logger.info(f'Processing {num_files} files using {num_workers} workers ...')

    temp_files = {'train': [], 'validation': []}
    num_tokens = {'train': 0, 'validation': 0}

    process_func = functools.partial(
        process_single_file,
        tokenizer=tokenizer,
        validation_ratio=validation_ratio,
        output_dir=output_dir,
        max_chunk_size=max_chunk_size,
    )

    # For logging only
    total_num_tokens = 0
    last_logged = 0

    # Create a ProcessPoolExecutor with maximum N processes
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_func, file) for file in working_files]

        for future in as_completed(futures):
            result = future.result()
            _temp_files, _num_tokens = result

            for ds_prefix in temp_files.keys():
                if len(_temp_files[ds_prefix]) > 0:
                    temp_files[ds_prefix].extend(_temp_files[ds_prefix])
                    num_tokens[ds_prefix] += _num_tokens[ds_prefix]

            total_num_tokens = sum(num_tokens.values())

            if total_num_tokens - last_logged >= 1e8:
                logger.info(f'Processed {total_num_tokens/1e6:.2f} million tokens ...')
                last_logged = total_num_tokens

    logger.info(f'Finished processing {total_num_tokens/1e6:.2f} million tokens')

    data_type = np.uint16 if tokenizer.vocab_size < 2**16 else np.uint32

    metadata['vocab_size'] = tokenizer.vocab_size
    metadata['data_type'] = np.dtype(data_type).name
    metadata[
        'data_structure'
    ] = 'A stream of token sequences created by combining various tokenized texts together, where document boundary is separated by BOS and EOS tokens <s></s>'

    # For each dataset, combining different temp files together and save the content into a single numpy memmap array
    for ds_prefix in temp_files.keys():
        if len(temp_files[ds_prefix]) > 0:
            logger.info(f'Saving {ds_prefix} dataset ...')
            _save_dataset_to_disk(
                metadata=copy.deepcopy(metadata),
                output_dir=output_dir,
                data_type=data_type,
                dataset_prefix=ds_prefix,
                num_tokens=num_tokens[ds_prefix],
                temp_files=temp_files[ds_prefix],
                logger=logger,
            )


if __name__ == '__main__':
    seed = 1
    np.random.seed(seed)
    random.seed(seed)

    tokenizer = Tokenizer(model_path='./meta_checkpoints/tokenizer.model')

    process_redparjama_dataset(
        src_dir='/home/michael/datasets/redpajama-data-v2-mini',
        output_dir='./datasets/red_pajama_mini',
        tokenizer=tokenizer,
        num_workers=12,
    )
