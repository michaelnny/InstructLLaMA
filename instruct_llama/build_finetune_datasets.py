"""
Module for build instruct fine-tuning datasets.
"""

from typing import Tuple, List, Mapping, Text
import functools
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import shutil
import json
import random
import pickle
import torch

from instruct_llama.tokenizer import Tokenizer

from instruct_llama.utils import (
    create_logger,
    find_certain_files_under_dir,
    read_txt_file,
    read_json_file,
    read_jsonl_file,
    count_words,
    build_prompt_completion,
)

logger = create_logger()

Metadata = Mapping[Text, Text]

DEFAULT_SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and correct. If you don't know the answer to a question, please don't share false information.",
}

# build the training data using dialog style
DEFAULT_DIALOG = [DEFAULT_SYSTEM_PROMPT]


def _split_and_save_datasets(
    datasets,
    validation_ratio,
    train_output_file,
    val_output_file,
    meta_output_file,
    meta,
):
    # split into train and validation datasets as dolly only have one single .json file
    random.shuffle(datasets)

    val_size = int(len(datasets) * validation_ratio)
    train_size = len(datasets) - val_size

    train_set, val_set = torch.utils.data.random_split(datasets, [train_size, val_size])

    for data, out_file in zip(
        (train_set, val_set), (train_output_file, val_output_file)
    ):
        if len(data) > 0:
            logger.info(f'Saving {len(data)} processed data to "{out_file}" ...')
            pickle.dump(data, open(out_file, "wb"))

    meta = {
        **meta,
        "num_train_samples": len(train_set),
        "num_validation_samples": len(val_set),
    }

    logger.info(f'Saving metadata to "{meta_output_file}" ...')

    with open(meta_output_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def process_dolly_dataset(
    src_file: str,
    output_dir: str,
    tokenizer: Tokenizer,
    min_prompt_words: int = 5,
    validation_ratio: float = 0.08,
    max_seq_length: int = 2048,  # prompt + completion lengths greater than this are discarded
    overwrite_output: bool = False,
    metadata: Metadata = {},
):
    """Process dolly dataset and save the tokenized prompt:completion pairs to .pkl format.

    Here's an example format of prompt:completion pair before apply tokenization:
    {"prompt": "BOS [INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} [/INST]", "completion": " {1st response} EOS"}

    """

    assert os.path.exists(src_file) and os.path.isfile(src_file)
    assert 0 <= validation_ratio <= 0.2

    train_output_file = os.path.join(output_dir, "train.pkl")
    val_output_file = os.path.join(output_dir, "validation.pkl")
    meta_output_file = os.path.join(output_dir, "meta.json")

    if (
        any(
            os.path.exists(f)
            for f in (train_output_file, val_output_file, meta_output_file)
        )
        and not overwrite_output
    ):
        logger.error(
            f'The output files "{train_output_file}", "{val_output_file}", "{meta_output_file}" already exists, aborting...'
        )
        return

    if metadata is None:
        metadata = {}

    json_objs = read_jsonl_file(src_file)

    if json_objs is None:
        logger.error(f'Invalid content from src file "{src_file}"')
        return

    # Create the output directory if necessary
    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    logger.info("Start to processing dolly dataset...")
    datasets = []

    for item in json_objs:
        context = item["context"].strip()
        prompt = item["instruction"].strip()
        completion = item["response"].strip()

        # handle special cases where all prompt words are mixed together
        if count_words(prompt) < min_prompt_words:
            continue

        if len(completion) == 0:
            continue

        if len(context) > 0:
            prompt += f"\n\n{context}"

        dialog = DEFAULT_DIALOG + [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]

        prompt_tokens, completion_tokens = build_prompt_completion(dialog, tokenizer)

        assert prompt_tokens is not None and completion_tokens is not None

        if len(prompt_tokens) + len(completion_tokens) > max_seq_length:
            continue

        # if random.random() > 0.8:
        #     logger.info(f"Prompt: {tokenizer.decode(prompt_tokens)}")
        #     logger.info(f"Completion: {tokenizer.decode(completion_tokens)}")
        #     logger.info("-" * 40)

        datasets.append(
            {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
        )

    metadata["vocab_size"] = tokenizer.vocab_size
    metadata["data_structure"] = "A list of prompt:completion token sequences pairs."

    logger.info("Start to save processed dolly dataset...")
    _split_and_save_datasets(
        datasets,
        validation_ratio,
        train_output_file,
        val_output_file,
        meta_output_file,
        metadata,
    )


def process_alpaca_dataset(
    src_file: str,
    output_dir: str,
    tokenizer: Tokenizer,
    min_prompt_words: int = 5,
    validation_ratio: float = 0.05,
    overwrite_output: bool = False,
    metadata: Metadata = {},
):
    """Process alpaca dataset and save the tokenized prompt:completion pairs to .pkl format.

    Here's an example format of prompt:completion pair before apply tokenization:
    {"prompt": "BOS [INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} [/INST]", "completion": " {1st response} EOS"}

    """

    assert os.path.exists(src_file) and os.path.isfile(src_file)
    assert 0 <= validation_ratio <= 0.2

    train_output_file = os.path.join(output_dir, "train.pkl")
    val_output_file = os.path.join(output_dir, "validation.pkl")
    meta_output_file = os.path.join(output_dir, "meta.json")

    if (
        any(
            os.path.exists(f)
            for f in (train_output_file, val_output_file, meta_output_file)
        )
        and not overwrite_output
    ):
        logger.error(
            f'The output files "{train_output_file}", "{val_output_file}", "{meta_output_file}" already exists, aborting...'
        )
        return

    if metadata is None:
        metadata = {}

    json_objs = read_json_file(src_file)

    if json_objs is None:
        logger.error(f'Invalid content from src file "{src_file}"')
        return

    # Create the output directory if necessary
    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    logger.info("Start to processing alpaca dataset...")
    datasets = []

    for item in json_objs:
        context = item["input"].strip()
        prompt = item["instruction"].strip()
        completion = item["output"].strip()

        # handle special cases where all prompt words are mixed together
        if count_words(prompt) < min_prompt_words:
            continue

        if len(completion) == 0:
            continue

        if len(context) > 0:
            prompt += f"\n\n{context}"

        dialog = DEFAULT_DIALOG + [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]

        prompt_tokens, completion_tokens = build_prompt_completion(dialog, tokenizer)

        assert prompt_tokens is not None and completion_tokens is not None

        # if random.random() > 0.8:
        #     logger.info(f"Prompt: {tokenizer.decode(prompt_tokens)}")
        #     logger.info(f"Completion: {tokenizer.decode(completion_tokens)}")
        #     logger.info("-" * 40)

        datasets.append(
            {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
        )

    metadata["vocab_size"] = tokenizer.vocab_size
    metadata["data_structure"] = "A list of prompt:completion token sequences pairs."

    logger.info("Start to save processed alpaca dataset...")
    _split_and_save_datasets(
        datasets,
        validation_ratio,
        train_output_file,
        val_output_file,
        meta_output_file,
        metadata,
    )


def process_single_dm_math_txt_file(txt_file, tokenizer):
    pairs = []

    with open(str(txt_file), "r", encoding="utf-8") as f:
        lines = f.read().split("\n")[:-1]

    for i in range(0, len(lines), 2):
        prompt = lines[i].strip()
        completion = lines[i + 1].strip()

        dialog = DEFAULT_DIALOG + [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]

        prompt_tokens, completion_tokens = build_prompt_completion(dialog, tokenizer)

        assert prompt_tokens is not None and completion_tokens is not None

        # if random.random() > 0.8:
        #     logger.info(f"Prompt: {tokenizer.decode(prompt_tokens)}")
        #     logger.info(f"Completion: {tokenizer.decode(completion_tokens)}")
        #     logger.info("-" * 40)

        pairs.append(
            {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
        )

    return pairs


def process_deepmind_math_dataset(
    src_dirs,
    output_dir,
    tokenizer: Tokenizer,
    filter_by_keys=[],  # only include files matching the keys defined here, if empty all files will be used
    num_workers=8,
    max_num_sample=20000,  # limit total amount of samples to avoid imbalanced training data
    validation_ratio=0.05,
    overwrite_output=False,
    metadata: Metadata = {},
):
    """Process DeepMind Mathematics dataset and save the tokenized prompt:completion pairs to .pkl format.

    Here's an example format of prompt:completion pair before apply tokenization:
    {"prompt": "BOS [INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} [/INST]", "completion": " {1st response} EOS"}

    """
    assert len(src_dirs) > 0
    assert all(
        src_dir != output_dir and os.path.exists(src_dir) and os.path.isdir(src_dir)
        for src_dir in src_dirs
    )

    all_files = []
    working_files = []

    for src_dir in src_dirs:
        all_files.extend(find_certain_files_under_dir(src_dir, file_type=".txt"))

    # filter out file name not contain the keys
    if len(filter_by_keys) > 0:
        logger.info(
            f"Filter {len(all_files)} .txt files by key words {filter_by_keys} ..."
        )
        for file in all_files:
            basename = os.path.basename(file)
            for k in filter_by_keys:
                if basename.startswith(k):
                    working_files.append(file)
                    break
    else:
        working_files = all_files

    num_files = len(working_files)

    if num_files == 0:
        logger.warning(f"Found no .txt file")
        return
    else:
        logger.info(f"Found {num_files} .txt files")

    train_output_file = os.path.join(output_dir, "train.pkl")
    val_output_file = os.path.join(output_dir, "validation.pkl")
    meta_output_file = os.path.join(output_dir, "meta.json")

    if (
        any(
            os.path.exists(f)
            for f in (train_output_file, val_output_file, meta_output_file)
        )
        and not overwrite_output
    ):
        logger.error(
            f'The output files "{train_output_file}", "{val_output_file}", "{meta_output_file}" already exists, aborting...'
        )
        return

    if metadata is None:
        metadata = {}

    if os.path.exists(output_dir) and overwrite_output:
        logger.info(f'cleanup output folder "{output_dir}"')
        shutil.rmtree(output_dir)

    # Create the output directory if necessary
    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    logger.info(f"Processing {num_files} .txt files using {num_workers} workers ...")

    logger.info("Start to processing DeepMind Mathematics dataset...")

    process_txt_file_func = functools.partial(
        process_single_dm_math_txt_file,
        tokenizer=tokenizer,
    )

    datasets = []

    # Create a ProcessPoolExecutor with maximum N processes
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_txt_file_func, file)
            for i, file in enumerate(working_files)
        ]

        for future in as_completed(futures):
            pairs = future.result()
            datasets.extend(pairs)

    logger.info(f"Finished processing {num_files} .txt files, {len(datasets)} samples")

    metadata["vocab_size"] = tokenizer.vocab_size
    metadata["data_structure"] = "A list of prompt:completion token sequences pairs."

    if len(datasets) > max_num_sample:
        logger.info(f"Truncate data to max size of {max_num_sample}")
        random.shuffle(datasets)
        datasets = datasets[:max_num_sample]

    logger.info("Start to save processed DeepMind Mathematics dataset...")
    _split_and_save_datasets(
        datasets,
        validation_ratio,
        train_output_file,
        val_output_file,
        meta_output_file,
        metadata,
    )


if __name__ == "__main__":
    tokenizer = Tokenizer(model_path="/home/michael/llama-2/tokenizer.model")

    process_dolly_dataset(
        src_file="./raw_data/databricks-dolly-15k.jsonl",
        output_dir="./datasets/dolly",
        tokenizer=tokenizer,
        overwrite_output=False,
        metadata={
            "name": "Dolly",
            "language": "English",
            "home_page": "https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm",
        },
    )

    process_alpaca_dataset(
        src_file="./raw_data/alpaca_data_cleaned.json",
        output_dir="./datasets/alpaca",
        tokenizer=tokenizer,
        overwrite_output=False,
        metadata={
            "name": "Alpaca_cleaned",
            "language": "English",
            "home_page": "https://github.com/gururise/AlpacaDataCleaned",
        },
    )

    process_deepmind_math_dataset(
        src_dirs=["./raw_data/deepmind_mathematics/train-easy"],
        output_dir="./datasets/deepmind_mathematics",
        tokenizer=tokenizer,
        overwrite_output=True,
        filter_by_keys=[
            "arithmetic__add_or_sub",
            "arithmetic__add_sub_multiple",
            "arithmetic__div",
            "arithmetic__mul",
        ],
        metadata={
            "name": "DeepMind Mathematics - arithmetic easy",
            "language": "English",
            "home_page": "https://github.com/deepmind/mathematics_dataset",
        },
    )
