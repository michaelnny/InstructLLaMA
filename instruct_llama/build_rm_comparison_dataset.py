"""Module to build reward model (RM) comparison dataset"""
from typing import Tuple, List, Mapping, Text, Any
import functools
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import shutil
import json
import random
import pickle
import re
import torch
import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import bs4

# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from instruct_llama.tokenizer import Tokenizer


from instruct_llama.utils import (
    create_logger,
    find_certain_files_under_dir,
    read_json_file,
    read_jsonl_file,
    count_words,
    build_prompt_completion,
    Dialog,
)

logger = create_logger()

Metadata = Mapping[Text, Text]

DEFAULT_SYSTEM_PROMPT = {
    'role': 'system',
    'content': "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and correct. If you don't know the answer to a question, please don't share false information.",
}

# this will be inserted into the training data as the first system prompt
DEFAULT_DIALOG = [DEFAULT_SYSTEM_PROMPT]


# ----------------------------------- helper functions -----------------------------------

SKIP_KEY_WORDS = ['photo', 'video', 'movie', 'youtube', 'YouTube']


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
            logger.info(f'Saving {len(data)} processed data to "{out_file}" ...')
            pickle.dump(data, open(out_file, 'wb'))

    meta = {
        **meta,
        'num_train_samples': len(train_set),
        'num_validation_samples': len(val_set),
    }

    logger.info(f'Saving metadata to "{meta_output_file}" ...')

    with open(meta_output_file, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def _string_found(string1: str, string2: str) -> bool:
    if re.search(r'\b' + re.escape(string1) + r'\b', string2):
        return True
    return False


def _sort_answers_by_score_desc(answers: Answers) -> Answers:
    out = sorted(answers, key=lambda d: d['pm_score'], reverse=True)
    return out


def _deduplicate_answers_by_score(answers: Answers, shuffle: bool = True) -> Answers:
    if shuffle:
        # add some randomness so we are not always using the first occurence of some score
        random.shuffle(answers)

    scores = [a['pm_score'] for a in answers]

    if len(answers) == len(set(scores)):
        return answers

    _, unique_indeces = np.unique(scores, return_index=True)

    out = [answers[i] for i in unique_indeces]

    assert len(out) == len(set(scores))

    return out


def _remove_answers_with_zero_score(answers: Answers) -> Answers:
    out = [a for a in answers if int(a['pm_score']) != 0]

    assert all([int(a['pm_score']) != 0 for a in out])

    return out


def _question_contains_skip_words(question: str) -> bool:
    if any(_string_found(question, k) for k in SKIP_KEY_WORDS):
        return True
    return False


def _filter_answers(answers: Answers, min_completions: int, max_completions: int, remove_zero_score: bool) -> Answers:
    assert min_completions >= 2 and max_completions > min_completions

    if remove_zero_score:
        answers = _remove_answers_with_zero_score(answers)
    answers = _deduplicate_answers_by_score(answers)

    if len(answers) < min_completions:
        return []

    answers = _sort_answers_by_score_desc(answers)

    if len(answers) > max_completions:
        answers = answers[:max_completions]

    return answers


def _extract_text(input_string: str) -> str:
    """Extract raw text from the given string, since it often contains lots of HTML tags."""
    soup = bs4.BeautifulSoup(input_string, features='html.parser')

    out = soup.text.replace('\n', ' ').replace('  ', ' ')  # .replace("\'", "'")
    out = out.strip()
    return out


def _process_single_stackexchange_file(
    file_path: str,
    tokenizer: Tokenizer,
    max_seq_len: int,
    min_question_words: int,
    min_completions: int,
    max_completions: int,
    remove_zero_score: bool,
) -> List[Tuple[int]]:
    """
    Read one single .parquet file and go over each row to build the dataset samples.

    For each row, we apply these:
        * Check if question has minimum number of words, if not skip it
        * Check if question contains some key words which we should skip (like pictures, moviews etc)
        * Filter answers by apply the following rules:
            - Remove (semi-randomly) answers with duplicate scores
            - (Optional) remove answers with zero score
            - Sort answer by score in descending order
        * Build and tokenize prompt with standard chat format
        * Tokenize each answer, skip the answer if the length of prompt tokens + answer tokens are greater than max_seq_len

    """
    df = pq.read_table(file_path).to_pandas()

    samples = []

    for index, row in df.iterrows():
        question = row['question']
        answers = row['answers']

        if _question_contains_skip_words(question):
            continue

        question = _extract_text(question)
        if count_words(question) < min_question_words:
            continue

        answers = _filter_answers(
            answers, min_completions=min_completions, max_completions=max_completions, remove_zero_score=remove_zero_score
        )
        if len(answers) < min_completions:
            continue

        # build ptompt tokens once
        dialog = DEFAULT_DIALOG + [
            {'role': 'user', 'content': question},
        ]
        prompt_tokens, _ = build_prompt_completion(dialog, tokenizer)

        assert prompt_tokens is not None

        # build completions by loop through each answer
        completions_tokens = []
        for a in answers:
            answer_text = _extract_text(a['text'])
            answer = f' {(answer_text).strip()} '
            answer_tokens = tokenizer.encode(answer, bos=False, eos=True)

            if len(prompt_tokens) + len(answer_tokens) <= max_seq_len:
                completions_tokens.append(answer_tokens)

        if len(completions_tokens) < min_completions:
            continue

        samples.append({'prompt_tokens': prompt_tokens, 'completions_tokens': completions_tokens})

    return samples


# ----------------------------------- high quality datasets -----------------------------------


def process_stackexchange_dataset(
    src_dir: str,
    output_dir: str,
    tokenizer: Tokenizer,
    min_question_words: int = 8,
    min_completions: int = 3,
    max_completions: int = 9,
    remove_zero_score: bool = False,
    num_workers=8,
    validation_ratio: float = 0.08,
    max_seq_length: int = 2048,  # prompt + completion lengths greater than this are discarded
    overwrite_output: bool = False,
    metadata: Metadata = {
        'name': 'Stack exchange preferences',
        'language': 'English',
        'home_page': 'https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences',
    },
) -> None:
    """Process Stack exchange preferences dataset and save the tokenized prompt:completion pairs to .pkl format."""

    assert os.path.exists(src_dir) and os.path.isdir(src_dir)
    assert 0 <= validation_ratio <= 0.2

    train_output_file = os.path.join(output_dir, 'train.pkl')
    val_output_file = os.path.join(output_dir, 'validation.pkl')
    meta_output_file = os.path.join(output_dir, 'meta.json')

    if any(os.path.exists(f) for f in (train_output_file, val_output_file, meta_output_file)) and not overwrite_output:
        logger.error(
            f'The output files "{train_output_file}", "{val_output_file}", "{meta_output_file}" already exists, aborting...'
        )
        return

    # Create the output directory if necessary
    if os.path.exists(output_dir) and overwrite_output:
        logger.info(f'cleanup output folder "{output_dir}"')
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    if metadata is None:
        metadata = {}

    # the source files are p
    working_files = find_certain_files_under_dir(src_dir, '.parquet')

    # TODO delete this line
    random.shuffle(working_files)
    working_files = working_files

    num_files = len(working_files)

    if num_files == 0:
        logger.warning('Found no .parquet file')
        return

    logger.info(f'Processing {num_files} .parquet files using {num_workers} workers ...')

    process_file_func = functools.partial(
        _process_single_stackexchange_file,
        max_seq_len=max_seq_length,
        min_question_words=min_question_words,
        min_completions=min_completions,
        max_completions=max_completions,
        remove_zero_score=remove_zero_score,
        tokenizer=tokenizer,
    )

    datasets = []

    # Create a ProcessPoolExecutor with maximum N processes
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_file_func, file) for i, file in enumerate(working_files)]

        for future in as_completed(futures):
            samples = future.result()

            datasets.extend(samples)

    metadata['vocab_size'] = tokenizer.vocab_size
    metadata['data_structure'] = (
        'A list of dictionary object with two keys "prompt_tokens", "completions_tokens", '
        'where "prompt_tokens" contains the tokenized user prompt, and "completions_tokens" contains a list of ordered completion (answer) tokens, '
        'with the best answer at the beginning of the list (index 0), and worset answer at the end of the list.'
    )
    metadata['min_completions'] = min_completions
    metadata['max_completions'] = max_completions

    logger.info('Saving processed stack exchange preferences dataset...')
    _split_and_save_datasets(
        datasets,
        validation_ratio,
        train_output_file,
        val_output_file,
        meta_output_file,
        metadata,
    )


if __name__ == '__main__':
    tokenizer = Tokenizer(model_path='./checkpoints/llama-2/tokenizer.model')

    process_stackexchange_dataset(
        src_dir='./raw_data/stackexchange_dataset',
        output_dir='./datasets/stackexchange_dataset',
        tokenizer=tokenizer,
        num_workers=16,
        overwrite_output=True,
    )
