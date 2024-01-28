# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


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

import re
import torch

# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from instruct_llama.models.tokenizer import Tokenizer
from instruct_llama.utils.logger import create_logger
from instruct_llama.utils.file_helper import find_certain_files_under_dir, read_json_file, read_jsonl_file, count_words
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


def _process_single_dm_math_txt_file(txt_file: str, tokenizer: Tokenizer) -> List[dict]:
    pairs = []

    with open(str(txt_file), 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')[:-1]

    for i in range(0, len(lines), 2):
        prompt = lines[i].strip()
        completion = lines[i + 1].strip()

        dialog = DEFAULT_DIALOG + [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': completion},
        ]

        prompt_tokens, completion_tokens = build_prompt_completion(dialog, tokenizer)

        assert prompt_tokens is not None and completion_tokens is not None

        pairs.append({'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens})

    return pairs


def _raw_texts_to_dialog(dialog_texts: List[str]) -> Dialog:
    """Converts a list of raw text into dialog formation.
    Note it requires the texts follows the correct user/assistant/user/assistant ... order."""

    # requires at least one turn from each role (user, assistant)
    if len(dialog_texts) < 2:
        return None

    # try to trim the last one so we always get a pair of contents from user and assistant
    if len(dialog_texts) % 2 != 0:
        dialog_texts = dialog_texts[:-1]

    assert len(dialog_texts) % 2 == 0, f'dialog length: {len(dialog_texts)}'

    dialog = DEFAULT_DIALOG + [
        {'role': 'user' if i % 2 == 0 else 'assistant', 'content': raw_text.strip()} for i, raw_text in enumerate(dialog_texts)
    ]

    return dialog


# ----------------------------------- high quality datasets -----------------------------------


def process_dolly_dataset(
    src_file: str,
    output_dir: str,
    tokenizer: Tokenizer,
    min_prompt_words: int = 5,
    validation_ratio: float = 0.05,
    max_seq_length: int = 2048,  # prompt + completion lengths greater than this are discarded
    overwrite_output: bool = False,
    metadata: Metadata = {
        'name': 'Dolly',
        'language': 'English',
        'home_page': 'https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm',
    },
) -> None:
    """Process dolly dataset and save the tokenized prompt:completion pairs to .pkl format.

    Here's an example format of prompt:completion pair before apply tokenization:
    {"prompt": "<s>[INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} [/INST]", "completion": " {1st response} </s>"}

    """

    assert os.path.exists(src_file) and os.path.isfile(src_file)
    assert 0 <= validation_ratio <= 0.2

    train_output_file = os.path.join(output_dir, 'train.pkl')
    val_output_file = os.path.join(output_dir, 'validation.pkl')
    meta_output_file = os.path.join(output_dir, 'meta.json')

    if any(os.path.exists(f) for f in (train_output_file, val_output_file, meta_output_file)) and not overwrite_output:
        logger.error(
            f'The output files "{train_output_file}", "{val_output_file}", "{meta_output_file}" already exists, aborting ...'
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

    logger.info('Processing dolly dataset ...')
    datasets = []

    for item in json_objs:
        context = item['context'].strip()
        prompt = item['instruction'].strip()
        completion = item['response'].strip()

        # handle special cases where all prompt words are mixed together
        if count_words(prompt) < min_prompt_words:
            continue

        if len(completion) == 0:
            continue

        if len(context) > 0:
            prompt += f'\n\n{context}'

        dialog = DEFAULT_DIALOG + [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': completion},
        ]

        prompt_tokens, completion_tokens = build_prompt_completion(dialog, tokenizer)

        assert prompt_tokens is not None and completion_tokens is not None

        if len(prompt_tokens) + len(completion_tokens) > max_seq_length:
            continue

        datasets.append({'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens})

    metadata['vocab_size'] = tokenizer.vocab_size
    metadata['data_structure'] = 'A list of prompt:completion token sequences pairs.'

    logger.info('Saving processed dolly dataset ...')
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
    metadata: Metadata = {
        'name': 'Alpaca_cleaned',
        'language': 'English',
        'home_page': 'https://github.com/gururise/AlpacaDataCleaned',
    },
) -> None:
    """Process alpaca dataset and save the tokenized prompt:completion pairs to .pkl format.

    Here's an example format of prompt:completion pair before apply tokenization:
    {"prompt": "<s>[INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} [/INST]", "completion": " {1st response} </s>"}

    """

    assert os.path.exists(src_file) and os.path.isfile(src_file)
    assert 0 <= validation_ratio <= 0.2

    train_output_file = os.path.join(output_dir, 'train.pkl')
    val_output_file = os.path.join(output_dir, 'validation.pkl')
    meta_output_file = os.path.join(output_dir, 'meta.json')

    if any(os.path.exists(f) for f in (train_output_file, val_output_file, meta_output_file)) and not overwrite_output:
        logger.error(
            f'The output files "{train_output_file}", "{val_output_file}", "{meta_output_file}" already exists, aborting ...'
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

    logger.info('Processing alpaca dataset ...')
    datasets = []

    for item in json_objs:
        context = item['input'].strip()
        prompt = item['instruction'].strip()
        completion = item['output'].strip()

        # handle special cases where all prompt words are mixed together
        if count_words(prompt) < min_prompt_words:
            continue

        if len(completion) == 0:
            continue

        if len(context) > 0:
            prompt += f'\n\n{context}'

        dialog = DEFAULT_DIALOG + [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': completion},
        ]

        prompt_tokens, completion_tokens = build_prompt_completion(dialog, tokenizer)

        assert prompt_tokens is not None and completion_tokens is not None

        datasets.append({'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens})

    metadata['vocab_size'] = tokenizer.vocab_size
    metadata['data_structure'] = 'A list of prompt:completion token sequences pairs.'

    logger.info('Saving processed alpaca dataset ...')
    _split_and_save_datasets(
        datasets,
        validation_ratio,
        train_output_file,
        val_output_file,
        meta_output_file,
        metadata,
    )


def process_deepmind_math_dataset(
    src_dirs,
    output_dir,
    tokenizer: Tokenizer,
    filter_by_names=[],  # only include files matching the keys defined here, if empty all files will be used
    num_workers=8,
    max_samples=20000,  # limit total amount of samples to avoid imbalanced training data
    validation_ratio=0.05,
    overwrite_output: bool = False,
    metadata: Metadata = {
        'name': 'DeepMind Mathematics',
        'language': 'English',
        'home_page': 'https://github.com/deepmind/mathematics_dataset',
    },
) -> None:
    """Process DeepMind Mathematics dataset and save the tokenized prompt:completion pairs to .pkl format.

    Here's an example format of prompt:completion pair before apply tokenization:
    {"prompt": "<s>[INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} [/INST]", "completion": " {1st response} </s>"}

    """
    assert len(src_dirs) > 0
    assert all(src_dir != output_dir and os.path.exists(src_dir) and os.path.isdir(src_dir) for src_dir in src_dirs)

    all_files = []
    working_files = []

    for src_dir in src_dirs:
        all_files.extend(find_certain_files_under_dir(src_dir, file_type='.txt'))

    # filter out file name not contain the keys
    if len(filter_by_names) > 0:
        logger.info(f'Filter {len(all_files)} .txt files by file names {filter_by_names} ...')
        for file in all_files:
            basename = os.path.basename(file)
            for k in filter_by_names:
                if basename == k:
                    working_files.append(file)
                    break
    else:
        working_files = all_files

    num_files = len(working_files)

    if num_files == 0:
        logger.warning('Found no .txt file')
        return
    else:
        logger.info(f'Found {num_files} .txt files')

    train_output_file = os.path.join(output_dir, 'train.pkl')
    val_output_file = os.path.join(output_dir, 'validation.pkl')
    meta_output_file = os.path.join(output_dir, 'meta.json')

    if any(os.path.exists(f) for f in (train_output_file, val_output_file, meta_output_file)) and not overwrite_output:
        logger.error(
            f'The output files "{train_output_file}", "{val_output_file}", "{meta_output_file}" already exists, aborting ...'
        )
        return

    if metadata is None:
        metadata = {}

    if os.path.exists(output_dir) and overwrite_output:
        logger.info(f'cleanup output folder {output_dir!r}')
        shutil.rmtree(output_dir)

    # Create the output directory if necessary
    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    if num_files < num_workers:
        num_workers = num_files

    logger.info(f'Processing {num_files} .txt files using {num_workers} workers ...')

    process_txt_file_func = functools.partial(
        _process_single_dm_math_txt_file,
        tokenizer=tokenizer,
    )

    datasets = []

    # Create a ProcessPoolExecutor with maximum N processes
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_txt_file_func, file) for file in working_files]

        for future in as_completed(futures):
            pairs = future.result()
            datasets.extend(pairs)

    logger.info(f'Finished processing {num_files} .txt files, {len(datasets)} samples')

    metadata['vocab_size'] = tokenizer.vocab_size
    metadata['data_structure'] = 'A list of prompt:completion token sequences pairs.'

    if len(datasets) > max_samples:
        logger.info(f'Truncate data to max size of {max_samples}')
        random.shuffle(datasets)
        random.shuffle(datasets)
        random.shuffle(datasets)
        datasets = datasets[:max_samples]

    logger.info('Saving processed DeepMind Mathematics dataset ...')
    _split_and_save_datasets(
        datasets,
        validation_ratio,
        train_output_file,
        val_output_file,
        meta_output_file,
        metadata,
    )


def process_squad_dataset(
    src_dir: str,
    output_dir: str,
    tokenizer: Tokenizer,
    min_prompt_words: int = 5,
    overwrite_output: bool = False,
    metadata: Metadata = {
        'name': 'SQuAD',
        'language': 'English',
        'home_page': 'https://rajpurkar.github.io/SQuAD-explorer',
    },
) -> None:
    """Process SQuAD dataset and save the tokenized prompt:completion pairs to .pkl format.

    Here's an example format of prompt:completion pair before apply tokenization:
    {"prompt": "<s>[INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} [/INST] {1st response} </s><s>[INST] {2nd user prompt} [/INST]", "completion": " {2nd response} </s>"}

    """

    assert os.path.exists(src_dir) and os.path.isdir(src_dir)

    train_output_file = os.path.join(output_dir, 'train.pkl')
    val_output_file = os.path.join(output_dir, 'validation.pkl')
    meta_output_file = os.path.join(output_dir, 'meta.json')

    if any(os.path.exists(f) for f in (train_output_file, val_output_file, meta_output_file)) and not overwrite_output:
        logger.error(
            f'The output files "{train_output_file}", "{val_output_file}", "{meta_output_file}" already exists, aborting ...'
        )
        return

    if metadata is None:
        metadata = {}

    # Create the output directory if necessary
    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    def json_to_dataset(input_file):
        assert os.path.exists(input_file) and os.path.isfile(input_file)

        dataset = []

        content = None
        with open(input_file, 'r', encoding='utf-8') as file:
            content = file.read()

        content = json.loads(content)
        assert 'data' in content

        for item in content['data']:
            for data in item['paragraphs']:
                context = data['context'].strip()

                dialog_texts = []
                for i, qa in enumerate(data['qas']):
                    if qa['is_impossible']:
                        continue

                    question = qa['question'].strip()

                    # handle special cases where all words are mixed together
                    if count_words(question) < min_prompt_words:
                        continue

                    response = qa['answers'][0]['text'].strip()

                    if len(response) == 0:
                        continue

                    # insert context at the first user prompt
                    if i == 0:
                        question += f'\n\n{context}'

                    dialog_texts.append(question)
                    dialog_texts.append(response)

                dialog = _raw_texts_to_dialog(dialog_texts)

                if dialog is None:
                    continue

                prompt_tokens, completion_tokens = build_prompt_completion(dialog, tokenizer)
                assert prompt_tokens is not None and completion_tokens is not None

                dataset.append({'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens})

        return dataset

    logger.info('Processing SQuAD dataset ...')

    train_dataset = json_to_dataset(os.path.join(src_dir, 'train-v2.0.json'))
    val_dataset = json_to_dataset(os.path.join(src_dir, 'dev-v2.0.json'))

    metadata['vocab_size'] = tokenizer.vocab_size
    metadata['data_structure'] = 'A list of prompt:completion token sequences pairs.'
    metadata['num_train_samples'] = len(train_dataset)
    metadata['num_validation_samples'] = len(val_dataset)

    logger.info(f'Saving {len(train_dataset)} processed data to {train_output_file!r} ...')
    pickle.dump(train_dataset, open(train_output_file, 'wb'))

    logger.info(f'Saving {len(val_dataset)} processed data to {val_output_file!r} ...')
    pickle.dump(val_dataset, open(val_output_file, 'wb'))

    logger.info(f'Saving metadata to {meta_output_file!r} ...')
    with open(meta_output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def process_commonsense_dialog_dataset(
    src_dir: str,
    output_dir: str,
    tokenizer: Tokenizer,
    overwrite_output: bool = False,
    metadata: Metadata = {
        'name': 'Commonsense dialogue',
        'language': 'English',
        'home_page': 'https://www.amazon.science/blog/amazon-releases-new-dataset-for-commonsense-dialogue',
    },
) -> None:
    """Process Commonsense dialogues dataset and save the tokenized prompt:completion pairs to .pkl format.

    Here's an example format of prompt:completion pair before apply tokenization:
    {"prompt": "<s>[INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} [/INST]", "completion": " {1st response} </s>"}

    """

    assert os.path.exists(src_dir) and os.path.isdir(src_dir)

    train_output_file = os.path.join(output_dir, 'train.pkl')
    val_output_file = os.path.join(output_dir, 'validation.pkl')
    meta_output_file = os.path.join(output_dir, 'meta.json')

    if any(os.path.exists(f) for f in (train_output_file, val_output_file, meta_output_file)) and not overwrite_output:
        logger.error(
            f'The output files "{train_output_file}", "{val_output_file}", "{meta_output_file}" already exists, aborting ...'
        )
        return

    if metadata is None:
        metadata = {}

    # Create the output directory if necessary
    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    train_file = os.path.join(src_dir, 'train.json')
    val_file = os.path.join(src_dir, 'valid.json')

    def _dataset_from_json_file(input_file):
        assert os.path.exists(input_file) and os.path.isfile(input_file)

        content = None
        with open(input_file, 'r', encoding='utf-8') as file:
            content = json.loads(file.read())

        dialogs_dict = {}

        for item in content.values():
            key = re.sub(r'\s+', '', item['context'])

            if key not in dialogs_dict:
                dialogs_dict[key] = []

            for turn in item['turns']:
                cleaned_text = turn.replace('.  ', '. ').replace(',  ', ', ').replace('  ', ' ').replace('?.', '?')
                dialogs_dict[key].append(cleaned_text.strip())

        dataset = []

        for dialog_texts in dialogs_dict.values():
            dialog = _raw_texts_to_dialog(dialog_texts)

            if dialog is None:
                continue

            prompt_tokens, completion_tokens = build_prompt_completion(dialog, tokenizer)
            assert prompt_tokens is not None and completion_tokens is not None
            dataset.append({'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens})

        return dataset

    logger.info('Processing Commonsense dialogues dataset ...')

    train_dataset = _dataset_from_json_file(train_file)
    val_dataset = _dataset_from_json_file(val_file)

    metadata['vocab_size'] = tokenizer.vocab_size
    metadata['data_structure'] = 'A list of prompt:completion token sequences pairs.'
    metadata['num_train_samples'] = len(train_dataset)
    metadata['num_validation_samples'] = len(val_dataset)

    logger.info(f'Saving {len(train_dataset)} processed data to {train_output_file!r} ...')
    pickle.dump(train_dataset, open(train_output_file, 'wb'))

    logger.info(f'Saving {len(val_dataset)} processed data to {val_output_file!r} ...')
    pickle.dump(val_dataset, open(val_output_file, 'wb'))

    logger.info(f'Saving metadata to {meta_output_file!r} ...')
    with open(meta_output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


# ----------------------------------- low quality datasets -----------------------------------


# the conversation in MSC does not appears to be natural at all, some are very strange
def process_msc_dialog_dataset(
    src_dir: str,
    output_dir: str,
    tokenizer: Tokenizer,
    overwrite_output: bool = False,
    metadata: Metadata = {
        'name': 'Multi-Session Chat',
        'language': 'English',
        'home_page': 'https://parl.ai/projects/msc/',
    },
) -> None:
    """Process Multi-Session Chat dataset and save the tokenized prompt:completion pairs to .pkl format.

    Here's an example format of prompt:completion pair before apply tokenization:
    {"prompt": "<s>[INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} [/INST]", "completion": " {1st response} </s>"}

    """

    assert os.path.exists(src_dir) and os.path.isdir(src_dir)

    train_output_file = os.path.join(output_dir, 'train.pkl')
    val_output_file = os.path.join(output_dir, 'validation.pkl')
    meta_output_file = os.path.join(output_dir, 'meta.json')

    if any(os.path.exists(f) for f in (train_output_file, val_output_file, meta_output_file)) and not overwrite_output:
        logger.error(
            f'The output files "{train_output_file}", "{val_output_file}", "{meta_output_file}" already exists, aborting ...'
        )
        return

    if metadata is None:
        metadata = {}

    # the dataset comes with splitted .txt file, the content of these .txt files are json object
    train_files, val_files = [], []

    if os.path.exists(src_dir):
        for root, dirnames, filenames in os.walk(src_dir):
            for f in filenames:
                if f == 'train.txt':
                    train_files.append(os.path.join(root, f))
                elif f == 'valid.txt':
                    val_files.append(os.path.join(root, f))

    assert len(train_files) > 0 and len(val_files) > 0

    # Create the output directory if necessary
    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    def _file_content_to_dialogs(txt_file):
        """Read a single .txt file, for each item in the file,
        try to convert the raw conversion into dialog format,
        and return the converted list of dialogs."""
        content = []

        with open(txt_file, 'r', encoding='utf-8') as file:
            for line in file:
                content.append(json.loads(line.strip()))

        dialogs = []

        for item in content:
            conversation = item['dialog']

            if len(conversation) < 2:
                continue

            if len(conversation) % 2 != 0:
                conversation = conversation[:-1]

            assert len(conversation) % 2 == 0, f'dialog length: {len(conversation)}'

            first_id = conversation[0]['id']
            second_id = conversation[1]['id']

            assert first_id != second_id

            if not (
                all([msg['id'] == first_id for msg in conversation[0::2]])
                and all([msg['id'] == second_id for msg in conversation[1::2]])
            ):
                continue

            dialog = DEFAULT_DIALOG + [
                {'role': 'user' if i % 2 == 0 else 'assistant', 'content': item['text'].strip()}
                for i, item in enumerate(conversation)
            ]

            if random.random() > 0.8:
                logger.info(dialog)
                logger.info('-' * 40)

            dialogs.append(dialog)

        return dialogs

    def _dataset_from_files(files):
        dataset = []
        for f in files:
            dialogs = _file_content_to_dialogs(f)
            for dialog in dialogs:
                prompt_tokens, completion_tokens = build_prompt_completion(dialog, tokenizer)
                assert prompt_tokens is not None and completion_tokens is not None
                dataset.append({'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens})

        return dataset

    logger.info('Processing msc dataset ...')
    train_dataset = _dataset_from_files(train_files)
    val_dataset = _dataset_from_files(val_files)

    metadata['vocab_size'] = tokenizer.vocab_size
    metadata['data_structure'] = 'A list of prompt:completion token sequences pairs.'
    metadata['num_train_samples'] = len(train_dataset)
    metadata['num_validation_samples'] = len(val_dataset)

    logger.info(f'Saving {len(train_dataset)} processed data to {train_output_file!r} ...')
    pickle.dump(train_dataset, open(train_output_file, 'wb'))

    logger.info(f'Saving {len(val_dataset)} processed data to {val_output_file!r} ...')
    pickle.dump(val_dataset, open(val_output_file, 'wb'))

    logger.info(f'Saving metadata to {meta_output_file!r} ...')
    with open(meta_output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    seed = 1
    torch.manual_seed(seed)
    random.seed(seed)

    tokenizer = Tokenizer(model_path='/home/michael/models/meta_llama2/tokenizer.model')

    process_dolly_dataset(
        src_file='/home/michael/datasets/dolly_15k/databricks-dolly-15k.jsonl',
        output_dir='./datasets/dolly',
        tokenizer=tokenizer,
    )

    process_alpaca_dataset(
        src_file='/home/michael/datasets/alpaca_dataset/alpaca_cleaned.json',
        output_dir='./datasets/alpaca',
        tokenizer=tokenizer,
    )

    process_squad_dataset(
        src_dir='/home/michael/datasets/squad_2.0',
        output_dir='./datasets/squad',
        tokenizer=tokenizer,
    )

    process_commonsense_dialog_dataset(
        src_dir='/home/michael/datasets/commonsense_dialogues',
        output_dir='./datasets/commonsense_dialogues',
        tokenizer=tokenizer,
    )

    process_deepmind_math_dataset(
        src_dirs=['/home/michael/datasets/mathematics_dataset-v1.0/train-easy'],
        output_dir='./datasets/deepmind_mathematics',
        tokenizer=tokenizer,
        max_samples=20000,
        filter_by_names=[
            'arithmetic__add_or_sub.txt',
            'arithmetic__add_sub_multiple.txt',
            'arithmetic__div.txt',
            'arithmetic__mul.txt',
        ],
    )

    # process_msc_dialog_dataset(
    #     src_dir='./raw_data/msc_dialogue',
    #     output_dir='./datasets/msc_dialogue',
    #     tokenizer=tokenizer,
    # )
