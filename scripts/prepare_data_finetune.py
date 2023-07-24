"""
Module for prepare fine-tune raw data.
"""

from typing import Tuple, List, Mapping, Text
import functools
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import shutil
import json
import random


from instruct_llama.utils import (
    create_logger,
    find_certain_files_under_dir,
    read_txt_file,
    read_jsonl_file,
    count_words,
    build_prompt_completion,
    build_conversation_prompt_completions,
)


# check the official document from openAI
# https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset
NO_ANSWER = "Sorry, but there's no good answer to that question."


def process_squad_dataset(
    src_file,
    out_file,
    logger,
    overwrite_output=False,
):
    """Process SQuAD dataset and save to .jsonl format.

    Note we tread it as conversational style QA dataset."""
    assert src_file != out_file
    assert os.path.exists(src_file) and os.path.isfile(src_file)

    if os.path.exists(out_file) and os.path.isfile(out_file) and not overwrite_output:
        logger.error(f'The output file "{out_file}" already exists, aborting...')
        return

    content = None
    with open(src_file, "r", encoding="utf-8") as file:
        content = file.read()

    if content is None or 'data' not in content:
        logger.error(f'Invalid content from src file "{src_file}"')
        return

    # Create the output directory if necessary
    output_dir = os.path.dirname(out_file)
    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    parsed_json = json.loads(content)
    prompt_completion_pairs = []

    logger.info('Start to processing data, this may take few minute ...')

    for item in parsed_json['data']:
        for data in item['paragraphs']:
            context = data['context'].strip()

            pairs = []
            for i, qa in enumerate(data['qas']):
                prompt = qa['question'].strip()

                if len(prompt) == 0:
                    continue

                impossible = qa['is_impossible']
                completion = ""
                if impossible:
                    # randomly skip some of the questions without good answer
                    if random.random() < 0.9:
                        continue
                    else:
                        completion = NO_ANSWER
                else:
                    completion = qa['answers'][0]['text'].strip()

                if len(completion) == 0:
                    continue

                pairs.append({"prompt": prompt, "completion": completion})

                if len(pairs) >= 1:
                    formatted_prompt, formatted_completion = build_conversation_prompt_completions(pairs, context)

                    prompt_completion_pairs.append({'prompt': formatted_prompt, 'completion': formatted_completion})

                # only use last N turns to prevent prompt gets too long
                if len(pairs) > 8:
                    pairs.pop(0)

    if len(prompt_completion_pairs) > 0:
        logger.info(f'Saving {len(prompt_completion_pairs)} processed data to "{out_file}" ...')
        with open(out_file, 'w') as out_f:
            for obj in prompt_completion_pairs:
                jout = json.dumps(obj) + '\n'
                out_f.write(jout)


def process_marco_qna_dataset(
    src_file,
    out_file,
    logger,
    min_prompt_words: int = 6,
    overwrite_output=False,
):
    """Process SQuAD dataset and save to .jsonl format."""
    assert src_file != out_file
    assert os.path.exists(src_file) and os.path.isfile(src_file)

    if os.path.exists(out_file) and os.path.isfile(out_file) and not overwrite_output:
        logger.error(f'The output file "{out_file}" already exists, aborting...')
        return

    content = None
    with open(src_file, "r", encoding="utf-8") as file:
        content = file.read()

    if content is None:
        logger.error(f'Invalid content from src file "{src_file}"')
        return

    # Create the output directory if necessary
    output_dir = os.path.dirname(out_file)
    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    parsed_json = json.loads(content)

    passages_dict = parsed_json['passages']
    query_dict = parsed_json['query']
    answers_dict = parsed_json['answers']

    prompt_completion_pairs = []

    logger.info('Start to processing data, this may take few minute ...')

    for k in list(query_dict.keys()):
        context = ''

        for passage in passages_dict[k]:
            if passage['is_selected']:
                context = passage['passage_text']
                break

        prompt = query_dict[k].strip()
        completion = answers_dict[k][0].strip()

        if count_words(prompt) < min_prompt_words:  # skip bad samples where words are mixed together
            continue

        if len(completion) == 0:
            continue

        # MARCO QnA have a huge amount of questions without good answer,
        # so we randomly skip most of them
        if 'No Answer Present' in completion:
            if random.random() < 0.98:
                continue
            else:
                completion = NO_ANSWER

        formatted_prompt, formatted_completion = build_prompt_completion(prompt, completion, context)
        prompt_completion_pairs.append({'prompt': formatted_prompt, 'completion': formatted_completion})

    if len(prompt_completion_pairs) > 0:
        logger.info(f'Saving {len(prompt_completion_pairs)} processed data to "{out_file}" ...')
        with open(out_file, 'w') as out_f:
            for obj in prompt_completion_pairs:
                jout = json.dumps(obj) + '\n'
                out_f.write(jout)


def process_dolly_dataset(
    src_file,
    output_dir,
    logger,
    min_prompt_words: int = 6,
    eval_ratio=0.1,
    overwrite_output=False,
):
    """Process dolly dataset and save to .jsonl format.
    It will also split the dataset into train and evaluation subsets since the publicly free dolly only have one single json file.

    The content is formulated as follows:
    {"prompt":"<question>\n\n###\n\n", "completion":" <answer> END"}

    or if the question has context:
    {"prompt":"<context>\n\n\n\n<question>\n\n###\n\n", "completion":" <answer> END"}

    """

    assert os.path.exists(src_file) and os.path.isfile(src_file)
    assert 0 <= eval_ratio <= 0.2

    train_out_f = os.path.join(output_dir, 'train.jsonl')
    dev_out_f = os.path.join(output_dir, 'dev.jsonl')
    if any(os.path.exists(f) for f in (train_out_f, dev_out_f)) and not overwrite_output:
        logger.error(f'The output file "{train_out_f}" already exists, aborting...')
        return

    json_objs = read_jsonl_file(src_file)

    if json_objs is None:
        logger.error(f'Invalid content from src file "{src_file}"')
        return

    # Create the output directory if necessary
    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    prompt_completion_pairs = []

    for item in json_objs:
        context = item['context'].strip()
        prompt = item['instruction'].strip()
        completion = item['response'].strip()

        if count_words(prompt) < min_prompt_words:
            continue

        if len(completion) == 0:
            continue

        formatted_prompt, formatted_completion = build_prompt_completion(prompt, completion, context)
        prompt_completion_pairs.append({'prompt': formatted_prompt, 'completion': formatted_completion})

    # split into train and evaluation datasets as right now dolly only have one single .json file
    random.shuffle(prompt_completion_pairs)

    eval_idx = int(len(prompt_completion_pairs) * eval_ratio)

    eval_pairs = prompt_completion_pairs[:eval_idx]
    train_pairs = prompt_completion_pairs[eval_idx:]

    for data, out_f in zip((train_pairs, eval_pairs), (train_out_f, dev_out_f)):
        if len(data) > 0:
            logger.info(f'Saving {len(data)} processed data to "{out_f}" ...')
            with open(out_f, 'w') as f:
                for obj in data:
                    jout = json.dumps(obj) + '\n'
                    f.write(jout)


def process_common_dialogue_dataset(
    src_file,
    out_file,
    logger,
    overwrite_output=False,
):
    """Process Commonsense dialogues dataset and save to .jsonl format.

    Note we tread it as conversational style QA dataset. The commonsense dialogues dataset generally does not have context.

    The content is formulated as follows:
    {"prompt":"User: <message1>\nAgent:", "completion":" <response1> END"}

    or:
    {"prompt":"User: <message1>\nAgent: <response1>\nUser: <message2>\nAgent: <response2>\nUser: <message3>\nAgent:", "completion":" <response3> END"}
    """
    assert src_file != out_file
    assert os.path.exists(src_file) and os.path.isfile(src_file)

    if os.path.exists(out_file) and os.path.isfile(out_file) and not overwrite_output:
        logger.error(f'The output file "{out_file}" already exists, aborting...')
        return

    content = None
    with open(src_file, "r", encoding="utf-8") as file:
        content = file.read()

    if content is None or 'data' not in content:
        logger.error(f'Invalid content from src file "{src_file}"')
        return

    # Create the output directory if necessary
    output_dir = os.path.dirname(out_file)
    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    parsed_json = json.loads(content)
    prompt_completion_pairs = []

    logger.info('Start to processing data, this may take few minute ...')

    for item in parsed_json.values():
        pairs = []
        for i in range(0, len(item['turns']) - 1, 2):
            prompt = item['turns'][i].strip()
            completion = item['turns'][i + 1].strip()

            if len(prompt) == 0 or len(completion) == 0:
                continue

            pairs.append({"prompt": prompt, "completion": completion})

            if len(pairs) >= 1:
                formatted_prompt, formatted_completion = build_conversation_prompt_completions(pairs)

                prompt_completion_pairs.append({'prompt': formatted_prompt, 'completion': formatted_completion})

            # only use last N turns to prevent prompt gets too long
            if len(pairs) > 8:
                pairs.pop(0)

    if len(prompt_completion_pairs) > 0:
        logger.info(f'Saving {len(prompt_completion_pairs)} processed data to "{out_file}" ...')
        with open(out_file, 'w') as out_f:
            for obj in prompt_completion_pairs:
                jout = json.dumps(obj) + '\n'
                out_f.write(jout)


def process_dm_math_txt_file(txt_file):
    pairs = []

    with open(str(txt_file), 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')[:-1]

    for i in range(0, len(lines), 2):
        prompt = lines[i].strip()
        completion = lines[i + 1].strip()

        formatted_prompt, formatted_completion = build_prompt_completion(prompt, completion)
        pairs.append({'prompt': formatted_prompt, 'completion': formatted_completion})

    return pairs


def process_deepmind_math_dataset(
    src_dirs,
    output_dir,
    logger,
    filter_by_keys=[],  # only include files matching the keys defined here, if empty all files will be used
    num_workers=8,
    max_size=100000,  # limit total amount of samples to use
    eval_ratio=0.1,
    overwrite_output=False,
):
    """Process DeepMind mathematics dataset and save to .jsonl format.

    It will also split the samples into train and evaluation subsets.

    The content is formulated as follows:
    {"prompt":"<question>\n\n###\n\n", "completion":" <response1> END"}

    """
    assert len(src_dirs) > 0
    assert all(src_dir != output_dir and os.path.exists(src_dir) and os.path.isdir(src_dir) for src_dir in src_dirs)

    all_files = []
    working_files = []

    for src_dir in src_dirs:
        all_files.extend(find_certain_files_under_dir(src_dir, file_type='.txt'))

    # filter out file name not contain the keys
    if len(filter_by_keys) > 0:
        logger.info(f'Filter {len(all_files)} .txt files by key words {filter_by_keys} ...')
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
        logger.warning(f'Found no .txt file')
        return
    else:
        logger.info(f'Found {num_files} .txt files')

    if os.path.exists(output_dir):
        if len(os.listdir(output_dir)) != 0:
            # Remove the output directory and all it's content
            if overwrite_output:
                logger.info(f'cleanup output folder "{output_dir}"')
                shutil.rmtree(output_dir)
            else:
                logger.error(f'The output folder "{output_dir}" is not empty')
                return

    # Create the output directory if necessary
    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    logger.info(f'Processing {num_files} .txt files using {num_workers} workers ...')

    prompt_completion_pairs = []

    # Create a ProcessPoolExecutor with maximum N processes
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_dm_math_txt_file, file) for i, file in enumerate(working_files)]

        for future in as_completed(futures):
            pairs = future.result()
            prompt_completion_pairs.extend(pairs)

    logger.info(f'Finished processing {num_files} .txt files')

    # split into train and evaluation datasets
    random.shuffle(prompt_completion_pairs)

    # Limit amount of sample when fine-tuning
    if len(prompt_completion_pairs) > max_size:
        logger.info(f'Truncate data to max size of {max_size}')
        prompt_completion_pairs = prompt_completion_pairs[:max_size]

    eval_idx = int(len(prompt_completion_pairs) * eval_ratio)

    eval_pairs = prompt_completion_pairs[:eval_idx]
    train_pairs = prompt_completion_pairs[eval_idx:]

    train_out_f = os.path.join(output_dir, 'train.jsonl')
    dev_out_f = os.path.join(output_dir, 'dev.jsonl')
    if any(os.path.exists(f) for f in (train_out_f, dev_out_f)) and not overwrite_output:
        logger.error(f'The output file "{train_out_f}" already exists, aborting...')
        return

    for data, out_f in zip((train_pairs, eval_pairs), (train_out_f, dev_out_f)):
        if len(data) > 0:
            logger.info(f'Saving {len(data)} processed data to "{out_f}" ...')
            with open(out_f, 'w') as f:
                for obj in data:
                    jout = json.dumps(obj) + '\n'
                    f.write(jout)


if __name__ == "__main__":
    logger = create_logger()

    process_squad_dataset(
        src_file='./raw_data/SQuAD/train-v2.0.json',
        out_file='./clean_data/SQuAD/train_v2.0.jsonl',
        logger=logger,
        overwrite_output=False,
    )

    process_squad_dataset(
        src_file='./raw_data/SQuAD/dev-v2.0.json',
        out_file='./clean_data/SQuAD/dev_v2.0.jsonl',
        logger=logger,
        overwrite_output=False,
    )

    process_marco_qna_dataset(
        src_file='./raw_data/MARCO_QnA/train_v2.1.json',
        out_file='./clean_data/MARCO_QnA/train_v2.1.jsonl',
        logger=logger,
        overwrite_output=False,
    )

    process_marco_qna_dataset(
        src_file='./raw_data/MARCO_QnA/dev_v2.1.json',
        out_file='./clean_data/MARCO_QnA/dev_v2.1.jsonl',
        logger=logger,
        overwrite_output=False,
    )

    process_dolly_dataset(
        src_file='./raw_data/databricks-dolly-15k.jsonl',
        output_dir='./clean_data/dolly',
        logger=logger,
        overwrite_output=False,
    )

    process_common_dialogue_dataset(
        src_file='./raw_data/commonsense_dialogues/train.json',
        out_file='./clean_data/commonsense_dialogues/train.jsonl',
        logger=logger,
        overwrite_output=False,
    )

    process_common_dialogue_dataset(
        src_file='./raw_data/commonsense_dialogues/dev.json',
        out_file='./clean_data/commonsense_dialogues/dev.jsonl',
        logger=logger,
        overwrite_output=False,
    )

    process_deepmind_math_dataset(
        src_dirs=['./raw_data/mathematics_dataset-v1.0/train-easy'],
        output_dir='./clean_data/mathematics_dataset_v1.0',
        logger=logger,
        overwrite_output=False,
        filter_by_keys=['arithmetic__add_or_sub', 'arithmetic__add_sub_multiple', 'arithmetic__div', 'arithmetic__mul'],
        num_workers=20,
    )


