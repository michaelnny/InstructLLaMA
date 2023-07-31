import os
import json
import chardet
from typing import Iterable, List, Tuple, Mapping, Text, Any


def find_certain_files_under_dir(root_dir: str, file_type: str = '.txt') -> Iterable[str]:
    """Given a root folder, find all files in this folder and it's sub folders that matching the given file type."""
    assert file_type in ['.txt', '.jsonl']

    files = []
    if os.path.exists(root_dir):
        for root, dirnames, filenames in os.walk(root_dir):
            for f in filenames:
                if f.endswith(file_type):
                    files.append(os.path.join(root, f))
    return files


def _detect_encoding(input_file: str) -> str:
    encoding = 'utf-8'
    with open(input_file, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']

    return encoding


def read_txt_file(input_file: str, is_chinese: bool = False) -> Text:
    """Returns the raw text or None if input file not exists or is not .txt file."""
    if not os.path.exists(input_file) or not os.path.isfile(input_file) or not input_file.endswith('.txt'):
        return None

    encoding = 'utf-8'

    if is_chinese:
        encoding = _detect_encoding(input_file)

    try:
        with open(input_file, 'r', encoding=encoding) as file:
            text = file.read()
            return text
    except UnicodeDecodeError as e:  # noqa: F841
        # fallback to default encoding
        encoding = 'gb18030'
        print(f'Fallback to default encoding {encoding}')
        try:
            with open(input_file, 'r', encoding=encoding) as file:
                text = file.read()
                return text
        except Exception:
            print(f'Failed to read file {input_file}')
            return None


def read_json_file(input_file: str) -> Iterable[Mapping[Text, Any]]:
    """Returns json objects or None if input file not exists or is not .json file."""
    if not os.path.exists(input_file) or not os.path.isfile(input_file) or not input_file.endswith('.json'):
        return None

    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()
        return json.loads(content)


def read_jsonl_file(input_file: str) -> Iterable[Mapping[Text, Any]]:
    """Generator yields a list of json objects or None if input file not exists or is not .jsonl file."""
    if not os.path.exists(input_file) or not os.path.isfile(input_file) or not input_file.endswith('.jsonl'):
        return None

    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                yield json.loads(line.strip())
            except json.decoder.JSONDecodeError:
                print(f'Skip line in file {input_file}')
                continue


def count_words(raw_text, is_chinese=False) -> int:
    if raw_text is None or raw_text == '':
        return 0

    if is_chinese:
        return len(raw_text)

    return len(raw_text.split())
