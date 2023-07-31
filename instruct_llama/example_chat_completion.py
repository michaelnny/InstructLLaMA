# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from instruct_llama.generation import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 16,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        device='cuda',
    )

    # Meta fine-tuned chat model
    dialogs = [
        [{'role': 'user', 'content': 'what is the recipe of mayonnaise?'}],
        [
            {'role': 'user', 'content': 'I am going to Paris, what should I see?'},
            {
                'role': 'assistant',
                'content': """\
Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
            },
            {'role': 'user', 'content': 'What is so great about #1?'},
        ],
        [
            {'role': 'system', 'content': 'Always answer with Haiku'},
            {'role': 'user', 'content': 'I am going to Paris, what should I see?'},
        ],
        [
            {
                'role': 'system',
                'content': 'Always answer with emojis',
            },
            {'role': 'user', 'content': 'How to go from Beijing to NY?'},
        ],
    ]

    # our fine-tuned instruction model
    dialogs = [
        [
            {'role': 'user', 'content': 'Tell me a joke about a dog.'},
        ],
        [
            {'role': 'user', 'content': 'What is the meaning of life?'},
        ],
        [
            {'role': 'user', 'content': 'Explain what is the theory of relativity.'},
        ],
        [
            {'role': 'user', 'content': 'Who is John F. Kennedy?'},
        ],
        [
            {'role': 'user', 'content': 'Add 12 and 146669'},
        ],
        [
            {'role': 'user', 'content': 'Calculate 153 + -0.05'},
        ],
        [
            {'role': 'user', 'content': 'Calculate 21 times -2.1'},
        ],
        [
            {'role': 'user', 'content': 'I am going to Paris, what should I see?'},
            {
                'role': 'assistant',
                'content': """\
Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
            },
            {'role': 'user', 'content': 'What is so great about #1?'},
        ],
    ]

    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
        print('\n==================================\n')


if __name__ == '__main__':
    main(
        # ckpt_dir="./checkpoints/llama-2/llama-2-7b-chat",  # Meta fine-tuned chat model
        ckpt_dir='./checkpoints/7b-finetune',  # our fine-tuned chat model
        tokenizer_path='./checkpoints/llama-2/tokenizer.model',
    )
