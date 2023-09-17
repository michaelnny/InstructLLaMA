# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from instruct_llama.utils.generation import Llama


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
            {'role': 'user', 'content': 'Tell me a joke about a dog.'},
        ],
        [
            {'role': 'user', 'content': 'What is the meaning of life?'},
        ],
        [
            {'role': 'user', 'content': 'Explain what is the theory of relativity.'},
        ],
        [
            {'role': 'user', 'content': 'Explain moon landing to a 8 years old kid.'},
        ],
        [
            {'role': 'user', 'content': 'Who is John F. Kennedy?'},
        ],
        [
            {'role': 'user', 'content': 'Was Donald Trump a great president? Why?'},
        ],
        [
            {'role': 'user', 'content': 'How do i get a child to stop holding food in their cheeks when eating?'},
        ],
        [
            {'role': 'user', 'content': "What is the best concept album that's ever been recorded?"},
            {
                'role': 'assistant',
                'content': 'One thing I might ask here is if you mean best in the sense of most enjoyable, or best in some other sense?',
            },
            {'role': 'user', 'content': 'I just mean best selling'},
        ],
        [
            {'role': 'user', 'content': "How can I stop worrying about things that don't affect me?"},
            {'role': 'assistant', 'content': 'What sorts of things are you worried about?'},
            {'role': 'user', 'content': 'Just different things that are outside of my control.'},
            {
                'role': 'assistant',
                'content': 'I see. You could try practicing mindfulness. Many people find that mindfulness is helpful for dealing with worries. What do you think about that?',
            },
            {'role': 'user', 'content': 'I could try that. Do you have any other suggestions?'},
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
        print(f"---> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
        print('\n==================================\n')


if __name__ == '__main__':
    main(
        # ckpt_dir="./meta_checkpoints/llama-2/llama-2-7b-chat",  # Meta fine-tuned chat model
        # ckpt_dir='./checkpoints/7b-sft',  # our fine-tuned chat model
        ckpt_dir='./checkpoints/7b-ppo',  # our RL trained chat model
        tokenizer_path='./meta_checkpoints/llama-2/tokenizer.model',
    )
