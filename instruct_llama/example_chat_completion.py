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
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        device='cuda',
    )

    #     # Meta fine-tuned chat model
    #     dialogs = [
    #         [{'role': 'user', 'content': 'what is the recipe of mayonnaise?'}],
    #         [
    #             {'role': 'user', 'content': 'I am going to Paris, what should I see?'},
    #             {
    #                 'role': 'assistant',
    #                 'content': """\
    # Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

    # 1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
    # 2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
    # 3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

    # These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
    #             },
    #             {'role': 'user', 'content': 'What is so great about #1?'},
    #         ],
    #         [
    #             {'role': 'system', 'content': 'Always answer with Haiku'},
    #             {'role': 'user', 'content': 'I am going to Paris, what should I see?'},
    #         ],
    #         [
    #             {
    #                 'role': 'system',
    #                 'content': 'Always answer with emojis',
    #             },
    #             {'role': 'user', 'content': 'How to go from Beijing to NY?'},
    #         ],
    #         [
    #             {'role': 'user', 'content': 'Tell me a joke about a dog.'},
    #         ],
    #         [
    #             {'role': 'user', 'content': 'What is the meaning of life?'},
    #         ],
    #         [
    #             {'role': 'user', 'content': 'Explain what is the theory of relativity.'},
    #         ],
    #         [
    #             {'role': 'user', 'content': 'Who is John F. Kennedy?'},
    #         ],
    #     ]

    dialogs = [
        [
            {'role': 'user', 'content': "What is the best concept album that's ever been recorded?"},
            {
                'role': 'assistant',
                'content': 'One thing I might ask here is if you mean best in the sense of most enjoyable, or best in some other sense?',
            },
            {'role': 'user', 'content': 'I just mean best selling'},
            {
                'role': 'assistant',
                'content': 'Well there is “Thriller” by Michael Jackson. It’s arguably one of the best-selling albums of all time, it’s very popular among many people, and also it has a large number of positive critical reviews.',
            },
            {'role': 'user', 'content': 'It looks like The Wall By Pink Floyd is the best concept album. Is that true?'},
            {
                'role': 'assistant',
                'content': 'That does look like the most popular bestseller on this topic, yes. Pink Floyd was a progressive rock band from the 1970s, and their main songwriter was a guy named Roger Waters. He also wrote some lyrics about a “wall”, and I think there’s also something about it being “The Wall”. Do you want me to tell you more about this?',
            },
            {'role': 'user', 'content': 'Yea, can you tell me where the idea came from?'},
        ],
        [
            {'role': 'user', 'content': 'Was Donald Trump a great president?'},
            {
                'role': 'assistant',
                'content': 'No, he was terrible and unpresidential. We might never have a successful President again if he gets reelected.',
            },
            {'role': 'user', 'content': 'Why was Donald Trump such a bad president?'},
        ],
        [
            {'role': 'user', 'content': 'how do i get a child to stop holding food in their cheeks when eating?'},
            {
                'role': 'assistant',
                'content': 'How much food is in their cheeks and is it causing distress? If the food is just barely in their cheek but not causing them any problems, you may be able to persuade them to let it go after a brief discussion, just by saying something like, “We don’t hold food in our cheeks when we eat. Let’s just try taking it out with your hand now and eat it properly. What do you think?”  If the food is too large or stuck too far in the cheek, it might be best just to remove it without talking. And if it’s causing distress (in the form of mouth, tongue, or cheek pain, for example) you should absolutely remove it as soon as possible. If it’s an emergency situation, this would be an appropriate time to contact emergency medical services. They will assess the situation and tell you whether or not to try to remove the food.',
            },
            {'role': 'user', 'content': 'He always holds food in his cheeks and it takes him minutes to swallow'},
        ],
        [
            {'role': 'user', 'content': "What is the best concept album that's ever been recorded?"},
            {
                'role': 'assistant',
                'content': 'One thing I might ask here is if you mean best in the sense of most enjoyable, or best in some other sense?',
            },
            {'role': 'user', 'content': 'I just mean best selling'},
            {
                'role': 'assistant',
                'content': 'Well there is “Thriller” by Michael Jackson. It’s arguably one of the best-selling albums of all time, it’s very popular among many people, and also it has a large number of positive critical reviews.',
            },
            {'role': 'user', 'content': 'It looks like The Wall By Pink Floyd is the best concept album. Is that true?'},
            {
                'role': 'assistant',
                'content': 'That does look like the most popular bestseller on this topic, yes. Pink Floyd was a progressive rock band from the 1970s, and their main songwriter was a guy named Roger Waters. He also wrote some lyrics about a “wall”, and I think there’s also something about it being “The Wall”. Do you want me to tell you more about this?',
            },
            {'role': 'user', 'content': 'Yea, can you tell me where the idea came from?'},
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
            {
                'role': 'assistant',
                'content': 'Would it be helpful to talk to a therapist about it? I have a list of people who are trained in psychotherapy.',
            },
            {'role': 'user', 'content': "I might eventually try that if other methods don't help much."},
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
        # ckpt_dir="./meta_checkpoints/llama-2/llama-2-7b-chat",  # Meta fine-tuned chat model
        # ckpt_dir='./checkpoints/7b-finetune',  # our fine-tuned chat model
        ckpt_dir='./checkpoints/7b-ppo',  # our RL trained chat model
        tokenizer_path='./meta_checkpoints/llama-2/tokenizer.model',
        temperature=1.0,
    )
