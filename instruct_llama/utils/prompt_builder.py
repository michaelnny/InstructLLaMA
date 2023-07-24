"""Build prompt-completion pairs using LLaMA-2 predefined chat style format.

We rewrite the original code to make it easier to understand, we can also use the code to build fine-tuning samples.
"""
from typing import Tuple, List, Mapping, Text, Literal, Optional, Tuple, TypedDict

from instruct_llama.tokenizer import Tokenizer

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


def maybe_add_system_prompt(dialog: Dialog) -> Dialog:
    """Insert a dialog with system role at the beginning of the list if there're no such one."""
    assert dialog is not None and len(dialog) > 0

    if dialog[0]["role"] != "system":
        dialog = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + dialog

    return dialog


def build_prompt_completion(
    dialog: Dialog, tokenizer: Tokenizer
) -> Tuple[List[int], List[int]]:
    """Build prompt and completion pair following the Meta llama-2 format.

    Note we only build the training target completion if the last role in the dialog is 'assistant'.

    Here are some examples of the format that llama-2 uses (before apply tokenization), note we inserted BOS and EOS into the example for completeness:
    {"prompt": "BOS [INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} [/INST]", "completion": " {1st response} EOS"}
    {"prompt": "BOS [INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} [/INST] {1st response} EOS BOS[INST] {2nd user prompt} [/INST]", "completion": " {2nd response} EOS"}

    """

    assert dialog is not None and len(dialog) > 0

    # make sure the first one is always system prompt
    dialog = maybe_add_system_prompt(dialog)

    assert (
        dialog[0]["role"] == "system"
        and all([msg["role"] == "user" for msg in dialog[1::2]])
        and all([msg["role"] == "assistant" for msg in dialog[2::2]])
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )

    # store user-prompt:answer pairs so we can later add BOS, EOS tokens to each pair
    prompts = []
    for i in range(1, len(dialog), 2):  # skip the first one since it's system prompt
        full_prompt = ""
        prompt = dialog[i]

        if i == 1:
            # add system prompt, note Meta llama-2 insert the system prompt inside the first user prompt
            # as in this format: [INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} [/INST]
            sys_prompt = B_SYS + dialog[0]["content"] + E_SYS
            full_prompt = f"{B_INST} {sys_prompt}{(prompt['content']).strip()} {E_INST}"
        else:
            full_prompt = f"{B_INST} {(prompt['content']).strip()} {E_INST}"

        # add answer to the full prompt
        # here we skip the last answer by the assistant, since it's used for building the training target
        if i + 1 < len(dialog) - 1 and dialog[i + 1]["role"] == "assistant":
            answer = dialog[i + 1]
            full_prompt += f" {(answer['content']).strip()} "

        prompts.append(full_prompt)

    prompt_tokens = []

    # Note llama-2, we add BOS and EOS for every pair of user-prompt:answer, except for the last user-prompt
    for i, prompt in enumerate(prompts):
        if i < len(prompts) - 1:
            tokens = tokenizer.encode(prompt, bos=True, eos=True)
        else:
            tokens = tokenizer.encode(prompt, bos=True, eos=False)
        prompt_tokens.extend(tokens)

    # completion tokens for training
    completion_tokens = None
    if dialog[-1]["role"] == "assistant":
        answer = dialog[-1]
        target = f" {(answer['content']).strip()} "
        completion_tokens = tokenizer.encode(target, bos=False, eos=True)

    return prompt_tokens, completion_tokens


if __name__ == "__main__":
    tokenizer = Tokenizer(model_path="/home/michael/llama-2/tokenizer.model")

    example_dialogs = [
        [
            {"role": "user", "content": "Solve 1+37."},
            {"role": "assistant", "content": "38"},
        ],
        [
            {"role": "user", "content": "Tell me a joke about a dog."},
        ],
        [
            {
                "role": "system",
                "content": "You are a very clever and funny agent, make people laugh is your natural job.",
            },
            {
                "role": "user",
                "content": "Tell me a joke about a cat playing some toy car.",
            },
        ],
        [
            {"role": "user", "content": "I am going to Paris, what should I see?"},
            {"role": "assistant", "content": "You should go to The Eiffel Tower."},
            {"role": "user", "content": "What's so special about it?"},
        ],
        [
            {"role": "user", "content": "I am going to Paris, what should I see?"},
            {"role": "assistant", "content": "You should go to The Eiffel Tower."},
            {"role": "user", "content": "What's so special about it?"},
            {"role": "assistant", "content": "Just go there and you'll find out."},
        ],
    ]

    for dialog in example_dialogs:
        prompt_tokens, completion_tokens = build_prompt_completion(
            dialog,
            tokenizer,
        )

        print(f"Prompt: {tokenizer.decode(prompt_tokens)}")

        if completion_tokens is not None:
            print(f"Completion: {tokenizer.decode(completion_tokens)}")
        print("\n\n")
