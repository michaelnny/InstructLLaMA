# InstructLLaMA

Implements pre-training, supervised fine-tuning (SFT), and reinforcement learning from human feedback (RLHF), to train and fine-tune the LLaMA2 model to follow human instructions, similar to InstructGPT or ChatGPT, but on a much smaller scale.

Check the [article post](https://www.vectortheta.com/blog/InstructLLaMA) on the discussion of the project.

This project use a custom QLoRA implementation with basic tools such as PyTorch and Bitsandbytes, decoupled from any Hugging Face tools. For more information on QLoRA fine-tuning, check my [article post](https://www.vectortheta.com/blog/QLoRA-LLM) and project at
[QLoRA-LLM](https://github.com/michaelnny/QLoRA-LLM)

# Disclaimer

**Project Purpose:** This project is dedicated to research and education, focusing on the study of individual algorithms rather than the creation of a standard library. If you're looking for a ready-to-use library for production applications, this project may not be suitable for your needs.

**Bug Reporting and Contributions:** Rigorous testing has been conducted in specific scenarios, but we cannot guarantee it's bug-free. Bug reports and pull requests are highly encouraged and welcomed.

**Optimization:** For simplicity, we only focus on training on a single GPU (except for pre-training script which support FSDP), as the PyTorch FSDP and QLoRA seems not working very well yet. Additionally, the hyper-parameters for the different training scripts are not fine-tuned.

# Environment and Requirements

- Python 3.10.6
- PyTorch 2.1.1
- Tensorboard 2.13.0
- Bitsandbytes 0.41.3

# Code Structure

- `instruct_llama` directory contains main source code for the project.
  - `configs` directory contains all the training configurations like model type, data source, number of iterations, learning rate etc.
  - `utils` directory contains helper modules like custom datasets, logging, tokenization, LoRA module etc.
  - `models` contains the LLaMA model class and LoRA layers.
  - `run_pretrain.py` run pre-training (supports FSDP and multiple GPUs).
  - `run_sft_lora.py` run supervised fine-tuning starting from Meta's pre-trained model, using LoRA parameter efficient fine-tuning method (only supports single GPU).
  - `run_rm_lora.py` train reward model starting from supervised fine-tuning model, using LoRA parameter efficient fine-tuning method (only supports single GPU).
  - `run_rlhf_lora.py` train policy and value models starting from supervised fine-tuning model and reward model respectively, all using LoRA parameter efficient fine-tuning method.
- `scripts` directory contains all source code for convert the model weights and build datasets for different phases.
  - `build_pretrain_datasets.py` build pre-train datasets (save the dataset in Numpy memmap structure), the dataset is optional during training using RLHF with PPO phase.
  - `build_finetune_datasets.py` build fine-tuning datasets (save the dataset to .pkl files).
  - `build_rm_comparison_datasets.py` build comparison datasets(save the dataset to .pkl files), which is used to train the reward model .
  - `build_rlhf_prompt_datasets.py` build prompt only datasets (save the dataset to .pkl files), which is used during RLHF PPO fine-tuning phase.
  - `convert_meta_checkpoint.py` convert Meta's pre-trained LLaMA-2 weights to support our model in plain PyTorch code, so we can load it to start fine-tuning.
  - `convert_lora_checkpoint.py` convert fine-tunned LoRA weights to a full state_dict checkpoint.
- `examples` directory contains the source code for text generation as well as chat completion, code adapted from the original LLaMA2 project.
- `logs` directory contains training logs for the different phases.

# Project Setup

```
python3 -m pip install --upgrade pip setuptools

python3 -m pip install -r requirements.txt
```

# Project Overview

## Preparation

Here are the steps required to utilize the project:

1. **Download the pre-trained model weights** please refer to https://github.com/facebookresearch/llama on how to download it.
2. **Convert Meta's pre-trained model weights** so it's compatible with our naming convention. Remember to change the file path before running it.

```
python3 scripts/convert_meta_checkpoint.py
```

3. **Download and build training datasets** For each training stage, we need to use different datasets, we're already prepared some of the common datasets. However, some datasets are too big to upload, so you may need to re-build it. Here's an example of build fine-tuning datasets, remember to change the file path before running it.

```
python3 scripts/build_finetune_datasets.py
```

## Training Stages

1. Run the `run_pretrain.py` script to train a LLaMA model from scratch. We only use it to train a 1B model so we can later use it as the reward model. Most of the time we would want to use Meta's pre-trained weights and skip this stage altogether.
2. Run the `run_sft_lora.py` script to fine-tune the model, this requires a pre-trained model, such as the one from Meta or from above pretrain stage. Check and maintain the configuration inside `instruct_llama/configs/sft_lora.py` if necessary.
3. Run the `run_rm_lora.py` script to train a reward model, this requires a fine-tuned model. Check and maintain the configuration inside `instruct_llama/configs/rm_lora.py` if necessary.
4. Run the `run_rlhf_lora.py` script to train a policy model using RLHF and PPO, this requires a fine-tuned model and the reward model (frozen). Check and maintain the configuration inside `instruct_llama/configs/rlhf_lora.py` if necessary.

### QLoRA Options

The training settings are in `instruct_llama/configs`. These files lets us choose which layers to train and the quantization methods.

### LoRA parameters

We use a slightly modified LoRALayer class, where we set the scaling directly instead of using an alpha parameter, we found this more consistent and easy to maintain. Since in most case, using a scaling of 1 makes more sense.

```
lora_r: int = 64
lora_scaling: float = 1.0  # set the LoRA scaling, by default 1.0 no scaling
lora_dropout: float = 0.0
```

### Trainable layers

For example, we can specify which layers in the model should be trainable using options like the ones below.

```
lora_attn_query: bool = True  # train Attention query layer
lora_attn_key: bool = False  # train Attention key layer
lora_attn_value: bool = True  # train Attention value layer
lora_attn_proj: bool = False  # train Attention projection layer
lora_attn_mlp: bool = False  # train Attention MLP block
```

One thing to mention is that we don't apply LoRA or quantization to the lm_head layer. But we're not sure if this helps improve the performance or not.

### Quantization layers

We have various quantization options. For instance, we can quantize only the frozen linear layers or both the frozen linear layers and trainable LoRA layers.

When quantizing a LoRA layer, only the pre-trained weights are quantized, while the LoRA parameters remain unchanged.

It's important to mention that our current support is limited to 4-bit quantization, and we utilize Bitsandbytes.

```
quant_4bit: bool = True  # quantize frozen linear layer
quant_lora_4bit: bool = True  # quantize LoRA linear layer
quant_4bit_double: bool = True  # double quantize
quant_4bit_type: str = 'nf4'  # only supports 'fp4' or 'nf4'
```

## Merge LoRA weights

Since we're using LoRA method, when the training is done (for each stage except pre-training), we need to merge the LoRA weights with the pre-trained or fine-tuned model weights. Which can be summarized into the following steps:

1. Construct a model with LoRA layers, matching the configuration used in fine-tuning but without quantized layers.
2. Load the pre-trained or fine-tuned weights.
3. Load the LoRA weights
4. Set the model to evaluation mode (model.eval()) to merge the weights. This triggers the LoRALinear.train() method, and making the merging process.
5. Remove any LoRA parameters from the state dict
6. Save the merged checkpoint

You can use the following script to do the conversion, remember to update the file path in the script accordingly.

```
python3 scripts/convert_lora_checkpoint.py
```

# Stage 1 - Pre-training

This stage is when we turn a randomly initialized model into a one that can predict the next token (often called language modeling), this is often the most time and resource consuming phase. This requires a large amount of tokens and GPU power. Most of the time we'd want to use ready-to-use pre-trained model weights, for example from Meta.

It's important to mention, for pret-raining we can't use LoRA or QLoRA to reduce GPU resource. This is also the only stage (in this project) where we can utilize Pytorch FSDP and multiple GPUs to speed up training.

```
torchrun --nproc_per_node 1 instruct_llama/run_pretrain.py
```

# Stage 2 - Supervised Fine-Tuning (SFT)

This stage is when we turn a pre-trained language model from predicting next token to answer general questions, in a chat formation. This is also referred as the prompt completion, where the model is feed a prompt (user request), and it needs to generate the corresponding completion (answer).

Once we have a pre-trained model and the fine-tuning datasets are ready, we can start doing supervised fine-tuning using LoRA or QLoRA.

```
python3 instruct_llama/run_sft_lora.py
```

Keep in mind we need to merge the LoRA weights after the training, refer to the instruction mentioned in **Merge LoRA weights** on how to do it.

# Stage 3 - Train Reward Model (RM)

After the fine-tuned phase is done, and the RM comparison datasets are ready, we can start training a reward model using using LoRA or QLoRA.

Training the reward model involves using a comparison dataset, which has the following structure (a single sample):

- a single prompt -> a list of 2 or more completions, where the best one is at index 0, and the worst one is the last in the list

And the objective is train the model to assign a higher reward to the best completion and a lower reward to the worst one.

This training phase demands more GPU RAM compared to fine-tuning, as it involves maintaining multiple computation graphs/gradients during loss computation. To save computation, we use a smaller model with only 3 billion parameters. This model uses the first 12 transformer blocks from the fine-tuned 7B model, and the LM head is replaced with a linear layer that outputs a scalar score for the reward signal. This new scalar head is then jointly trained with the rest of LoRA/QLoRA layers.

```
python3 instruct_llama/run_rm_lora.py
```

Keep in mind we need to merge the LoRA weights after the training, refer to the instruction mentioned in **Merge LoRA weights** on how to do it.

# Stage 4 - RLHF with PPO

The last stage is to train the model using RLHF and the PPO algorithm. This is the most complex part of the project, where it involves lots of moving parts.

The goal of this stage is to train the policy model so that we can get higher reward for the completions. Here's an overview of the training pipeline:

```
while not converged:
  use a prompt only datasets and RL self-play to generate a large batch of L sample episodes

  for each episode in L:
    use the RM model to assign a reward signal according to the completion tokens.
    use the SFT model to compute a pre-token KL penalty for the completion tokens as part of the reward signal

  for each PPO training epoch:
    using PPO to update the policy and value networks based on the L samples
```

Here's an overview of the models involved in this stage:

1. A policy model with LM head initialized from the fine-tuned checkpoint, this is the model we want to optimize, as we often refer to it as policy network (or actor) in RL and PPO
2. A value model with scalar head initialized from the trained RM checkpoint, this is the model we want to optimize, as we often refer to it as value network (or critic) in RL and PPO
3. A reward model with scalar head initialized from the trained RM checkpoint, this model is fixed (frozen) and we only use it to assign rewards to the completions generated by the RL agent during self-play
4. A SFT model with LM head initialized from the fine-tuned checkpoint, this model is fixed (frozen) and we only use it compute pre-token KL penalty for reward.

As we need to run multiple models at the same time, this demands more GPU resource than any of the previous stages. If you have multiple GPUs then you can set the model devices inside the `instruct_llama/configs/rlhf_lora.py` module. Thanks to 4bit quantization and small-sized reward model (3B), when we use 7B model for policy and STF models, we can fit all these 4 models on a single RTX 3090 with 24GB GPU RAM during inference and self-play.

We can use the following script to launch the RLHF training session. Note in RL, we often need large amount of self-play episodes before we can observer significant performance gains.

```
python3 instruct_llama/run_rlhf_lora.py
```

After the training is done, we can merge the trained policy model with the pre-trained one to get a much better model.
Keep in mind we need to merge the LoRA weights after the training, refer to the instruction mentioned in **Merge LoRA weights** on how to do it.

# Monitoring with tensorboard

We can monitoring the training progress by using Tensorboard:

```
tensorboard --logdir=./logs
```

# License

This project is licensed under the MIT License (the "License")
see the LICENSE file for details

The LLaMA2 model weights are licensed for both researchers and commercial entities. For details, visit: https://github.com/facebookresearch/llama#license.

# Acknowledgments

This project is greatly influenced by the following projects:

- [Llama 2] (https://github.com/facebookresearch/llama)
- [lm-human-preferences] (https://github.com/openai/lm-human-preferences)
- [Lit-LLaMA] (https://github.com/Lightning-AI/lit-llama)
- [LoRA] (https://github.com/microsoft/LoRA)
- [QLoRA-LLM] (https://github.com/michaelnny/QLoRA-LLM)
