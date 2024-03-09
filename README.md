# InstructLLaMA

Implements pre-training, supervised fine-tuning (SFT), and reinforcement learning from human feedback (RLHF), to train and fine-tune the LLaMA2 model to follow human instructions, similar to InstructGPT or ChatGPT, but on a much smaller scale.

Check the [article post](https://www.vectortheta.com/blog/InstructLLaMA) on the discussion of the project.

# Disclaimer

**Project Purpose:** This project is dedicated to research and education, focusing on the study of individual algorithms rather than the creation of a standard library. If you're looking for a ready-to-use library for production applications, this project may not be suitable for your needs.

**Bug Reporting and Contributions:** Although regular tests have been conducted, we cannot guarantee the code is bug-free. Bug reports and pull requests are highly encouraged and welcomed.

**Optimization:** Due to limited GPU resources, we only focus on training on a single GPU. Additionally, the hyper-parameters for the different training scripts are not fine-tuned.

# Environment and Requirements

- Python 3.10.6
- PyTorch 2.1.1
- Tensorboard 2.13.0
- Bitsandbytes 0.41.3

# Code Structure

- `instruct_llama` directory contains main source code for the project.

  - `configs` directory contains all the training configurations like model type, data source, number of iterations, learning rate etc.
  - `cores` directory contains core modules like custom datasets, RL PPO agent etc.
  - `models` directory contains the LLaMA model class and LoRA layers.
  - `utils` directory contains helper modules like logging, checkpointing etc.
  - `run_pretrain.py` run (full-scale) pre-training starting from a random model (supports FSDP and multiple GPUs).
  - `run_sft.py` run (full-scale) supervised fine-tuning starting from pre-trained model(only supports single GPU).
  - `run_sft_lora.py` basically the same as `run_sft.py`, but support 4-bit QLoRA (only supports single GPU).
  - `run_rm.py` train reward model (full-scale) starting from supervised fine-tuning model (only supports single GPU).
  - `run_rlhf.py` run (full-scale) RL PPO to train policy and value models, starting from supervised fine-tuning model and reward model respectively (supports allocate different models on different GPUs on a single machine, but no DDP or FSDP).

- `scripts` directory contains all source code for convert the model weights and build datasets for different phases.
  - `build_pretrain_datasets.py` build pre-train datasets (save the dataset in Numpy memmap structure), the dataset is optional during training using RLHF with PPO phase.
  - `build_finetune_datasets.py` build fine-tuning datasets (save the dataset to .pkl files).
  - `build_rm_comparison_datasets.py` build comparison datasets(save the dataset to .pkl files), which is used to train the reward model .
  - `build_rlhf_prompt_datasets.py` build prompt only datasets (save the dataset to .pkl files), which is used during RLHF PPO fine-tuning phase.
  - `convert_meta_checkpoint.py` convert Meta's pre-trained LLaMA-2 weights to support our model in plain PyTorch code, so we can load it to start fine-tuning.
  - `convert_lora_checkpoint.py` convert fine-tunned LoRA weights to a full state_dict checkpoint.
- `inference` directory contains the code to do inference, which was adapted from the original LLaMA2 project.
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
2. Run the `run_sft.py` or `run_sft_lora.py` script to fine-tune the model, this requires a pre-trained model, such as the one from Meta or from above pretrain stage. Check and maintain the configuration inside `configs/sft.py` and `configs/sft_lora.py` if necessary.
3. Run the `run_rm.py` script to train a reward model, this requires a fine-tuned model. Check and maintain the configuration inside `configs/rm.py` if necessary.
4. Run the `run_rlhf.py` script to train a policy model using RLHF and PPO, this requires a fine-tuned model and the reward model (frozen). Check and maintain the configuration inside `configs/rlhf.py` if necessary.

It's important to mention that the training prompts in RLHF stage must coming from the same data distribution as the samples used to train the reward model. Otherwise, the reward signal would be noise and the model will collapse. In addition, the fine-tuning stage should also contain the same sample distribution, since we need the SFT model as a reference policy to compute KL penalties.

# Stage 1 - Pre-training (full-scale FSDP)

This stage is when we turn a randomly initialized model into a one that can predict the next token (often called language modeling), this is often the most time and resource consuming phase. This requires a large amount of tokens and GPU power. Most of the time we'd want to use ready-to-use pre-trained model weights, for example from Meta.

It's important to mention, for pret-raining we can't use LoRA or QLoRA to reduce GPU resource. This is also the only stage (in this project) where we can utilize Pytorch FSDP and multiple GPUs to speed up training.

```
torchrun --nproc_per_node 1 instruct_llama/run_pretrain.py
```

# Stage 2 - Supervised Fine-Tuning (full-scale or 4-bit QLoRA)

This stage is when we turn a pre-trained language model from predicting next token to answer general questions, in a chat formation. This is also referred as the prompt completion, where the model is feed a prompt (user request), and it needs to generate the corresponding completion (answer).

Once we have a pre-trained model and the fine-tuning datasets are ready, we can start doing supervised fine-tuning using full-scale. We start with Meta's pre-trained 7B model, the model context window is limited to 512, and use the `hh-rlhf helpful-base` dataset which consists of 41k samples to train the model over 2 epochs.

```
python3 instruct_llama/run_sft.py
```

We can also use the following script for single node with 4-bit QLoRA.

```
python3 instruct_llama/run_sft_lora.py
```

## QLoRA Options

The training settings are in `configs/sft_lora`. These files lets us choose which layers to train and the quantization methods. Note if using 4-bit quantized linear layers, the training speed will slow down 50%~70%, depending on the configurations.

## Merge LoRA weights

Since we're using LoRA method, when the training is done, we need to merge the LoRA weights with the pre-trained or fine-tuned model weights. Which can be summarized into the following steps:

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

# Stage 3 - Train Reward Model (full-scale)

After the RM comparison datasets are ready, we can start training a reward model.

Training the reward model involves using a comparison dataset, which has the following structure (a single sample):

- a single prompt -> a list of 2 or more completions, where the best one is at index 0, and the worst one is the last in the list

And the objective is to train the model to assign a higher reward to the best (chosen) completion and a lower reward to the worser (rejected) ones.

This training phase demands more GPU RAM compared to fine-tuning, as it involves maintaining multiple computation graphs/gradients during loss computation. The reward model can be initialized from the fine-tuned 7B model, or the pretrained 7B model, where the LM head is replaced with a linear layer that outputs a scalar score. This new scalar head is then jointly trained with the rest of decoder layers. To save computation, we have the option to freeze first N decoder layers.

We don't use LoRA when train the reward model since the scalar head layer is initialized randomly, full-scale training tends to yields better results. We start with Meta's pre-trained 7B model, the model context window is limited to 512, and use the `hh-rlhf helpful-base` dataset which consists of 41k samples to train the model over 1 epoch.

```
python3 instruct_llama/run_rm.py
```

Although the above script only supports training on a single GPU, it can be extended to support multiple GPUs and FSDP, please reference `run_pretrain.py` on how to do FSDP.

# Stage 4 - RLHF with PPO (full-scale)

The last stage is to train the model using the PPO algorithm. This is the most complex part of the project, where it involves lots of moving parts.

The goal of this stage is to train the policy model so that we can get higher reward for the completions when given some prompt as input. Here's an overview of the training pipeline:

```
while not converged:
  use a prompt only datasets and RL self-play to generate a large batch of L sample episodes while following the current PPO policy

  for each episode in L:
    use the RM model to assign a reward signal according to the response tokens.
    use the SFT model to compute a pre-token KL penalty for the response tokens as part of the reward signal

  for each PPO training epoch:
    using PPO to update the policy and value networks based on the L samples
```

Here's an overview of the models involved in this stage:

1. A policy model with LM head initialized from the fine-tuned checkpoint, this is the model we want to optimize, as we often refer to it as policy network (or actor) in RL and PPO
2. A value model with scalar head initialized from the trained RM checkpoint, this is the model we want to optimize, as we often refer to it as value network (or critic) in RL and PPO
3. A reward model with scalar head initialized from the trained RM checkpoint, this model is fixed (frozen) and we only use it to assign rewards to the completions generated by the RL agent during self-play
4. A SFT model with LM head initialized from the fine-tuned checkpoint, this model is fixed (frozen) and we only use it compute pre-token KL penalty for reward.

As we need to run multiple models at the same time, this demands more GPU resource than any of the previous stages. If you have multiple GPUs then you can set the model devices inside the `configs/rlhf.py` module. When using 7B model for policy, reward, and value models, we can fit all these 4 models on a single RTX 3090 with 24GB GPU RAM during self-play by swapping these models between GPU and CPU, training is possible by frozen most of the layers in the model.

We can use the following script to launch the RLHF full-scale training session. To save computation, we have the option to freeze first N decoder layers for both PPO policy and value models. We start with SFT trained 7B model from stage 2, and the reward model from stage 3, the model context window is limited to 512, and use the same `hh-rlhf helpful-base` 41k dataset to train the policy model.

```
python3 instruct_llama/run_rlhf.py
```

# Monitoring with tensorboard

We can monitoring the training progress by using Tensorboard:

```
tensorboard --logdir=./logs
```

It's very beneficial to monitor the generated samples during RL training. To do so, go to Tensorboard - Text, we can then check the prompt text, model generated response, and the corresponding scalar reward. This works better if we enable MarkDown in Tensorboard.

# License

This project is licensed under the MIT License (the "License")
see the LICENSE file for details

For details about LLaMA2 model weights license, visit: https://github.com/facebookresearch/llama#license

# Acknowledgments

This project is greatly influenced by the following projects:

- [Llama 2] (https://github.com/facebookresearch/llama)
- [lm-human-preferences] (https://github.com/openai/lm-human-preferences)
- [Lit-LLaMA] (https://github.com/Lightning-AI/lit-llama)
- [LoRA] (https://github.com/microsoft/LoRA)
- [QLoRA-LLM] (https://github.com/michaelnny/QLoRA-LLM)
