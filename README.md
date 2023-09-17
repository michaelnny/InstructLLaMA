# InstructLLaMA
Implements SFT and RLHF with PPO to fine-tune and train LLaMA2 model to follow human instructions, similar to InstructGPT or ChatGPT.

This project is for education and research purpose only. Where we focus on studying the algorithm rather than creating a standard library. If you're looking for a ready to use library for your productive application, this is probably the wrong place.


# What we got
* support supervised fine-tuning (SFT), train rewrad model (RM), and train the policy using RLHF with PPO algorithm
* support PyTorch FSDP for distributed training (only for training SFT and RM phases)
* scripts for building fine-tuning, comparison, and prompt only datasets for different training stages


# Environment and Requirements
* Python        3.10.6
* PyTorch       2.0.1
* Tensorboard   2.13.0


# Code Structure
*   `instruct_llama` directory contains all source code for training the model.
    *   `configs` directory contains all the training configurations like model type, data source, number of iterations, learning rate etc.
    *   `utils` directory contains helper modules like custom datasets, logging, tokenization, LoRA module etc.
    *   `model.py` contains the LLaMA model class.
    *   `train_sft_lora.py` contains the code to run supervised fine-tuning starting from Facebook's pretrained model, using LoRA parameter efficient fine-tuning method (only supports single GPU).
    *   `train_sft_fsdp_lora.py` contains the code to run supervised fine-tuning starting from Facebook's pretrained model, using LoRA parameter efficient fine-tuning method (supports multiple GPU, but not fully tested).
    *   `train_rm_lora.py` contains the code to train reward model starting from supervised fine-tuning model, using LoRA parameter efficient fine-tuning method (only supports single GPU).
    *   `train_rm_fsdp_lora.py` contains the code to train reward model starting from supervised fine-tuning model, using LoRA parameter efficient fine-tuning method (supports multiple GPU, but not fully tested).
    *   `train_ppo_lora.py` contains the code to train policy and value models starting from supervised fine-tuning model and reward model respectively, all using LoRA parameter efficient fine-tuning method (only supports single GPU).
*   `scripts` directory contains all source code for convert the model weights and build datasets for different phases.
    *   `build_pretrain_datasets.py` contains code to build pre-train datasets (save the dataset in Numpy memmap structure), the dataset is optional during training using RLHF with PPO phase.
    *   `build_finetune_datasets.py` contains code to build fine-tuning datasets (save the dataset to .jsonl files).
    *   `build_rl_comparison_datasets.py` contains code to build comparison datasets(save the dataset to .jsonl files), which is used to train the reward model .
    *   `build_rl_prompt_datasets.py` contains code to build prompt only datasets (save the dataset to .jsonl files), which is used during RLHF PPO fine-tuning phase.
    *   `convert_meta_checkpoint.py` contains code to convert Facebook pre-trained LLaMA-2 weights to support our model in plain PyTorch code, so we can load it to start fine-tuning.
    *   `convert_lora_checkpoint.py` contains code to convert fine-tunned LoRA weights to a full state_dict checkpoint.
*   `examples` directory contains the source code for text generation as well as chat completion, code adapted from the original LLaMA2 project.
*   `logs` directory contains training logs for the different phases.

# Project Setup

```
python3 -m pip install --upgrade pip setuptools

python3 -m pip install -r requirements.txt
```


# Download and prepare datasets
You need download the source files for the individual dataset from the Internet, then using our data preparation and build dataset scripts to turn them into ready to use datasets. We don't provide any ready to use dataset files.

Once you have downloaded the source files, use the `build_dataset` scripts in `scripts` folder  to build the training datasets.


# Supervised Fine-Tuning (SFT)

Once we have a pre-trained model and the fine-tuning datasets are ready, we can start doing supervised fine-tuning.

We provide two options (both using LoRA) to do fine-tuning:
1. LoRA fine-tuning on a single GPU (what we use during the project)
2. LoRA fine-tuning on multiple GPUs with FSDP (not fully tested, use it at your own risk)

## Single GPU
```
torchrun --standalone --nproc_per_node 1 instruct_llama/train_sft_lora.py
```

## Multiple GPU
```
torchrun --nproc_per_node 4 instruct_llama/train_sft_fsdp_lora.py
```

We can monitoring the progress by using Tensorboard:
```
tensorboard --logdir=./logs
```

![SFT Tensorboard](/screenshots/sft_logs.png)


# Train Reward Model (RM)

Once we have a fine-tuned model and the comparison datasets are ready, we can start training a reward model.

We provide two options (both using LoRA) to train the reward model:
1. LoRA fine-tuning on a single GPU (what we use during the project)
2. LoRA fine-tuning on multiple GPUs with FSDP (not fully tested, use it at your own risk)

## Single GPU
```
torchrun --standalone --nproc_per_node 1 instruct_llama/train_rm_lora.py
```

## Multiple GPU
```
torchrun --nproc_per_node 4 instruct_llama/train_rm_fsdp_lora.py
```

We can monitoring the progress by using Tensorboard:
```
tensorboard --logdir=./logs
```

![RM Tensorboard 1](/screenshots/rm_logs_1.png)
![RM Tensorboard 2](/screenshots/rm_logs_2.png)


# RLHF with PPO

The last stage is to train the policy using RLHF and the PPO algorithm.

For simplicity reasons, we only provide training the model on a single GPU.


## Single GPU
```
torchrun --standalone --nproc_per_node 1 instruct_llama/train_ppo_lora.py
```

We can monitoring the progress by using Tensorboard:
```
tensorboard --logdir=./logs
```

![RLHF PPO Tensorboard 1](/screenshots/ppo_logs_1.png)
![RLHF PPO Tensorboard 2](/screenshots/ppo_logs_2.png)


# License
This project is licensed under the MIT License (the "License")
see the LICENSE file for details

Note, the original LLaMA2 model weights are only licensed for both researchers and commercial entities, for more information please check:
https://github.com/facebookresearch/llama#license


# Acknowledgments

This project is greatly influenced by the following projects:
* [Llama 2] (https://github.com/facebookresearch/llama)
* [lm-human-preferences] (https://github.com/openai/lm-human-preferences)
* [Lit-LLaMA] (https://github.com/Lightning-AI/lit-llama)
* [LoRA] (https://github.com/microsoft/LoRA)
