"""Train model using PPO algorithm (RL), starting from fine-tuned model and reward model (RM) checkpoints, and with LoRA parameter efficient method."""

import os
import itertools
import functools
from typing import Tuple, List
import tqdm
import random
import math
import numpy as np
from contextlib import nullcontext

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.distributions import Categorical

from torch.utils.tensorboard import SummaryWriter

# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

import torch.distributed as dist


# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from instruct_llama.model import Transformer, ModelArgs
from instruct_llama.tokenizer import Tokenizer
from instruct_llama.utils import PromptOnlyDataset
from instruct_llama.lora import lora, lora_state_dict, mark_only_lora_as_trainable

from instruct_llama.configs.train_ppo_policy_lora import config as cfg


from instruct_llama.utils import (
    CosineDecayWithWarmupLRScheduler,
    create_logger,
)


def setup():
    # initialize the process group
    dist.init_process_group('nccl')


def cleanup():
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    print(f'clearing cache for rank {rank}')
    torch.cuda.empty_cache()


def create_trace_profiler(tb_trace_dir):
    torch_profiler = torch.profiler.profile(
        activities=[
            # torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_trace_dir),
        profile_memory=True,
        with_stack=False,
        record_shapes=False,
    )

    return torch_profiler


def create_optimizer(
    model: torch.nn.Module, lr: float, eps: float, weight_decay: float, betas: Tuple[float], fused: bool
) -> torch.optim.AdamW:
    """
    Returns the PyTorch AdamW optimizer for the model,
    where we skip apply weight decay to layer norm, embedding, and all bias,
    and apply weight decay to the reset of parameters.
    """

    # filter out those do not require gradients
    params_dict = {p_name: params for p_name, params in model.named_parameters() if params.requires_grad}

    # Create empty lists to store parameters for weight decay and no weight decay.
    decay = []
    no_decay = []

    for p_name, params in params_dict.items():
        # Check for parameters corresponding to torch.nn.LayerNorm or torch.nn.Embedding.
        # Note we use hard-coded names where 'ln' is for LayerNorm, and 'embed' is for Embedding, this works better with FSDP
        if (
            p_name.endswith('bias')
            or p_name.endswith('attention_norm.weight')
            or p_name.endswith('ffn_norm.weight')
            or p_name.endswith('post_norm.weight')
            or p_name.endswith('token_embeddings.weight')
        ):
            no_decay.append(params)
        else:
            decay.append(params)

    num_decay_params = sum(p.numel() for p in decay)
    num_nodecay_params = sum(p.numel() for p in no_decay)
    total_num_params = sum(p.numel() for p in params_dict.values())
    assert num_decay_params + num_nodecay_params == total_num_params

    print(f'num decayed parameter tensors: {len(decay)}, with {num_decay_params:,} parameters')
    print(f'num non-decayed parameter tensors: {len(no_decay)}, with {num_nodecay_params:,} parameters')

    # create the pytorch optimizer object
    optim_groups = [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]

    if cfg.use_bnb_8bit:
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(optim_groups, lr=lr, eps=eps, betas=betas)
    else:
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, eps=eps, betas=betas, fused=fused)

    return optimizer


# adapted from
# https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/utils/core.py#L336e
def whiten(values, shift_mean=True):
    assert isinstance(values, np.ndarray)

    mean = np.mean(values, axis=tuple(range(values.ndim)))
    var = np.var(values, axis=tuple(range(values.ndim)))
    whitened = (values - mean) * np.sqrt(1.0 / (var + 1e-8))
    if not shift_mean:
        whitened += mean
    return whitened


def split_indices_into_bins(bin_size: int, max_indices: int, min_indices: int = 0, shuffle: bool = False) -> List[List[int]]:
    """Split indices to small bins."""

    bin_size = int(bin_size)
    max_indices = int(max_indices)
    min_indices = int(min_indices)

    if max_indices < bin_size:
        raise ValueError(f'Expect max_indices to be greater than bin_size, got {max_indices} and {bin_size}')

    # Split indices into 'bins' with bin_size.
    indices = np.arange(min_indices, max_indices)

    if shuffle:
        np.random.shuffle(indices)

    indices_list = []
    for i in range(0, len(indices), bin_size):
        indices_list.append(indices[i : i + bin_size])  # noqa: E203

    # Make sure the last one has the same 'bin_size'.
    if len(indices_list[-1]) != bin_size:
        indices_list[-1] = indices[-bin_size:]

    return indices_list


class PPOAgent:
    def __init__(
        self,
        model: Transformer,
        optimizer: torch.optim.AdamW,
        reward_model: Transformer,
        sft_model: Transformer,
        tokenizer: Tokenizer,
        train_dataset: PromptOnlyDataset,
        max_seq_len: int,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.reward_model = reward_model
        self.sft_model = sft_model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.max_seq_len = max_seq_len

        self.rm_device = 'cpu'
        self.sft_device = 'cpu'

        self.discount = 1.0
        self.gae_lambda = 0.95
        self.clip_eps = 0.2
        self.value_clip_eps = 0.2

        self.kl_coef = 0.1
        self.ptx_coef = 0.5
        self.value_coef = 0.25
        self.entropy_coef = 0.0

        self.max_grad_norm = 1.0

        self.normalize_reward = True
        self.normalize_advantage = True

        self.episodes = []

        self.num_epochs = 4
        self.batch_size = 8

        self.update_t = 0

    def compute_returns_and_advantages(self, v_t, r_t, v_tp1, done_tp1):
        if self.normalize_reward:
            r_t = whiten(r_t, shift_mean=False)

        discount_tp1 = (~done_tp1).astype(np.float32) * self.discount

        lambda_ = np.ones_like(discount_tp1) * self.gae_lambda  # If scalar, make into vector.

        delta_t = r_t + discount_tp1 * v_tp1 - v_t

        advantage_t = np.zeros_like(delta_t, dtype=np.float32)

        gae_t = 0
        for i in reversed(range(len(delta_t))):
            gae_t = delta_t[i] + discount_tp1[i] * lambda_[i] * gae_t
            advantage_t[i] = gae_t

        return_t = advantage_t + v_t

        if self.normalize_advantage:
            advantage_t = whiten(advantage_t)

        return return_t, advantage_t

    def do_learning(self):
        # always disable cache for learning
        self.model.disable_cache()
        self.model.train()
        torch.cuda.empty_cache()

        transitions = self.get_transitions_from_episodes()

        # Run M epochs to update network parameters
        for _ in range(self.num_epochs):
            # Split sequence into batches
            batch_indices = split_indices_into_bins(self.batch_size, len(transitions), shuffle=True)

            for indices in batch_indices:
                mini_batch = [transitions[i] for i in indices]
                self.update_policy(mini_batch)

        # Remove old transitions
        del self.episodes[:]

    def add_kl_penalty(self, rewards, logprobs, sft_logprobs):
        # here we use log trick, where log(A/B) = log(A) - log(B)
        kl = logprobs - sft_logprobs
        kl_penalties = -self.kl_coef * kl
        rewards += kl_penalties
        return rewards, kl_penalties

    def get_transitions_from_episodes(self):
        transitions = []
        for episode in self.episodes:
            s_t = episode['states']
            a_t = episode['actions']
            logprob_a_t = episode['logprobs']
            sft_logprob_a_t = episode['sft_logprobs']

            # compute returns and GAE advantages
            v_t = episode['values']
            r_t = episode['rewards']

            # pad 0 to make sure v_tp1 and v_t have same length, so can work with compute_returns_and_advantages
            v_tp1 = np.zeros_like(v_t)
            done_tp1 = np.ones_like(v_t).astype(bool)

            v_tp1[:-1] = episode['values'][1:]
            done_tp1[:-1] = episode['dones'][1:]

            # add KL penalty to the reward
            r_t, kl_penalties = self.add_kl_penalty(r_t, logprob_a_t, sft_logprob_a_t)

            return_t, advantage_t = self.compute_returns_and_advantages(v_t, r_t, v_tp1, done_tp1)

            transitions.append(
                {
                    'prompt_len': episode['prompt_len'],  # need this to compute pre-training loss and other stuff
                    's_t': s_t,
                    'a_t': a_t,
                    'logprob_a_t': logprob_a_t,
                    'return_t': return_t,
                    'advantage_t': advantage_t,
                }
            )

        return transitions

    def update_policy(self, mini_batch):
        self.optimizer.zero_grad()

        for sample in mini_batch:
            prompt_len = sample['prompt_len']

            s_t = torch.from_numpy(sample['s_t']).to(device='cuda', dtype=torch.long)
            a_t = torch.from_numpy(sample['a_t']).to(device='cuda', dtype=torch.long)  # Actions are discrete
            behavior_logprob_a_t = torch.from_numpy(sample['logprob_a_t']).to(device='cuda', dtype=torch.float32)
            return_t = torch.from_numpy(sample['return_t']).to(device='cuda', dtype=torch.float32)
            advantage_t = torch.from_numpy(sample['advantage_t']).to(device='cuda', dtype=torch.float32)

            # Given past states, get predicted action probabilities and state value
            outputs = self.model(s_t.unsqueeze(0))

            pi_logits_t, v_t = outputs['logits'].squeeze(0)[prompt_len - 1 :], outputs['values'].squeeze(0)[prompt_len - 1 :]

            pi_m = Categorical(logits=pi_logits_t)
            pi_logprob_a_t = pi_m.log_prob(a_t)
            entropy_loss = pi_m.entropy()

            assert len(pi_logprob_a_t) == len(behavior_logprob_a_t)
            assert len(v_t) == len(return_t)

            # Compute clipped surrogate objective
            ratio = torch.exp(pi_logprob_a_t - behavior_logprob_a_t.detach())
            clipped_ratio = torch.clamp(ratio, min=1.0 - self.clip_eps, max=1.0 + self.clip_eps)

            policy_loss = torch.min(ratio * advantage_t.detach(), clipped_ratio * advantage_t.detach())

            # Compute state value loss, where we use a non-standard method other than traditional PPO or Actor-Critic algorithm
            # here we also clip the value ranges similar to the policy objective
            # https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/train_policy.py#L343C38-L347
            v_t_clipped = torch.clamp(v_t, return_t - self.value_clip_eps, return_t + self.value_clip_eps)
            vloss_1 = torch.square(v_t - return_t)
            vloss_2 = torch.square(v_t_clipped - return_t)
            value_loss = 0.5 * torch.maximum(vloss_1, vloss_2)

            # Compute ptx loss, mixing pretraining gradients into PPO
            ptx_logits = outputs['logits'].squeeze(0)[1:prompt_len]
            ptx_target = s_t[1:prompt_len]
            ptx_loss = F.cross_entropy(ptx_logits.view(-1, ptx_logits.size(-1)), ptx_target.view(-1), reduction='mean')

            # Averaging over batch dimension
            policy_loss = torch.mean(policy_loss)
            entropy_loss = torch.mean(entropy_loss)
            value_loss = torch.mean(value_loss)
            ptx_loss = torch.mean(ptx_loss)

            print(f'policy_loss: {policy_loss.item()}')
            print(f'entropy_loss: {entropy_loss.item()}')
            print(f'value_loss: {value_loss.item()}')
            print(f'ptx_loss: {ptx_loss.item()}')

            # Negative sign to indicate we want to maximize the policy gradient objective function and entropy to encourage exploration
            loss = -(policy_loss + self.entropy_coef * entropy_loss) + self.value_coef * value_loss + self.ptx_coef * ptx_loss

            loss.backward()

        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.max_grad_norm,
                error_if_nonfinite=True,
            )

        self.optimizer.step()

    @torch.no_grad()
    def compute_terminal_step_reward(self, tokens: List[int]) -> float:
        tokens = torch.tensor(tokens, dtype=torch.long, device=self.rm_device)

        # [1, sequence_length, 1]
        outputs = self.reward_model(tokens.unsqueeze(0))

        # get reward for terminal time step
        reward = outputs.squeeze()[-1]

        return reward.cpu().item()

    @torch.no_grad()
    def run_single_episode(self):
        # always use cache to speed up acting
        self.model.enable_cache()

        # randomly sample a prompt to start a new episode
        prompt_tokens = self.train_dataset.sample()

        prompt_len = len(prompt_tokens)
        assert prompt_len <= self.max_seq_len

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((1, self.max_seq_len), pad_id, dtype=torch.long, device='cuda')
        tokens[:, :prompt_len] = torch.tensor(prompt_tokens, dtype=torch.long, device='cuda')

        # store needed transitions for PPO algorithm
        completion_tokens = []
        token_logprobs = []
        token_values = []
        rewards = []
        dones = []

        prev_pos = 0
        for cur_pos in range(prompt_len, self.max_seq_len):
            output = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            logits = output['logits'].squeeze(0)  # [seq_length, vocab_size]
            values = output['values'].squeeze(0)  # [seq_length, 1]

            # we only need the last one from the output sequence
            logits = logits[-1]  # [1, vocab_size]
            values = values[-1]  # [1, 1]

            pi_dist = Categorical(logits=logits)

            # # assume temperature = 0
            next_token = torch.argmax(logits, dim=-1)  # [1]

            _logprob = pi_dist.log_prob(next_token).cpu().item()
            _value = values.squeeze(-1).cpu().item()

            completion_tokens.append(next_token.cpu().item())
            token_logprobs.append(_logprob)
            token_values.append(_value)

            tokens[:, cur_pos] = next_token
            if next_token == self.tokenizer.eos_id:
                # compute the reward for the terminal state
                r_T = self.compute_terminal_step_reward(prompt_tokens + completion_tokens)
                rewards.append(r_T)
                dones.append(True)
                break
            else:
                # rewards are always zero for non-terminal state,
                # so we can use PPO to compute the returns and advantages
                rewards.append(0.0)
                dones.append(False)

        # we skip the final completion token, as we want the agent to choose which action during learning
        states = prompt_tokens + completion_tokens[:-1]

        # compute logprobs for completion tokens using the SFT model,
        # which is required to compute the KL penalties for pre-token reward score

        state_tokens = torch.tensor(states, dtype=torch.long).unsqueeze(0)

        sft_logits = self.sft_model.forward(state_tokens)  # [1, seq_len, vocab_size]

        # note the first completion token begins given the last prompt token, not one after it
        sft_logits = sft_logits.squeeze(0)[prompt_len - 1 :]  # [completion_len, vocab_size]

        sft_dist = Categorical(logits=sft_logits)

        sft_token_logprobs = sft_dist.log_prob(torch.tensor(completion_tokens, dtype=torch.long)).tolist()

        assert len(sft_token_logprobs) == len(token_logprobs)

        print('-' * 80)
        print(self.tokenizer.decode(prompt_tokens))
        print('\n -->')
        print(self.tokenizer.decode(completion_tokens))
        print('-' * 80)

        self.episodes.append(
            {
                'prompt_len': prompt_len,
                'states': np.array(states, dtype=np.int64),  # environment states
                'actions': np.array(completion_tokens, dtype=np.int64),  # agent actions
                'logprobs': np.array(token_logprobs, dtype=np.float32),  # logprobs of agent actions
                'values': np.array(token_values, dtype=np.float32),  # predicted state values
                'rewards': np.array(rewards, dtype=np.float32),  # environment reward
                'dones': np.array(dones, dtype=bool),  # terminal mark for environment states
                'sft_logprobs': np.array(sft_token_logprobs, dtype=np.float32),  # logprobs of agent actions under SFT model
            }
        )


def disable_grad(model: Transformer):
    for p in model.parameters():
        p.requires_grad = False


def main():
    assert cfg.gradient_accum_steps >= 1
    assert cfg.log_interval >= 10

    if not os.path.exists(cfg.sft_ckpt_file):
        raise ValueError(f'Invalid SFT model checkpoint "{cfg.sft_ckpt_file}", aborting...')

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    setup()

    logger = create_logger()

    # # --------------- Load datasets ---------------

    logger.info('Loading datasets...')

    tokenizer = Tokenizer(cfg.tokenizer_file)

    train_dataset = PromptOnlyDataset(
        data_sources=cfg.train_datasources,
        max_seq_len=int(cfg.max_seq_len * 0.6),
        seed=cfg.seed,
    )

    logger.info(f'Train dataset metadata:\n{train_dataset.get_metadata()}')

    # # create validation dataset
    # val_loader = None
    # if cfg.val_interval > 0:
    #     val_dataset = ComparisonsDataset(
    #         data_sources=cfg.val_datasources,
    #         min_completions=cfg.min_completions,
    #         max_completions=cfg.max_completions,
    #         max_seq_len=cfg.max_seq_len,
    #     )
    #     val_loader = DataLoader(val_dataset, **cuda_kwargs)
    #     logger.info(f'Validation dataset metadata:\n{val_dataset.get_metadata()}')

    # --------------- Setup model and optimizer ---------------

    logger.info('Initialize model and optimizer...')

    torch.cuda.set_device(local_rank)
    clear_gpu_cache(local_rank)

    logger.info('Initialize SFT model...')
    sft_args = ModelArgs.from_model_type(cfg.model_type)
    sft_args.vocab_size = tokenizer.vocab_size
    sft_args.max_seq_len = cfg.max_seq_len
    sft_args.max_batch_size = 1
    sft_args.use_cache = False
    sft_args.embed_dropout = 0.0
    sft_args.attn_dropout = 0.0
    sft_args.resid_dropout = 0.0
    sft_args.head_type = 'lm_head'

    sft_model = Transformer(sft_args)

    if os.path.exists(cfg.sft_ckpt_file):
        logger.info(f'Loading SFT model checkpoint {cfg.sft_ckpt_file}...')
        sft_state = torch.load(cfg.sft_ckpt_file)
        sft_model.load_state_dict(sft_state, strict=False)
        del sft_state  # free up CPU RAM

    # make reward model fixed as we don't need to train the reward model during RL
    disable_grad(sft_model)
    sft_model = sft_model.to(torch.bfloat16)
    sft_model.eval()

    logger.info('Initialize RM model...')

    rm_args = ModelArgs.from_model_type(cfg.model_type)
    rm_args.vocab_size = tokenizer.vocab_size
    rm_args.max_seq_len = cfg.max_seq_len
    rm_args.max_batch_size = 1
    rm_args.use_cache = False
    rm_args.embed_dropout = 0.0
    rm_args.attn_dropout = 0.0
    rm_args.resid_dropout = 0.0
    rm_args.head_type = 'scalar_head'

    reward_model = Transformer(rm_args)

    if os.path.exists(cfg.rm_ckpt_file):
        logger.info(f'Loading RM model checkpoint {cfg.rm_ckpt_file}...')
        rm_state = torch.load(cfg.rm_ckpt_file)
        reward_model.load_state_dict(rm_state, strict=False)
        del rm_state  # free up CPU RAM

    # make reward model fixed as we don't need to train the reward model during RL
    disable_grad(reward_model)
    reward_model = reward_model.to(torch.bfloat16)
    reward_model.eval()

    logger.info('Initialize policy model...')

    with lora(r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout, enabled=True):
        model_args = ModelArgs.from_model_type(cfg.model_type)
        model_args.vocab_size = tokenizer.vocab_size
        model_args.max_seq_len = cfg.max_seq_len
        model_args.max_batch_size = 1
        model_args.use_cache = True
        model_args.embed_dropout = cfg.embed_dropout
        model_args.attn_dropout = cfg.attn_dropout
        model_args.resid_dropout = cfg.resid_dropout
        model_args.head_type = cfg.head_type

        assert model_args.head_type == 'lm_and_scalar_heads'

        model = Transformer(model_args)

        # Load SFT model checkpoint using strict=False,
        # because there's not scalar head weights in the checkpoint state
        if os.path.exists(cfg.sft_ckpt_file):
            logger.info(f'Loading weights from SFT model checkpoint {cfg.sft_ckpt_file}...')
            sft_state = torch.load(cfg.sft_ckpt_file)
            model.load_state_dict(sft_state, strict=False)
            del sft_state
        # only need the scalar_head from RM model checkpoint
        if os.path.exists(cfg.rm_ckpt_file):
            logger.info(f'Loading scalar head weights from RM model checkpoint {cfg.rm_ckpt_file}...')
            rm_state = torch.load(cfg.rm_ckpt_file)
            rm_partial_state = {k: rm_state[k] for k in rm_state.keys() if 'scalar_head' in k}
            model.load_state_dict(rm_partial_state, strict=False)
            del rm_state, rm_partial_state
        else:
            model.init_scalar_head_weights()

    mark_only_lora_as_trainable(model, train_bias=cfg.train_bias, train_head=cfg.train_head)

    # try to convert the model to half precision, otherwise we can't even move the 7B model to a single RTX 3090
    bf16_ready = torch.version.cuda and torch.cuda.is_bf16_supported()
    train_dtype = torch.float32

    scaler = None

    if cfg.mixed_precision:
        if bf16_ready:
            train_dtype = torch.bfloat16
        else:
            train_dtype = torch.float16
            scaler = torch.cuda.amp.GradScaler()
    else:
        logger.warning('Training in float32 mode, make sure you have enough GPU RAM')

    # BUG in pytorch 2.0.1, as we found out using torch.autocast will increase GPU RAM usage,
    # and cause CUDA OUT OF MEMORY error when run the training script on a single RTX 3090
    # so we manually convert the model to half precision before moving it to GPU

    # mp_ctx = torch.cuda.amp.autocast(dtype=train_dtype, cache_enabled=False)

    for name, module in model.named_modules():
        if 'norm' in name:  # for better performance, always use full precision for normalization layers
            module = module.to(dtype=torch.float32)
        else:
            module = module.to(dtype=train_dtype)

    model = model.to(local_rank)

    if cfg.compile_model:
        logger.info('compile model using torch.compile()...')
        model = torch.compile(model)

    logger.info('Initialize optimizer...')

    optimizer = create_optimizer(
        model=model,
        lr=cfg.lr,
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
        betas=cfg.adam_betas,
        fused=cfg.adam_fused,
    )

    agent = PPOAgent(
        model=model,
        optimizer=optimizer,
        reward_model=reward_model,
        sft_model=sft_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        max_seq_len=cfg.max_seq_len,
    )

    for _ in range(10):
        agent.run_single_episode()

    agent.do_learning()

    # # --------------- Start Training ---------------

    # logger.info(f'Starting to run {cfg.max_train_iters} training iterations...')

    # torch_profiler = None
    # # Careful as the logs will grow very fast
    # if cfg.use_profiler:
    #     torch_profiler = create_trace_profiler(os.path.join(cfg.log_dir, 'profile_traces'))

    # tb_writer = None
    # inner_pbar = None
    # train_stats = None
    # val_stats = None

    # if rank == 0:
    #     os.makedirs(cfg.log_dir, exist_ok=True)
    #     os.makedirs(cfg.ckpt_dir, exist_ok=True)

    #     if cfg.use_tensorboard:
    #         tb_writer = SummaryWriter(os.path.join(cfg.log_dir, cfg.model_type))

    #     inner_pbar = tqdm.tqdm(range(cfg.max_train_iters), colour='blue', desc='Training iterations')

    # model.train()
    # for iter in range(1, cfg.max_train_iters + 1):
    #     train_stats = run_single_train_step(
    #         model=model,
    #         rank=rank,
    #         world_size=world_size,
    #         train_loader=train_loader,
    #         optimizer=optimizer,
    #         scheduler=scheduler,
    #         scaler=scaler,
    #         return_stats=iter % cfg.log_interval == 0 or iter == cfg.max_train_iters,
    #     )

    #     if inner_pbar is not None:
    #         inner_pbar.update(1)

    #     if torch_profiler is not None:
    #         torch_profiler.step()

    # # logging
    # if train_stats is not None and rank == 0:
    #     logger.info(
    #         f'Training iteration {iter}: train loss: {train_stats["loss"]:.4f}, train accuracy: {train_stats["accuracy"]:.2f}%, learning rate: {train_stats["learning_rate"]:.10f}'
    #     )

    #     if tb_writer is not None:
    #         tb_writer.add_scalar('train/loss', train_stats['loss'], iter)
    #         tb_writer.add_scalar('train/accuracy', train_stats['accuracy'], iter)
    #         tb_writer.add_scalar('train/learning_rate', train_stats['learning_rate'], iter)

    # # checkpointing
    # if cfg.ckpt_interval > 0 and iter % cfg.ckpt_interval == 0 or iter == cfg.max_train_iters:
    #     # save model state
    #     checkpoint = lora_state_dict(model, train_bias=cfg.train_bias, train_head=cfg.train_head)

    #     torch.save(checkpoint, os.path.join(cfg.ckpt_dir, f'lora_{cfg.model_type}-iter-{iter}.pth'))

    # # validation steps
    # if cfg.val_iters > 0 and (cfg.val_interval > 0 and iter % cfg.val_interval == 0 or iter == cfg.max_train_iters):
    #     val_stats = run_validation_steps(model=model, rank=rank, world_size=world_size, val_loader=val_loader)

    #     if rank == 0:
    #         logger.info(
    #             f'Training iteration {iter}: validation loss: {val_stats["loss"]:.4f}, validation accuracy: {val_stats["accuracy"]:.2f}%'
    #         )

    #         if tb_writer is not None:
    #             tb_writer.add_scalar('val/loss', val_stats['loss'], iter)
    #             tb_writer.add_scalar('val/accuracy', val_stats['accuracy'], iter)

    if rank == 0:
        # training is done...show some training stats.
        logger.info(f'CUDA Memory Summary After Last training:\n{torch.cuda.memory_summary()}')

    # all done, set barrier to ensure all GPU's complete, and then cleanup
    dist.barrier()
    cleanup()


if __name__ == '__main__':
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    torch.set_float32_matmul_precision('high')

    main()
