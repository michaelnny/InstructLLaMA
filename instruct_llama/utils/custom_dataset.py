from typing import Iterable
import os
import random
import math
import itertools
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader


class DataSource:
    """A simple class to load data source from numpy.memmap structure."""

    def __init__(self, name, weights, data_file, metadata_file):
        self.name = name
        self.weights = weights
        self.data_file = data_file
        self.metadata_file = metadata_file

        self.num_tokens = 0
        self.data_type = None
        self.data = None
        self.metadata = None

        self._sanity_check()

    def _sanity_check(self):
        assert len(self.name) > 0
        assert 0 <= self.weights <= 1
        assert os.path.exists(self.data_file) and self.data_file.endswith(".npy")
        assert os.path.exists(self.metadata_file) and self.metadata_file.endswith(".pkl")

    def update_weights(self, v) -> None:
        assert 0 <= v <= 1
        self.weights = v

    def load_metadata(self) -> None:
        metadata = pickle.load(open(self.metadata_file, "rb"))
        assert "num_tokens" in metadata and "data_type" in metadata
        assert metadata["data_type"] in ["uint16", "unit32"]

        self.metadata = metadata
        self.num_tokens = max(0, int(metadata["num_tokens"]))
        self.data_type = np.uint16 if metadata["data_type"] == "uint16" else np.uint32

        self.weights = self.weights if self.num_tokens > 0 else 0.0

    def load_data(self, fully_load: bool = False) -> None:
        if self.weights > 0 and self.num_tokens > 0:
            arr_memmap = np.memmap(self.data_file, dtype=self.data_type, mode="r", shape=(self.num_tokens,))

            # load the entire dataset into memory
            if fully_load:
                self.data = np.array(arr_memmap)
                del arr_memmap
            else:
                self.data = arr_memmap

    def extract_metadata(self):
        return {
            "name": self.name,
            "weights": self.weights,
            "num_tokens": self.num_tokens,
            "vocab_size": self.metadata["vocab_size"],
            "tokenizer": self.metadata["tokenizer"],
            "data_type": self.metadata["data_type"],
            "data_file": self.data_file,
            "metadata_file": self.metadata_file,
        }


class BlendedDataset(IterableDataset):
    """A blended dataset used for pre-training.

    It supports mixed data sources, where we can define the weights of each data source.
    Additionally, it also supports shard the dataset based on the world size.
    """

    def __init__(
        self,
        data_sources: Iterable[DataSource],
        max_seq_len: int,
        rank=0,
        world_size=0,
        seed: int = 1,
        fully_load: bool = False,
    ) -> None:
        """
        Args:
            data_sources: a list of DataSource objects to specify where to load the data, and how often we should use it
            max_seq_len: the context window (or block size) of the GTP model.
            rank: rank of the process to shard data, default 0.
            world_size: how many partitions to use when shard data, default 0 no shard.
            seed: random seed, default 1.
            fully_load: load the entire dataset into memory, default off.
        """

        assert len(data_sources) > 0
        assert max_seq_len is not None and max_seq_len > 0
        assert rank is not None and rank >= 0
        assert world_size is not None and world_size >= 0

        random.seed(seed)

        self.data_sources = data_sources

        self.rank = rank
        self.world_size = world_size

        self.max_seq_len = max_seq_len
        self.fully_load = fully_load
        self.total_num_tokens = 0

        # Load data sources
        for source in self.data_sources:
            source.load_metadata()
            source.load_data(self.fully_load)
            self.total_num_tokens += source.num_tokens

        assert self.total_num_tokens > 0

        # Derive and normalize data source sampling probabilities
        sample_weights = np.array([source.weights for source in self.data_sources], dtype=np.float16)
        assert 0 < np.sum(sample_weights)

        self.sample_weights = (sample_weights / np.sum(sample_weights)).tolist()

        for source, p in zip(self.data_sources, self.sample_weights):
            source.update_weights(p)

        # pre-compute shard start and end indices for each data source
        self.shard_indices = []

        for source in self.data_sources:
            num_tokens = source.num_tokens
            start_idx = 0
            end_idx = num_tokens - 1

            if self.world_size > 1:
                shard_size = int(math.ceil(num_tokens / float(self.world_size)))

                start_idx = shard_size * self.rank
                end_idx = start_idx + shard_size

                if end_idx > num_tokens - 1:
                    end_idx = num_tokens - 1

                assert start_idx >= 0 and end_idx - start_idx > self.max_seq_len

            self.shard_indices.append((start_idx, end_idx))

    def generator(self):
        while True:
            ds_idx = self._choose_datasource()
            source = self.data_sources[ds_idx]

            num_tokens = source.num_tokens
            data = source.data

            assert data is not None and num_tokens > self.max_seq_len

            # Get shard start and end indices for the chosen data source
            min_idx, max_idx = self.shard_indices[ds_idx]

            start = random.randint(a=min_idx, b=max_idx - self.max_seq_len - 1)

            end = start + self.max_seq_len

            assert end <= num_tokens - 1

            # here the high is exclusive
            x = torch.from_numpy(data[start:end]).to(dtype=torch.long)
            y = torch.from_numpy(data[start + 1 : end + 1]).to(dtype=torch.long)

            yield x, y

    def __iter__(self):
        return iter(self.generator())

    def _choose_datasource(self) -> int:
        return random.choices(range(len(self.data_sources)), weights=self.sample_weights, k=1)[0]

    def get_metadata(self):
        return {
            "dataset_type": "blended",
            "num_tokens": self.total_num_tokens,
            "fully_loaded": self.fully_load,
            "data_sources": [source.extract_metadata() for source in self.data_sources],
        }


class FineTuneDataset(Dataset):
    def __init__(self, data_sources: Iterable[str], max_seq_len: int = 2048) -> None:
        """
        Args:
            data_sources: a list of string path to where to load the dataset.
            max_seq_len: prompt_tokens + completion_tokens length greater than this will be discarded.
        """

        assert len(data_sources) > 0
        assert max_seq_len > 128

        self.data_sources = data_sources
        self.max_seq_len = max_seq_len

        self.data = []

        seq_lengths = []  # track statistics

        # Load datasets
        for source in data_sources:
            samples = pickle.load(open(source, "rb"))
            for sample in samples:
                x, y = sample["prompt_tokens"], sample["completion_tokens"]
                seq_length = len(x) + len(y)
                if seq_length <= self.max_seq_len:
                    self.data.append((x, y))
                    seq_lengths.append(seq_length)

        self.total_num_tokens = sum(seq_lengths)
        self.seq_length_stats = {
            "min": int(np.min(seq_lengths)),
            "max": int(np.max(seq_lengths)),
            "mean": int(np.mean(seq_lengths)),
            "std": int(np.std(seq_lengths)),
        }

        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    def get_metadata(self):
        return {
            "num_samples": len(self),
            "num_tokens": self.total_num_tokens,
            "sequence_length_stats": self.seq_length_stats,
            "data_sources": self.data_sources,
        }


class ComparisonsDataset(Dataset):
    def __init__(self, data_sources: Iterable[str], max_seq_len: int = 2048) -> None:
        """
        Args:
            data_sources: a list of string path to where to load the dataset.
            max_seq_len: prompt_tokens + completion_tokens length greater than this will be discarded.
        """
        assert len(data_sources) > 0
        assert max_seq_len > 128

        self.data_sources = data_sources
        self.max_seq_len = max_seq_len

        self.data = []

        # Load datasets
        for source in data_sources:
            samples = pickle.load(open(source, "rb"))
            for sample in samples:
                # here completions is a (descending) ordered list of completion tokens, with the best answer at the begining (index 0)
                x, ys = sample["prompt_tokens"], sample["completions_tokens"]

                # exclude those samples with length greater than max sequence length
                ys = [y for y in ys if len(x) + len(y) <= self.max_seq_len]

                if len(ys) < 2:  # comparison requires at least 2 samples
                    continue

                self.data.append((x, ys))

        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, ys = self.data[idx]

        return x, ys

    def get_metadata(self):
        return {
            "num_samples": len(self),
            "data_sources": self.data_sources,
        }


if __name__ == "__main__":
    # pretrain_ds = BlendedDataset(
    #     data_sources=[
    #         DataSource(
    #             name="books1",
    #             weights=0.7,
    #             data_file="./datasets/books1/test.npy",
    #             metadata_file="./datasets/books1/test_meta.pkl",
    #         ),
    #         DataSource(
    #             name="books2",
    #             weights=0.1,
    #             data_file="./datasets/books1/val.npy",
    #             metadata_file="./datasets/books1/val_meta.pkl",
    #         ),
    #         DataSource(
    #             name="books3",
    #             weights=0.2,
    #             data_file="./datasets/books1/train.npy",
    #             metadata_file="./datasets/books1/train_meta.pkl",
    #         ),
    #     ],
    #     max_seq_len=512,
    # )

    # meta = pretrain_ds.get_metadata()
    # print(meta)

    # batch_size = 32
    # train_dl = DataLoader(pretrain_ds, batch_size=batch_size, num_workers=2)

    # iterations = 10
    # grad_accum_steps = 10

    # # Iterate over the DataLoader with limited iterations
    # for iter in range(iterations):
    #     for batch_sequence, batch_next_token in itertools.islice(
    #         train_dl, grad_accum_steps
    #     ):
    #         assert batch_sequence.shape == batch_next_token.shape
    #         for i in range(1, len(batch_sequence)):
    #             assert not torch.equal(batch_sequence[0], batch_sequence[i])

    finetune_ds = FineTuneDataset(
        data_sources=[
            "./datasets/alpaca/train.pkl",
            "./datasets/dolly/train.pkl",
        ],
    )

    # print(len(finetune_ds))
    print(finetune_ds.get_metadata())

    dl = DataLoader(finetune_ds, batch_size=1, shuffle=True)

    pause_after_iterations = 10

    # Iterate over the DataLoader with limited iterations
    for x, y in itertools.islice(dl, pause_after_iterations):
        print(f"x: {x.shape}")
        print(f"y: {y.shape}")
