import os
from pathlib import Path
from typing import List, Mapping, Text, Any
from logging import getLogger
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer

logger = getLogger()


class Tokenizer:
    """Tokenizer for LLaMA."""

    def __init__(self, model_path: Path) -> None:
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f'Reloaded SentencePiece model from {model_path}')

        # BOS / EOS token IDs
        self.vocab_size: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        self.unk_id: int = self.sp_model.unk_id()

        logger.info(f'#words: {self.vocab_size}, BOS ID: {self.bos_id}, EOS ID: {self.eos_id}, PAD ID: {self.pad_id}')
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    @property
    def special_tokens(self) -> Mapping[Text, Any]:
        return {
            'unk': {
                'id': self.unk_id,
                'piece': self.sp_model.IdToPiece(self.unk_id),
            },
            'bos': {
                'id': self.bos_id,
                'piece': self.sp_model.IdToPiece(self.bos_id),
            },
            'eos': {
                'id': self.eos_id,
                'piece': self.sp_model.IdToPiece(self.eos_id),
            },
        }

    def encode(
        self,
        string: str,
        bos: bool = False,
        eos: bool = False,
        max_length: int = -1,
        pad: bool = False,
    ) -> List[int]:
        tokens = self.sp_model.encode(string)
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        if max_length > 0:
            tokens = tokens[:max_length]
        if pad and len(tokens) < max_length:
            tokens += [self.pad_id] * (max_length - len(tokens))

        return tokens

    def decode(self, tokens: List[int]) -> str:
        return self.sp_model.decode(tokens)

    @staticmethod
    def train(input: str, destination: str, vocab_size=32000) -> None:
        model_prefix = os.path.join(destination, 'tokenizer')
        SentencePieceTrainer.Train(input=input, model_prefix=model_prefix, vocab_size=vocab_size)
