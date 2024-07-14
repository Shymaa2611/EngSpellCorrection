from typing import Tuple, Union
from args import get_model_args
from torch import nn
from layers import (
    RNNDecoder,
    RNNEncoder
    )
from torch import Tensor
import torch

class Seq2SeqRNN(nn.Module):
    def __init__(
            self,
            voc_size: int,
            emb_size: int,
            n_layers: int,
            hidden_size: int,
            bidirectional: bool,
            padding_idx: int,
            padding_value: int,
            p_dropout: float,
            max_len: int
            ) -> None:
        super().__init__()
        self.encoder = RNNEncoder(
            voc_size=voc_size,
            emb_size=emb_size,
            n_layers=n_layers,
            hidden_size=hidden_size,
            p_dropout=p_dropout,
            bidirectional=bidirectional,
            padding_idx=padding_idx,
            padding_value=padding_value
        )
        self.decoder = RNNDecoder(
            max_len=max_len,
            voc_size=voc_size,
            emb_size=emb_size,
            n_layers=n_layers,
            hidden_size=hidden_size,
            p_dropout=p_dropout,
            bidirectional=bidirectional,
            padding_idx=padding_idx,
            padding_value=padding_value
        )

    def get_lengths(self, mask: Tensor) -> Tensor:
        return (~mask).sum(dim=-1)

    def forward(
            self,
            enc_inp: Tensor,
            dec_inp: Tensor,
            enc_mask: Tensor,
            dec_mask: Tensor
            ) -> Tuple[Tensor, Tensor]:
        enc_lengths = self.get_lengths(enc_mask).cpu()
        dec_lengths = self.get_lengths(dec_mask).cpu()
        enc_values, h = self.encoder(enc_inp, enc_lengths)
        result, attention = self.decoder(
            enc_values=enc_values,
            hn=h,
            x=dec_inp,
            lengths=dec_lengths
            )
        return nn.functional.log_softmax(result, dim=-1), attention

    def predict(
            self,
            dec_inp: Tensor,
            enc_inp: Tensor,
            enc_mask,
            h: Tensor,
            key=None,
            value=None
            ):
        enc_lengths = self.get_lengths(enc_mask).cpu()
        if key is None and value is None:
            enc_values, h = self.encoder(enc_inp, enc_lengths)
        else:
            enc_values = None
        h, att, result, key, value = self.decoder.predict(
            hn=h,
            x=dec_inp,
            enc_values=enc_values,
            key=key,
            value=value
        )
        return h, att, result, key, value

def get_model(
        args,
        rank: int,
        voc_size: int,
        pad_idx: int
        ) -> nn.Module:
        return Seq2SeqRNN(
            **get_model_args(args, voc_size, rank, pad_idx)
        )
