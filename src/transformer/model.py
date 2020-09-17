import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from utils import weight_init


class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, max_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, input_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            -torch.arange(0, input_dim, 2, dtype=torch.float)
            * torch.log(torch.tensor(10000.0))
            / input_dim
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe: (max_len, 1, input_dim)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """ x: (max_len, bs, input_dim) """
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


def make_pad_mask(max_seqlen, lengths):
    mask = torch.ones(
        lengths.size(0), max_seqlen, dtype=torch.bool, device=lengths.device
    )
    for i, l in enumerate(lengths):
        mask[i, : l.item()] = False
    return mask


class TransformerModel(nn.Module):
    """
    Transformer Encoder + Decoder
    Args:
        vocab_size:     size of vocabulary
        feature_dim:    dimension of feature
        max_srclen:     max input sequence length
        max_tgtlen:     max output sequence length
        bos_idx:        <bos> token index, for inference
        pad_idx:        <pad> token index, for inference
        n_layers:       number of encoder/decoder layers
        n_head:         number of heads in multi head attention
        input_dim:      input dimension
        dropout_p:      dropout probability
        ss_prob:        scheduled sampling rate, 0 for teacher forcing and 1 \
                        for free running
    """

    def __init__(self, CONFIG, vocab_size, bos_idx, pad_idx):
        super().__init__()
        feature_dim = CONFIG.hyperparam.transformer.feature_dim
        input_dim = CONFIG.hyperparam.transformer.input_dim
        self.input_length = CONFIG.hyperparam.transformer.input_seqlen
        self.output_length = CONFIG.hyperparam.tokenizer.max_len
        n_head = CONFIG.hyperparam.transformer.n_head
        n_layer = CONFIG.hyperparam.transformer.n_layer
        dropout_p = CONFIG.hyperparam.transformer.dropout_prob
        self.vocab_size = vocab_size

        self.encoder = nn.Linear(feature_dim, input_dim)
        self.pos_enc = PositionalEncoding(input_dim, self.input_length)
        self.pos_dec = PositionalEncoding(input_dim, self.output_length)
        self.transformer = nn.Transformer(
            input_dim, n_head, n_layer, n_layer, input_dim * 4, dropout_p
        )
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.decoder = nn.Linear(input_dim, vocab_size)
        # use subsequent mask for only target
        self.tgt_mask = self.transformer.generate_square_subsequent_mask(
            self.output_length
        )
        self.xeloss = nn.CrossEntropyLoss(ignore_index=pad_idx)
        self.bos_idx = bos_idx
        self.pad_idx = pad_idx
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def forward(self, src, src_len, tgt):
        """
        Args:
            torch.Tensor src:           (bs x max_srclen x feature_dim), torch.float
            torch.Tensor src_len:       (bs), torch.long
            torch.Tensor tgt:           (bs x max_tgtlen), torch.long
        Returns:
            losses:                     dict( loss_name: str, loss_val: torch.Tensor )
        """
        # src_input: (max_srclen x bs x input_dim)
        src_input = self.pos_enc(self.encoder(src)).transpose(0, 1)
        src_pad_mask = make_pad_mask(self.input_length, src_len)
        # tgt_input: (max_tgtlen x bs x input_dim)
        tgt_input = self.pos_dec(self.embedding(tgt)).transpose(0, 1)
        tgt_len = tgt.ne(self.pad_idx).sum(dim=1)
        tgt_pad_mask = make_pad_mask(self.output_length, tgt_len)
        # out: (bs x input_dim x input_length)
        out = self.transformer(
            src_input,
            tgt_input,
            src_mask=None,
            tgt_mask=self.tgt_mask.to(self.device),
            memory_mask=None,
            src_key_padding_mask=src_pad_mask.to(self.device),
            tgt_key_padding_mask=tgt_pad_mask.to(self.device),
        ).transpose(0, 1)
        # out: (bs x vocab_size x output_length)
        out = self.decoder(out).transpose(1, 2)
        losses = {}
        losses["XELoss"] = self.xeloss(out[:, :, :-1], tgt[:, 1:]).unsqueeze(0)
        return losses

    def sample(self, src, src_len):
        """
        Args:
            torch.Tensor src:           (bs x max_srclen x feature_dim), torch.float
            torch.Tensor src_len:       (bs), torch.long
        Returns:
            logprobs:                   (bs x vocab_size x output_length - 1), torch.float
            tgt:                        (bs x output_length - 1), torch.long
        """
        bs = src.size(0)
        # src_input: (input_length x bs x input_dim)
        src_input = self.pos_enc(self.encoder(src)).transpose(0, 1)
        src_pad_mask = make_pad_mask(self.input_length, src_len)
        # tgt: (bs x output_length)
        tgt = torch.full((bs, self.output_length), self.bos_idx, dtype=torch.long).to(
            self.device
        )
        # logprobs: (bs x vocab_size x output_length - 1)
        logprobs = torch.empty(bs, self.vocab_size, self.output_length - 1)
        for step in range(self.output_length - 1):
            # tgt_input: (output_length x bs x input_dim)
            tgt_input = self.pos_dec(self.embedding(tgt)).transpose(0, 1)
            tgt_length = torch.full((bs,), step + 1, dtype=torch.long)
            tgt_pad_mask = make_pad_mask(self.output_length, tgt_length)
            # out: (bs x output_length x input_dim)
            out = self.transformer(
                src_input,
                tgt_input,
                src_mask=None,
                tgt_mask=self.tgt_mask.to(self.device),
                memory_mask=None,
                src_key_padding_mask=src_pad_mask.to(self.device),
                tgt_key_padding_mask=tgt_pad_mask.to(self.device),
            ).transpose(0, 1)
            # out: (bs x vocab_size x output_length)
            out = self.decoder(out).transpose(1, 2)
            logprobs[:, :, step] = F.log_softmax(out[:, :, step], dim=1)
            # greedy
            tgt[:, step + 1] = torch.argmax(logprobs[:, :, step], dim=1)
            # sampled
            # tgt[:, step + 1] = torch.multinomial(
            #     F.softmax(out[:, :, step], dim=1), 1, replacement=True)[:, 0]
        return logprobs, tgt[:, 1:]


if __name__ == "__main__":
    bs = 32
    maxlen = 100
    vocab = 200
    model = TransformerModel(vocab, maxlen, maxlen, 10, 1)
    src = torch.randn((bs, maxlen, 768))
    src_len = torch.randint(1, maxlen + 1, (bs,))
    tgt = torch.randint(vocab, (bs, maxlen))
    tgt_len = torch.randint(1, maxlen + 1, (bs,))
    out = model(src, src_len, tgt, tgt_len)
    print(out["XELoss"].item())
