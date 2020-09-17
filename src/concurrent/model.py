import math

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from addict import Dict
from torch import nn

from utils import weight_init


class Attention(nn.Module):
    def __init__(self, ft_dim, rnn_dim, attn_dim):
        super().__init__()
        self.enc_attn = nn.Linear(ft_dim, attn_dim)
        self.dec_attn = nn.Linear(rnn_dim, attn_dim)
        self.attn = nn.Linear(attn_dim, ft_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feature, memory):
        """
        Args:
            torch.Tensor feature:               (B x ft_dim)
            torch.Tensor memory:                (B x rnn_dim)
        Returns:
            torch.Tensor attn_weights:          (B x ft_dim)
            torch.Tensor weighted_feature:      (B x ft_dim)
        """
        # encoded_feature: (B x attn_dim)
        encoded_feature = self.enc_attn(feature)
        # encoded_memory: (B x attn_dim)
        encoded_memory = self.dec_attn(memory)
        # attn_weights: (B x ft_dim)
        attn_weights = self.attn(self.relu(encoded_feature + encoded_memory))
        attn_weights = self.softmax(attn_weights)
        weighted_feature = feature * attn_weights
        return attn_weights, weighted_feature


class EventEncoderCell(nn.Module):
    """
    Event Encoder, Works as a cell
    """

    def __init__(self, CONFIG):
        super().__init__()
        self.CONFIG = CONFIG
        self.init_s = nn.Linear(CONFIG.dim.D, CONFIG.dim.M_2)
        self.init_h = nn.Linear(CONFIG.dim.D, CONFIG.dim.M_1)
        self.init_c = nn.Linear(CONFIG.dim.D, CONFIG.dim.M_1)
        self.linear1 = nn.Linear(CONFIG.dim.D + CONFIG.dim.M_2, CONFIG.dim.M_1)
        self.rnn = nn.LSTMCell(CONFIG.dim.M_1, CONFIG.dim.M_1)
        self.linear2 = nn.Linear(CONFIG.dim.M_1, CONFIG.dim.E)

    def forward(self, f, s=None, h=None, c=None):
        """
        Args:
            torch.Tensor f:             (B x D)
            torch.Tensor s:             (B x M_2), optional
            torch.Tensor h, c:          (B x M_1), optional
        Returns:
            torch.Tensor h, c:          (B x M_1), internal states
            torch.Tensor o:             (B x E), input to decoder
        """
        if s is None:
            s = self.init_s(f)
            h = self.init_h(f)
            c = self.init_c(f)
        # f: (B x D) + s: (B x M_2) -> trans_ft: (B x M_1)
        trans_ft = self.linear1(torch.cat([f, s], dim=1))
        (h, c) = self.rnn(trans_ft, (h, c))
        # h: (B x M_1) -> o: (B x E)
        o = self.linear2(h)
        return h, c, o


class CaptionDecoderCell(nn.Module):
    """
    RNN decoder with attention for captioning, Show, Attend and Tell
    Args:
        V:        vocabulary size
    """

    def __init__(self, CONFIG, V, bos_idx, eos_idx=10):
        super().__init__()
        self.V = V
        self.D = CONFIG.dim.D
        self.E = CONFIG.dim.E
        self.M_2 = CONFIG.dim.M_2
        self.max_caption_len = CONFIG.dim.max_caption_len
        self.dropout_p = CONFIG.hyp.dropout
        self.ss_prob = CONFIG.hyp.ss_prob
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        self.init_h = nn.Linear(self.E, self.M_2)
        self.init_c = nn.Linear(self.E, self.M_2)
        self.emb = nn.Embedding(V, self.E)
        self.rnn = nn.LSTMCell(2 * self.E, self.M_2)
        self.linear = nn.Linear(self.M_2, V)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, o, t=None, length=None):
        """
        Args:
            torch.Tensor o:             (B x E), torch.float
            torch.Tensor t:             (B x max_caption_len), torch.long
            torch.Tensor length:        (B), torch.long, None for inference
        Returns:
            torch.Tensor logits:        (B x V x max_caption_len), \
                                        contains logits over words
            torch.Tensor s:             (B x M_2), context vector
        """
        B = o.size(0)
        if t is None:
            t = torch.full(
                (B, self.max_caption_len),
                self.bos_idx,
                dtype=torch.long,
                device=o.device,
            )
            self.ss_prob = 1.0
        # hn, cn: (B x M_2)
        hn = self.init_h(o)
        cn = self.init_c(o)
        # cap_emb: (B x max_caption_len x E)
        cap_emb = self.emb(t)
        # o: (B x E)
        tn = o
        # logits: (B x V x max_caption_len)
        logits = torch.empty((B, self.V, self.max_caption_len), device=o.device)
        # hidden: (B x M_2 x max_caption_len)
        hidden = torch.empty((B, self.M_2, self.max_caption_len), device=o.device)
        for step in range(self.max_caption_len):
            # hn, cn: (B x M_2)
            hn, cn = self.rnn(torch.cat([tn, o], dim=1), (hn, cn))
            hidden[:, :, step] = hn
            # on: (B x V)
            on = self.linear(self.dropout(hn))
            logits[:, :, step] = on
            # xn: (B x emb_dim)
            tn = (
                self.emb(on.argmax(dim=1))
                if np.random.uniform() < self.ss_prob
                else cap_emb[:, step, :]
            )
        if length is None:
            mat = torch.eq(logits.argmax(1), self.eos_idx)
            # find first occurence of <eos> token
            length = first_true(mat, axis=1)
        else:
            length = torch.clamp(length - 1, min=0)
        # length: (B, M_2, 1)
        length = length.view(-1, 1, 1).expand(-1, self.M_2, 1)
        # hidden: (B x M_2)
        hidden = torch.gather(hidden, 2, length).squeeze(-1)
        return logits, hidden


def first_true(x, axis):
    """
    returns first true element in matrix along axis.
    """
    nonz = x > 0
    return ((nonz.cumsum(axis) == 1) & nonz).max(axis)[1]


class Captioning(nn.Module):
    def __init__(self, CONFIG, V, bos_idx, eos_idx):
        super().__init__()
        self.V = V
        self.max_actions = CONFIG.dim.max_actions
        self.max_caption_len = CONFIG.dim.max_caption_len

        self.enc = EventEncoderCell(CONFIG)
        self.dec = CaptionDecoderCell(CONFIG, V, bos_idx, eos_idx)

    def forward(self, feature, action_len, caption=None, caption_len=None):
        """
        Args:
            torch.Tensor feature:       (B x max_actions x D), torch.float
            torch.Tensor action_len:    (B), torch.long
            torch.Tensor caption:       (B x max_actions x max_caption_len), \
                                        torch.long
            torch.Tensor caption_len:   (B x max_actions), torch.long
        Returns:
            torch.Tensor logits:        (B x V x max_actions x max_caption_len), \
                                        output logits
        """
        B = feature.size(0)
        s = h = c = None
        # logits: (B x V x max_actions x max_caption_len)
        logits = torch.zeros(
            (B, self.V, self.max_actions, self.max_caption_len), device=feature.device,
        )
        for i in range(self.max_actions):
            t = None if caption is None else caption[:, i, :]
            l = None if caption_len is None else caption_len[:, i]
            # h, c: (B x M_1)
            h, c, o = self.enc(feature[:, i, :], s, h, c)
            # s: (B x M_2)
            # logit: (B x V x max_caption_len)
            logits[:, :, i, :], s = self.dec(o, t, l)
        # captions = logits.argmax(1)
        return logits


if __name__ == "__main__":
    cfg_path = "../../cfg/debug_aist.yml"
    CONFIG = Dict(yaml.safe_load(open(cfg_path)))
    V = 10000

    enc = EventEncoderCell(CONFIG)
    dec = CaptionDecoderCell(CONFIG, V)
    full_model = Captioning(CONFIG, V)

    f = torch.randn((CONFIG.dim.B, CONFIG.dim.D))
    t = torch.randint(V, (CONFIG.dim.B, CONFIG.dim.max_caption_len))
    l = torch.randint(CONFIG.dim.max_caption_len, (CONFIG.dim.B,))
    s = h = c = None
    for _ in range(CONFIG.dim.max_actions):
        h, c, o = enc(f, s, h, c)
        logprob, s = dec(o, t)

    f = torch.randn((CONFIG.dim.B, CONFIG.dim.max_actions, CONFIG.dim.D))
    t = torch.randint(
        V, (CONFIG.dim.B, CONFIG.dim.max_actions, CONFIG.dim.max_caption_len)
    )
    n_actions = torch.randint(CONFIG.dim.max_actions, (CONFIG.dim.B,))
    logits = full_model(f, t, n_actions)
    print(logits.size())
    print(logits.argmax(2))
