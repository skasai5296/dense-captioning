import numpy as np
import torch
import torch.nn as nn

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
            torch.Tensor feature:               (bs x ft_dim)
            torch.Tensor memory:                (bs x rnn_dim)
        Returns:
            torch.Tensor attn_weights:          (bs x ft_dim)
            torch.Tensor weighted_feature:      (bs x ft_dim)
        """
        # encoded_feature: (bs x attn_dim)
        encoded_feature = self.enc_attn(feature)
        # encoded_memory: (bs x attn_dim)
        encoded_memory = self.dec_attn(memory)
        # attn_weights: (bs x ft_dim)
        attn_weights = self.attn(self.relu(encoded_feature + encoded_memory))
        attn_weights = self.softmax(attn_weights)
        weighted_feature = feature * attn_weights
        return attn_weights, weighted_feature


class AttentionDecoder(nn.Module):
    """
    RNN decoder with attention for captioning, Show, Attend and Tell
    Args:
        feature_dim:    dimension of image feature
        emb_dim:        dimension of word embeddings
        memory_dim:     dimension of LSTM memory and attention
        vocab_size:     vocabulary size
        max_seqlen:     max sequence size
        dropout_p:      dropout probability for LSTM memory
        ss_prob:        scheduled sampling rate, 0 for teacher forcing and 1 \
                        for free running
    """

    def __init__(
        self,
        feature_dim,
        emb_dim,
        memory_dim,
        vocab_size,
        max_seqlen,
        dropout_p,
        ss_prob,
        bos_idx,
        pad_idx,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.vocab_size = vocab_size
        self.max_seqlen = max_seqlen
        self.ss_prob = ss_prob
        self.bos_idx = bos_idx

        self.init_h = nn.Linear(feature_dim, memory_dim)
        self.init_c = nn.Linear(feature_dim, memory_dim)
        self.attention = Attention(feature_dim, memory_dim, memory_dim)
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTMCell(emb_dim + feature_dim, memory_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.linear = nn.Linear(memory_dim, vocab_size)
        self.f_beta = nn.Linear(memory_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.xeloss = nn.CrossEntropyLoss(ignore_index=pad_idx)

        self.init_h.apply(weight_init)
        self.init_c.apply(weight_init)
        self.rnn.apply(weight_init)
        self.emb.apply(weight_init)
        self.linear.apply(weight_init)
        self.f_beta.apply(weight_init)

    def forward(self, feature, caption, length):
        """
        Args:
            torch.Tensor feature:       (bs x feature_dim), torch.float
            torch.Tensor caption:       (bs x max_seqlen), torch.long
            torch.Tensor length:        (bs), torch.long
        Returns:
            torch.Tensor out:           (bs x vocab_size x max_seqlen-1), \
                                        contains logits
        """
        bs = feature.size(0)
        # hn, cn: (bs x memory_dim)
        hn = self.init_h(feature)
        cn = self.init_c(feature)
        # cap_emb: (bs x max_seqlen x emb_dim)
        cap_emb = self.emb(caption)
        xn = cap_emb[:, 0, :]
        # out: (bs x vocab_size x max_seqlen-1)
        out = torch.empty(
            (bs, self.vocab_size, self.max_seqlen - 1), device=feature.device
        )
        # alphas: (bs x feature_dim x max_seqlen-1)
        alphas = torch.empty(
            (bs, self.feature_dim, self.max_seqlen - 1), device=feature.device
        )
        for step in range(self.max_seqlen - 1):
            # alpha: (bs x feature_dim)
            # weighted_feature: (bs x feature_dim)
            alpha, weighted_feature = self.attention(feature, hn)
            alphas[:, :, step] = alpha
            # beta: (bs x 1)
            beta = self.sigmoid(self.f_beta(hn))
            weighted_feature *= beta
            # hn, cn: (bs x memory_dim)
            hn, cn = self.rnn(torch.cat([xn, weighted_feature], dim=1), (hn, cn))
            # on: (bs x vocab_size)
            on = self.linear(self.dropout(hn))
            out[:, :, step] = on
            # xn: (bs x emb_dim)
            xn = (
                self.emb(on.argmax(dim=1))
                if np.random.uniform() < self.ss_prob
                else cap_emb[:, step + 1, :]
            )
        losses = {}
        losses["XELoss"] = self.xeloss(out, caption[:, 1:])
        losses["DSLoss"] = ((1.0 - alphas.sum(dim=2)) ** 2).mean()
        return losses

    def sample(self, feature):
        bs = feature.size(0)
        # hn, cn: (bs x memory_dim)
        hn = self.init_h(feature)
        cn = self.init_c(feature)
        # xn: (bs x emb_dim)
        xn = self.emb(
            torch.full((bs,), self.bos_idx, dtype=torch.long, device=feature.device)
        )
        # out: (bs x vocab_size x max_seqlen-1)
        out = torch.empty(
            (bs, self.vocab_size, self.max_seqlen - 1), device=feature.device
        )
        for step in range(self.max_seqlen - 1):
            # alpha: (bs x feature_dim)
            # weighted_feature: (bs x feature_dim)
            _, weighted_feature = self.attention(feature, hn)
            # beta: (bs x 1)
            beta = self.sigmoid(self.f_beta(hn))
            weighted_feature *= beta
            # hn, cn: (bs x memory_dim)
            hn, cn = self.rnn(torch.cat([xn, weighted_feature], dim=1), (hn, cn))
            # on: (bs x vocab_size)
            on = self.linear(self.dropout(hn))
            out[:, :, step] = on
            # xn: (bs x emb_dim)
            xn = self.emb(on.argmax(dim=1))
        return out


class SimpleDecoder(nn.Module):
    """
    RNN decoder for captioning, Google NIC
    Args:
        feature_dim:    dimension of image feature
        emb_dim:        dimension of word embeddings
        memory_dim:     dimension of LSTM memory
        vocab_size:     vocabulary size
        max_seqlen:     max sequence size
        dropout_p:      dropout probability for LSTM memory
        ss_prob:        scheduled sampling rate, 0 for teacher forcing and 1 \
                        for free running
    """

    def __init__(
        self,
        feature_dim,
        emb_dim,
        memory_dim,
        vocab_size,
        max_seqlen,
        dropout_p,
        ss_prob,
        bos_idx,
        pad_idx,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seqlen = max_seqlen
        self.ss_prob = ss_prob
        self.bos_idx = bos_idx

        self.init_h = nn.Linear(feature_dim, memory_dim)
        self.init_c = nn.Linear(feature_dim, memory_dim)
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTMCell(emb_dim, memory_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.linear = nn.Linear(memory_dim, vocab_size)
        self.xeloss = nn.CrossEntropyLoss(ignore_index=pad_idx)

        self.init_h.apply(weight_init)
        self.init_c.apply(weight_init)
        self.rnn.apply(weight_init)
        self.emb.apply(weight_init)
        self.linear.apply(weight_init)

    def forward(self, feature, caption, length):
        """
        Args:
            torch.Tensor feature:       (bs x spatial_size*spatial_size x \
                                        feature_dim), torch.float
            torch.Tensor caption:       (bs x max_seqlen), torch.long
            torch.Tensor length:        (bs), torch.long
        Returns:
            torch.Tensor out:           (bs x vocab_size x max_seqlen-1), \
                                        contains logits
        """
        bs = feature.size(0)
        # hn, cn: (bs x memory_dim)
        hn = self.init_h(feature)
        cn = self.init_c(feature)
        # cap_emb: (bs x max_seqlen x emb_dim)
        cap_emb = self.emb(caption)
        xn = cap_emb[:, 0, :]
        # out: (bs x vocab_size x max_seqlen-1)
        out = torch.empty(
            (bs, self.vocab_size, self.max_seqlen - 1), device=feature.device
        )
        for step in range(self.max_seqlen - 1):
            # hn, cn: (bs x memory_dim)
            hn, cn = self.rnn(xn, (hn, cn))
            # on: (bs x vocab_size)
            on = self.linear(self.dropout(hn))
            out[:, :, step] = on
            # xn: (bs x emb_dim)
            xn = (
                self.emb(on.argmax(dim=1))
                if np.random.uniform() < self.ss_prob
                else cap_emb[:, step + 1, :]
            )
        losses = {}
        losses["XELoss"] = self.xeloss(out, caption[:, 1:])
        return losses

    def sample(self, feature):
        bs = feature.size(0)
        # hn, cn: (bs x memory_dim)
        hn = self.init_h(feature)
        cn = self.init_c(feature)
        # xn: (bs x emb_dim)
        xn = self.emb(
            torch.full((bs,), self.bos_idx, dtype=torch.long, device=feature.device)
        )
        # out: (bs x vocab_size x max_seqlen-1)
        out = torch.empty(
            (bs, self.vocab_size, self.max_seqlen - 1), device=feature.device
        )
        for step in range(self.max_seqlen - 1):
            # hn, cn: (bs x memory_dim)
            hn, cn = self.rnn(xn, (hn, cn))
            # on: (bs x vocab_size)
            on = self.linear(self.dropout(hn))
            out[:, :, step] = on
            # xn: (bs x emb_dim)
            xn = self.emb(on.argmax(dim=1))
        return out
