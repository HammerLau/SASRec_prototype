import torch
import torch.nn as nn
import numpy as np


class MambaBlock(nn.Module):
    def __init__(self, dim, seq_len):
        super(MambaBlock, self).__init__()
        self.dim = dim
        self.seq_len = seq_len

        self.in_proj = nn.Linear(dim, dim)
        self.conv1d = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=1)
        self.act = nn.GELU()
        self.out_proj = nn.Linear(dim, dim)

        # Optional: learnable gating mechanism
        self.gate = nn.Parameter(torch.ones(1, 1, dim))

    def forward(self, x):
        # x: (B, T, C)
        residual = x
        x = self.in_proj(x)  # (B, T, C)
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.conv1d(x)  # (B, C, T)
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.act(x * self.gate)  # gating (element-wise)
        x = self.out_proj(x)
        return x + residual  # Residual connection


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        return outputs + inputs


class SASRec(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)

        self.mamba_layernorms = nn.ModuleList()
        self.mamba_blocks = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            self.mamba_layernorms.append(nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.mamba_blocks.append(MambaBlock(args.hidden_units, args.maxlen))
            self.forward_layernorms.append(nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(args.hidden_units, args.dropout_rate))

    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))  # (B, T, C)
        seqs *= self.item_emb.embedding_dim ** 0.5

        pos_ids = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        pos_ids *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(pos_ids).to(self.dev))
        seqs = self.emb_dropout(seqs)

        for i in range(len(self.mamba_blocks)):
            seqs = self.mamba_layernorms[i](seqs)
            seqs = self.mamba_blocks[i](seqs)
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]  # Use last position

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits
