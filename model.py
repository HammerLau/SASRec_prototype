import numpy as np
import torch
from torch import nn


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
        outputs += inputs
        return outputs


# def select_prototype_prefix(user_emb, prototype_embs, pad_len, pos_ratio=0.7):
#     """
#     根据用户兴趣向量 user_emb，从原型集合 prototype_embs 中选出 pad_len 个向量。
#     - 其中按 pos_ratio 选择相似的正原型，其余为相对不相似的难负原型。
#     - 返回 shape: (B, pad_len, D)
#     """
#     sim = torch.matmul(user_emb, prototype_embs.T)  # (B, K)
#     sorted_sim, sorted_idx = torch.sort(sim, dim=-1, descending=True)
#
#     B, K = sim.size()
#     M = (pad_len * pos_ratio)
#     M = torch.clamp(M.long(), min=1, max=pad_len)  # 至少1个正原型
#     N = pad_len - M
#
#     proto_list = []
#     for i in range(B):
#         pos_idx = sorted_idx[i][:M[i]]
#         neg_idx = sorted_idx[i][-N[i]:] if N[i] > 0 else []
#         proto_pos = prototype_embs[pos_idx]
#         proto_neg = prototype_embs[neg_idx] if N[i] > 0 else []
#         proto_full = torch.cat([proto_neg, proto_pos], dim=0)  # (pad_len, D)
#         proto_list.append(proto_full)
#
#     return torch.stack(proto_list)  # (B, pad_len, D)
def select_prototype_prefix(user_emb, prototype_embs, pad_len, pos_ratio):
    """
    user_emb: (B, D)
    prototype_embs: (K, D)
    pad_len: int，表示需要填充的原型个数
    pos_ratio: 正原型占比（如0.7表示正:负=7:3）

    返回:
        proto_list: (B, pad_len, D)  每个用户填充 pad_len 个原型，其中一部分是正的，一部分是难负样本
    """
    sim = torch.matmul(user_emb, prototype_embs.T)  # (B, K)
    sorted_sim, sorted_idx = torch.sort(sim, dim=-1, descending=True)

    B, K = sim.size()
    proto_list = []

    for i in range(B):
        num_pos = int(pad_len * pos_ratio)
        num_pos = min(num_pos, pad_len - 1)  # 至少留1个负样本
        num_neg = pad_len - num_pos

        pos_idx = sorted_idx[i][:num_pos]
        neg_idx = sorted_idx[i][-num_neg:] if num_neg > 0 else []

        proto_pos = prototype_embs[pos_idx]  # (num_pos, D)
        proto_neg = prototype_embs[neg_idx] if num_neg > 0 else []

        proto_full = torch.cat([proto_neg, proto_pos], dim=0)  # (pad_len, D)
        proto_list.append(proto_full)

    return torch.stack(proto_list)  # (B, pad_len, D)


class GlobalCausalAttention(nn.Module):
    def __init__(self, hidden_units, num_heads, dropout):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_units,
                                         num_heads=num_heads,
                                         dropout=dropout,
                                         batch_first=True)  # 使用 batch_first 更直观

    def forward(self, q, k, v, attn_mask):
        # attn_mask: shape (L, L) or (B, L, L)
        output, _ = self.mha(q, k, v, attn_mask=attn_mask)
        return output


class SASRec(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.args = args

        self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)

        self.layernorms = nn.ModuleList()
        self.attn_layers = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.ratio = nn.Parameter(torch.zeros(1)).cuda()

        for _ in range(args.num_blocks):
            self.layernorms.append(nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.attn_layers.append(GlobalCausalAttention(args.hidden_units, args.num_heads, args.dropout_rate))
            self.ffn_norms.append(nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.ffn_layers.append(PointWiseFeedForward(args.hidden_units, args.dropout_rate))

        self.prototype_embs = None  # 外部注入原型向量

    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5

        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        # if self.prototype_embs is not None:
        #     pad_len = self.args.maxlen - seqs.size(1)
        #     if pad_len >= 1:
        #         prototype = self.prototype_embs[torch.randint(0, self.prototype_embs.size(0), (1,))]
        #         prototype = prototype.unsqueeze(0).repeat(seqs.size(0), pad_len, 1)
        #         seqs = torch.cat([prototype, seqs], dim=1)

        if self.prototype_embs is not None:
            pad_len = self.args.maxlen - seqs.size(1)
            if pad_len >= 1:
                with torch.no_grad():
                    user_vec = seqs.mean(dim=1)  # (B, D)，可替换成更精细的 user representation 提取方式
                    prefix_proto = select_prototype_prefix(user_vec, self.prototype_embs, pad_len, pos_ratio=self.args.proto_pos_ratio)
                    seqs = torch.cat([prefix_proto.to(seqs.device), seqs], dim=1)

        B, L, D = seqs.size()

        # ✅ 生成 (B, L, L) 的 causal attention mask，True 表示 masked
        base_mask = torch.triu(torch.ones(L, L, device=self.dev), diagonal=1).bool()  # (L, L)
        attention_mask = base_mask.unsqueeze(0).expand(B, L, L)  # (B, L, L)

        for i in range(len(self.attn_layers)):
            Q = self.layernorms[i](seqs)
            seqs = Q + self.attn_layers[i](Q, seqs, seqs, attention_mask)
            seqs = self.ffn_norms[i](seqs)
            seqs = self.ffn_layers[i](seqs)

        return self.last_layernorm(seqs)

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits


def compute_prototypes(user_train, item_emb_layer, K=10):
    from sklearn.cluster import KMeans
    user_vecs = []
    for u in user_train:
        items = user_train[u]
        if len(items) == 0:
            continue
        item_tensor = torch.LongTensor(items).to(item_emb_layer.weight.device)
        emb_mean = item_emb_layer(item_tensor).mean(dim=0)
        user_vecs.append(emb_mean.cpu().detach().numpy())
    kmeans = KMeans(n_clusters=K).fit(user_vecs)
    return torch.FloatTensor(kmeans.cluster_centers_).to(item_emb_layer.weight.device)
