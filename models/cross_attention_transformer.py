from typing import Optional
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads

    def forward(self, x, mask=None):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        attn = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        attn = self.drop(F.softmax(attn, dim=-1))

        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (attn @ v).transpose(1, 2).contiguous()

        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        return h

class MultiHeadedCrossAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads

    def forward(self, x, y, mask=None):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(y), self.proj_v(y)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])

        # (B, H, S, W) @ (B, H, W, S) --> (B, H, S, S) --softmax--> (B, H, S, S)
        cattn = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        cattn = self.drop(F.softmax(cattn, dim=-1))

        # (B, H, S, S) @ (B, H, S, W) --> (B, H, S, W) --trans--> (B, S, H, W)
        h = (cattn @ v).transpose(1, 2).contiguous()

        # --merge--> (B, S, D)
        h = merge_last(h, 2)
        return h

class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        out = self.fc1(x)
        out = F.gelu(out)
        out = self.fc2(out)
        return out

class CrossAttentionBlock(nn.Module):
    """Cross Attention Block"""
    def __init__(self, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.cattn = MultiHeadedCrossAttention(dim, num_heads, dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.norm3 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, y, mask=None):
        h = self.drop(self.proj(self.cattn(self.norm1(x), self.norm2(y), mask)))
        x = x + h
        h = self.drop(self.pwff(self.norm3(x)))
        x = x + h
        return x

class CrossAttentionTransformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])

    def forward(self, x, y, mask=None):
        for block in self.blocks:
            x = block(x, y, mask)
        return x

class CAT_B16(nn.Module):
    def __init__(self, num_layers=12, embed_dim=768, num_heads=12, ff_dim=3072, droprate=0.0):
        super(CAT_B16, self).__init__()
        self.transformer = CrossAttentionTransformer(num_layers=num_layers, dim=embed_dim, num_heads=num_heads,
                                                     ff_dim=ff_dim, dropout=droprate)
        self.init_weights()

    def forward(self, x, y):
        out = self.transformer(x, y)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

def CAT_B16_8layer(**kwargs):
    return CAT_B16(num_layers=8, **kwargs)

def CAT_B16_10layer(**kwargs):
    return CAT_B16(num_layers=10, **kwargs)

def CAT_B16_12layer(**kwargs):
    return CAT_B16(num_layers=12, **kwargs)

model_dict ={
    'cat_b16_8layer': [CAT_B16_8layer, 768],
    'cat_b16_10layer': [CAT_B16_10layer, 768],
    'cat_b16_12layer': [CAT_B16_12layer, 768]
}

class SupCECAT(nn.Module):
    def __init__(self, name='cat_b16_12layer', num_classes=10, droprate=0.0):
        super(SupCECAT, self).__init__()
        cat, feat_dim = model_dict[name]
        self.transformer = cat(droprate=droprate)
        self.norm = nn.LayerNorm(feat_dim, eps=1e-6)
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x, y):
        feats = self.transformer(x, y)
        cls_result = self.norm(feats)[:, 0]
        output = self.fc(cls_result)
        return cls_result, output
