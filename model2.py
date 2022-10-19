import itertools
from dataclasses import dataclass
from typing import Optional, Generator

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def forward(self, x: Tensor, mask: Optional[Tensor]):
        # multi head attention used in the encoder
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        wv = self.qkv_attention(q, k, v, mask)
        return self.out(wv)

    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor]):
        n_batch, n_ctx, n_state = q.size()
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(q.size(0), q.size(1), self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(k.size(0), k.size(1), self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(v.size(0), v.size(1), self.n_head, -1).permute(0, 2, 1, 3)
        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        w = F.softmax(qk, dim=-1)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int, keygen: Generator):
        super().__init__()
        self.k_id = next(keygen)
        self.v_id = next(keygen)
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def forward(
        self, x: Tensor, xa: Tensor, mask: Optional[Tensor], kv_cache: dict[int, Tensor]
    ):
        # multi head attention used in the decoder
        q = self.query(x)
        k = self.key(xa)
        v = self.value(xa)

        # managing cache
        if self.k_id in kv_cache and self.v_id in kv_cache:
            kv_cache[self.k_id] = torch.cat([kv_cache[self.k_id], k])
            kv_cache[self.v_id] = torch.cat([kv_cache[self.v_id], v])
        else:
            kv_cache[self.k_id] = k
            kv_cache[self.v_id] = v

        wv = self.qkv_attention(q, k, v, mask)
        return self.out(wv)

    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor]):
        n_batch, n_ctx, n_state = q.size()
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(q.size(0), q.size(1), self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(k.size(0), k.size(1), self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(v.size(0), v.size(1), self.n_head, -1).permute(0, 2, 1, 3)
        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        w = F.softmax(qk, dim=-1)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)
        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_mlp),
            nn.GELU(),
            nn.Linear(n_mlp, n_state),
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None):
        # standard encoder attention block with skip connection
        x = x + self.attn(self.attn_ln(x), mask=mask)
        x = x + self.mlp(self.mlp_ln(x))
        return x


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, keygen: Generator):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)
        self.cross_attn = MultiHeadCrossAttention(n_state, n_head, keygen)
        self.cross_attn_ln = nn.LayerNorm(n_state)
        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_mlp),
            nn.GELU(),
            nn.Linear(n_mlp, n_state),
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(
        self, x: Tensor, xa: Tensor, kv_cache: dict[int, Tensor], mask: Tensor = None
    ):
        # decoder attn and cross-attn block with skip connection
        x = x + self.attn(self.attn_ln(x), mask=mask)
        x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache, mask=mask)
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = nn.LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x[0].size() == self.positional_embedding.size(), "incorrect audio shape"
        x = x + self.positional_embedding

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        keygen: Generator,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks = nn.ModuleList(
            [
                ResidualCrossAttentionBlock(n_state, n_head, keygen)
                for _ in range(n_layer)
            ]
        )
        self.ln = nn.LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: dict[int, Tensor]):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        offset = kv_cache[0].size(1) if len(kv_cache) > 0 else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.size(-1)]
        )

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        logits = x @ torch.transpose(self.token_embedding.weight, 0, 1)

        return logits


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.keygen = itertools.count()
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
            self.keygen,
        )

    def forward(self, mel: Tensor, tokens: Tensor):
        cache: dict[int, Tensor] = {}
        encoded = self.encoder(mel)
        logits = self.decoder(tokens, encoded, kv_cache=cache)
        return logits

    def greedy_decode(self):
        # TODO torch.jit.export, cache dict should be from here
        # TODO greedy decode in forward to enable __call__ behaviour?
        pass
