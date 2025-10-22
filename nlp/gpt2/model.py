from attention import MultiHeadSelfAttentionModule
from mlp import FeedForwardModule

import jax.numpy as jnp
import chex
import flax.linen as nn

class DecoderBlock(nn.Module):
    n_embed: int
    num_heads: int

    @nn.compact
    def __call__(self, x:chex.Array) -> chex.Array:
        chex.assert_shape(x, [None, None, self.n_embed])

        attn_out = MultiHeadSelfAttentionModule(
            num_heads=self.num_heads, n_embed=self.n_embed
        )(nn.LayerNorm()(x))
        x = x + attn_out

        ffwd_out = FeedForwardModule(n_embed=self.n_embed)(
            nn.LayerNorm()(x)
        )
        x = x + ffwd_out

        chex.assert_shape(x, [None, None, self.n_embed])
        return x
    

class GPT2(nn.Module):
    vocab_size: int 
    n_embed: int 
    block_size: int 
    num_heads: int
    num_blocks: int

    @nn.compact
    def __call__(self, idx: chex.Array) -> chex.Array:
        # idx -> (B, T (tokens)) 
        # outputs (logits) -> (B, T, vocab_size)
        chex.assert_rank(idx, 2)
        _B, T = idx.shape

        tok_emb = nn.Embed(
            num_embeddings=self.vocab_size, features=self.n_embed
        )(idx)

        pos_emb_table = self.param(
            "pos_emb_table,",
            nn.initializers.normal(),
            (self.block_size, self.n_embed)            
        )
        pos_emb = pos_emb_table[jnp.arange(T)]
        x = tok_emb + pos_emb

        x = nn.Sequential([
            DecoderBlock(n_embed=self.n_embed, num_heads=self.num_heads)
            for _ in range(self.num_blocks)
        ])(x)

        x = nn.LayerNorm()(x)
        logits = nn.Dense(self.vocab_size)(x)

        chex.assert_shape(
            logits, (_B, T, self.vocab_size)
        )

        return logits