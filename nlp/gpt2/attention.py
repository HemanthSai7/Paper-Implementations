import chex
import jax.numpy as jnp
import flax.linen as nn


class MultiHeadSelfAttentionModule(nn.Module):
    num_heads: int
    n_embed: int

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        B, T, C = x.shape
        chex.assert_equal(C, self.n_embed)

        causal_mask = nn.make_causal_mask(jnp.ones((B, T))) #(B,1,T,T)

        return nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.n_embed,
        )(inputs_q=x, mask=causal_mask)