import chex
import flax.linen as nn


class FeedForwardModule(nn.Module):
    n_embed: int

    @nn.compact
    def __call__(self, x:chex.Array) -> chex.Array:
        # x -> (B,T,C)
        # output -> (B,T,C)
        chex.assert_shape(x, [None, None, self.n_embed])
        net = nn.Sequential([
            nn.Dense(4* self.n_embed),
            nn.gelu,
            nn.Dense(self.n_embed)
        ])

        return net(x)