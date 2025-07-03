import tensorflow as tf

from .mlp import MLP


__all__ = [
    "MultiHeadAttentionBlock"
]

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        n_heads: int,
        n_dims: int
    ):
        super(MultiHeadSelfAttention, self).__init__()
        self.n_heads = n_heads
        self.n_dims = n_dims
    
    def build(self, input_shape): # input_shape: (batch_size, seq_len, embed_dim)
        self.WQ = self.add_weight(
            name = "query_matrix",
            shape = (self.n_heads, input_shape[-1], self.n_dims),
            initializer = "glorot_uniform",
            regularizer = tf.keras.regularizers.l2(1e-6),
            trainable = True
        )

        self.WK = self.add_weight(
            name = "key_matrix",
            shape = (self.n_heads, input_shape[-1], self.n_dims),
            initializer = "glorot_uniform",
            regularizer = tf.keras.regularizers.l2(1e-6),
            trainable = True
        )

        self.WV_down_projection = self.add_weight(
            name = "value_matrix_down_projection",
            shape = (self.n_heads, input_shape[2], self.n_dims),
            initializer = "glorot_uniform",
            trainable = True
        )

        self.WV_up_projection = self.add_weight(
            name = "value_matrix_up_projection",
            shape = (self.n_heads, self.n_dims, input_shape[2]),
            initializer = "glorot_uniform",
            trainable = True
        )

        self.WV = self.WV_down_projection @ self.WV_up_projection
    
    def call(self, inputs): # inputs: (batch_size, seq_len, embed_dim) (bse)
        Q = tf.einsum("bse,ned->nbsd", inputs, self.WQ)
        K = tf.einsum("bse,ned->nbsd", inputs, self.WK)

        attn_scores = tf.einsum("nbsd,nbdS->nbsS", K, tf.transpose(Q, perm=[0, 1, 3, 2])) / tf.sqrt(float(self.n_dims))
        attn_scores = tf.linalg.band_part(attn_scores, 0, -1) # upper triangular matrix
        attn_scores = tf.where(attn_scores == 0, tf.constant(tf.float32.min), attn_scores)
        attn_scores = tf.nn.softmax(attn_scores, axis=-1)

        inp_down_projection = tf.einsum("bse,ned->nbsd", inputs, self.WV_down_projection)
        v_down = tf.einsum("nbss,nbsd->nbsd", attn_scores, inp_down_projection)
        V = tf.einsum("nbsd,nde->nbse", v_down, self.WV_up_projection)

        delta_e = tf.reduce_mean(V, axis=0) # bse

        return inputs + delta_e


class MultiHeadAttentionBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        n_heads: int,
        n_dims: int
    ):
        super(MultiHeadAttentionBlock, self).__init__()
        self.mha = MultiHeadSelfAttention(n_heads=n_heads, n_dims=n_dims)
        self.mlp = MLP()
        self.pre_attn_layernorm = tf.keras.layers.LayerNormalization()
        self.post_attn_layernorm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(0.1)
    
    def call(self, inputs):
        layernorm = self.pre_attn_layernorm(inputs=inputs)
        updated_emb = self.mha(inputs=layernorm)
        updated_emb = self.dropout(inputs=updated_emb)
        updated_emb += inputs
        post_updated_emb = self.post_attn_layernorm(inputs=updated_emb)
        post_updated_emb = self.mlp(inputs=post_updated_emb)
        post_updated_emb = self.dropout(inputs=post_updated_emb)
        post_updated_emb += updated_emb
        return post_updated_emb