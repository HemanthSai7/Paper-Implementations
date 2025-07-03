from layers import (
    TokenizerLayer,
    EmbeddingLayer,
    PositionalEncoding,
    MultiHeadAttentionBlock,
    UnembeddingLayer
)

import tensorflow as tf

class Sparrow(tf.keras.Model):
    def __init__(self,
        tokenizer_args: dict,
        n_blocks: int = 1,
        n_heads: int = 2,
        n_dims: int = 128,
        max_len: int = 150,
        is_serving: bool = False,
        temperature: float = 1.0,
        **kwargs
    ):
        super(Sparrow, self).__init__(**kwargs)

        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.n_dims = n_dims
        self.max_len = max_len
        self.is_serving = is_serving
        self.temperature = temperature if is_serving else 1.0

        self.tokenizer_layer = TokenizerLayer(tokenizer_args)
        self.embedding_layer = EmbeddingLayer(max_len=max_len)
        self.positional_encoding = PositionalEncoding()
        self.mha_blocks = [MultiHeadAttentionBlock(n_heads=n_heads, n_dims=n_dims) for _ in range(n_blocks)]
        self.unembedding_layer = UnembeddingLayer(vocab_size=len(self.embedding_layer.wordToIndex))    
        self.final_layernorm = tf.keras.layers.LayerNormalization()    

    def build(self, input_shape):
        return super().build(input_shape)


    def call(self, inputs: list[str] | tf.Tensor) -> tf.Tensor:
        tokens = inputs
        if self.is_serving:
            tokens = self.tokenizer_layer(inputs=inputs)
        
        if self.is_serving:
            emb = self.embedding_layer(inputs=tokens)
        else:
            emb = self.embedding_layer.layer(inputs=tokens)
        
        pos_emb =  self.positional_encoding(inputs=emb)
        
        updated_emb = pos_emb
        for mha_block in self.mha_blocks:
            updated_emb = mha_block(inputs=updated_emb)
            updated_emb = self.final_layernorm(inputs=updated_emb)

        unembedding_tensor = self.unembedding_layer(inputs=updated_emb[:, -1, :])

        temperature_scaled_unembedding_tensor = unembedding_tensor / (self.temperature + 1e-7)
        logits = tf.nn.softmax(temperature_scaled_unembedding_tensor, axis=-1)
        # word_indices = tf.argmax(logits, axis=-1)
        # for word_index in word_indices:
        #     print(self.embedding_layer.indexToWord[word_index.numpy()])
        return logits
    

# model = Sparrow(
#     tokenizer_args={"stop_words": ["an", "a", "the"], "allowed_delimiters":[",", ".", "'", "?", "!"]},
#     n_blocks=1,
#     n_heads=2,
#     n_dims=128,
#     max_len=50,
#     is_serving=True
# )
# print(model(inputs=["Hello, world!", "the fox!!!"]))
# print(model.summary())
# print(model(inputs=["hi", "hello"]))

# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# data = tf.data.Dataset.from_tensor_slices((["Hello ", "the fox "], ["world", "animal"])).batch(2)

# model.fit(data, epochs=1, batch_size=2)
# print(model(inputs=["hi"]))