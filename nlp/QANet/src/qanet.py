import tensorflow as tf

# from src.tokenizer import TokenizerLayer
from src.layers.decoder import QANetDecoder
from src.layers.attention import ContextQueryAttention
from src.layers.embedding import WordEmbedding, CharEmbedding
from src.layers.encoder import EmbeddingEncoderBlock, QANetEncoderBlock

class QANet(tf.keras.Model):
    def __init__(
        self,
        # tokenizer_args: dict,
        word_vocab_size: int,
        char_vocab_size: int,
        word_embedding_dim: int,
        char_embedding_dim: int,
        num_filters: int = 128,
        kernel_size: int = 7,
        num_heads: int = 8,
        ffn_dim: int = 128,
        num_encoder_blocks:int = 1,
        num_model_blocks: int = 7,
        dropout_rate: float = 0.1,
        pretrained_embeddings: bool = False,
        is_serving: bool = False,
        name="qanet",
        **kwargs
    ):
        super(QANet, self).__init__(name=name, **kwargs)
        # self.tokenizer_layer = TokenizerLayer(tokenizer_args)
        self.word_embedding_layer = WordEmbedding(
            word_vocab_size=word_vocab_size,
            word_embedding_dim=word_embedding_dim,
            pretrained_embeddings=pretrained_embeddings,
            trainable=False,
            mask_zero=False,
        )
        self.char_embedding_layer = CharEmbedding(
            char_vocab_size=char_vocab_size,
            char_embedding_dim=char_embedding_dim,
            num_filters=num_filters,
            kernel_size=3,
            trainable=True,
            name="char_embedding_layer",
        )
        self.embedding_encoder = EmbeddingEncoderBlock(
            num_conv_layers=4,
            num_filters=num_filters,
            kernel_size=kernel_size,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout_rate=dropout_rate,
            project_input=True,
        )
        self.cq_attention = ContextQueryAttention(
            hidden_dim=num_filters,
            name="context_query_attention",
        )
        self.model_encoder = QANetEncoderBlock(
            num_blocks=num_model_blocks,
            num_conv_layers=2,
            num_filters=num_filters,
            kernel_size=kernel_size,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout_rate=dropout_rate,
        )
        self.decoder = QANetDecoder(hidden_dim=num_filters * 4)
        self.attention_projection = tf.keras.layers.Dense(num_filters, name="attn_proj")
        self.is_serving = is_serving

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, context_mask=None, query_mask=None, training=False) -> tf.Tensor:
        context_word_tokens = inputs["context_word_ids"]
        context_char_tokens = inputs["context_char_ids"]
        query_word_tokens = inputs["query_word_ids"]
        query_char_tokens = inputs["query_char_ids"]

        if self.is_serving:
            tokens = self.tokenizer_layer(inputs=inputs)

        word_embedding_context = self.word_embedding_layer(context_word_tokens)
        word_embedding_query = self.word_embedding_layer(query_word_tokens)
        char_embedding_context = self.char_embedding_layer(context_char_tokens)
        char_embedding_query = self.char_embedding_layer(query_char_tokens)

        context_embedding = tf.concat([word_embedding_context, char_embedding_context], axis=-1)
        query_embedding = tf.concat([word_embedding_query, char_embedding_query], axis=-1)

        context_encoder = self.embedding_encoder(context_embedding, training=training, mask=context_mask)
        query_encoder = self.embedding_encoder(query_embedding, training=training, mask=query_mask)

        attention = self.cq_attention([context_encoder, query_encoder], context_mask=context_mask, query_mask=query_mask)
        attention_proj = self.attention_projection(attention)

        M0 = self.model_encoder(attention_proj, training=training, mask=context_mask)
        M1 = self.model_encoder(M0, training=training, mask=context_mask)
        M2 = self.model_encoder(M1, training=training, mask=context_mask)

        prob_start, prob_end = self.decoder(inputs=[M0, M1, M2], mask=context_mask)
        return (prob_start, prob_end)

    def compute_output_shape(self, input_shape):
        batch_size = input_shape["context_word_tokens"][0]
        context_len = input_shape["context_word_tokens"][1]

        return (tf.TensorShape([batch_size, context_len]), tf.TensorShape([batch_size, contex]))

    def get_config(self):
        config = super().get_config()
        config.update({
            'word_embedding': self.word_embedding.get_config(),
            'char_embedding': self.char_embedding.get_config(),
            'embedding_encoder': self.embedding_encoder.get_config(),
            'cq_attention': self.cq_attention.get_config(),
            'model_encoder': self.model_encoder.get_config(),
            'decoder': self.decoder.get_config(),
            'attention_projection': self.attention_projection.get_config()
        })
        return config

