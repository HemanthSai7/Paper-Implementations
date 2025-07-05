import tensorflow as tf
from tensorflow.keras import layers
from embedding import WordEmbedding, CharEmbedding
from attention import ContextQueryAttention
from model_encoder import EmbeddingEncoderBlock, ModelEncoderBlock
from decoder import OutputLayer

@tf.keras.utils.register_keras_serializable()
class QANet(tf.keras.Model):
    def __init__(
        self,
        word_vocab_size,
        char_vocab_size,
        word_embedding_dim=300,
        char_embedding_dim=64,
        num_filters=128,
        kernel_size=7,
        num_heads=8,
        ffn_dim=128,
        num_encoder_blocks=1,
        num_model_blocks=7,
        dropout_rate=0.1,
        pretrained_embeddings=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        # Embedding layers
        self.word_embedding = WordEmbedding(
            vocab_size=word_vocab_size,
            embedding_dim=word_embedding_dim,
            pretrained_embeddings=pretrained_embeddings,
            trainable=False,
            mask_zero=False
        )
        self.char_embedding = CharEmbedding(
            char_vocab_size=char_vocab_size,
            char_embedding_dim=char_embedding_dim,
            num_filters=num_filters,
            kernel_size=5,
            trainable=True
        )
        # Embedding encoder block
        self.embedding_encoder = EmbeddingEncoderBlock(
            num_conv_layers=4,
            num_filters=num_filters,
            kernel_size=kernel_size,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout_rate=dropout_rate,
            project_input=True
        )
        # Context-query attention
        self.cq_attention = ContextQueryAttention(hidden_dim=num_filters)
        # Model encoder blocks (stacked)
        self.model_encoder = ModelEncoderBlock(
            num_blocks=num_model_blocks,
            num_conv_layers=2,
            num_filters=num_filters,
            kernel_size=kernel_size,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout_rate=dropout_rate
        )
        # Output layer
        self.output_layer = OutputLayer(hidden_dim=num_filters * 4)
        # Projection after attention to match model encoder input
        self.attention_projection = layers.Dense(num_filters)

    def call(self, inputs, context_mask=None, query_mask=None, training=False):
        # Accepts a dict of inputs for Keras model.fit compatibility
        context_word_ids = inputs["context_word_ids"]
        context_char_ids = inputs["context_char_ids"]
        query_word_ids = inputs["query_word_ids"]
        query_char_ids = inputs["query_char_ids"]
        # Embedding
        word_emb_c = self.word_embedding(context_word_ids)
        word_emb_q = self.word_embedding(query_word_ids)
        char_emb_c = self.char_embedding(context_char_ids)
        char_emb_q = self.char_embedding(query_char_ids)
        emb_c = tf.concat([word_emb_c, char_emb_c], axis=-1)
        emb_q = tf.concat([word_emb_q, char_emb_q], axis=-1)
        # Embedding encoder
        enc_c = self.embedding_encoder(emb_c, training=training)
        enc_q = self.embedding_encoder(emb_q, training=training)
        # Context-query attention
        att = self.cq_attention(enc_c, enc_q, context_mask=context_mask, query_mask=query_mask)
        # Project attention output to num_filters for model encoder residual compatibility
        att_proj = self.attention_projection(att)
        # Model encoder blocks (3 times for output, as in QANet)
        M0 = self.model_encoder(att_proj, training=training, mask=context_mask)
        M1 = self.model_encoder(M0, training=training, mask=context_mask)
        M2 = self.model_encoder(M1, training=training, mask=context_mask)
        # Output layer
        prob_start, prob_end = self.output_layer(M0, M1, M2, mask=context_mask)
        return (prob_start, prob_end)

    def compute_output_shape(self, input_shape):
        # input_shape is a dict of shapes
        batch_size = input_shape["context_word_ids"][0]
        context_len = input_shape["context_word_ids"][1]
        # Output: tuple of (prob_start, prob_end) each (batch, context_len)
        return (tf.TensorShape([batch_size, context_len]), tf.TensorShape([batch_size, context_len]))

    def get_config(self):
        config = super().get_config()
        config.update({
            'word_embedding': self.word_embedding.get_config(),
            'char_embedding': self.char_embedding.get_config(),
            'embedding_encoder': self.embedding_encoder.get_config(),
            'cq_attention': self.cq_attention.get_config(),
            'model_encoder': self.model_encoder.get_config(),
            'output_layer': self.output_layer.get_config(),
            'attention_projection': self.attention_projection.get_config()
        })
        return config
