from src.model.layers import (
    RoPEPositionalEncoding, CausalMultiHeadAttention, MultiHeadAttention, 
    FFNModule
)

import tensorflow as tf

__all__ = ["MoonshineDecoder"]


@tf.keras.utils.register_keras_serializable(package=__name__)
class TextEmbeddings(tf.keras.layers.Layer):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        name: str = "text_embeddings",
        **kwargs,
    ):
        super(TextEmbeddings, self).__init__(name=name, **kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.text_emb = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=d_model,
            name="text_embedding",
            embeddings_initializer="glorot_uniform",
            embeddings_regularizer=None,
        )

    def call(self, inputs, training=False, mask=None):
        outputs = self.text_emb(inputs, training=training)
        return outputs
    
    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        return (batch_size, seq_len, self.d_model)
    
    def get_config(self):
        config = super(TextEmbeddings, self).get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
        })
        return config
    

@tf.keras.utils.register_keras_serializable(package=__name__)
class MoonshineDecoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim: int,
        decoder_num_blocks: int = 6,
        num_heads: int = 8,
        head_size: int= 64,
        fc_factor: int = 4,
        activation: str = "swiglu",
        dropout: float = 0.0,
        kernel_initializer: str = "glorot_uniform",
        kernel_regularizer: str = None,
        bias_regularizer: str = None,
        bias_initializer: str = "zeros",
        name: str = "decoder_block",
        **kwargs,
    ):
        super(MoonshineDecoderBlock, self).__init__(name=name, **kwargs)
        self.decoder_num_blocks = decoder_num_blocks
        self.num_heads = num_heads
        self.head_size = head_size
        self.fc_factor = fc_factor

        self.self_attn = CausalMultiHeadAttention(
            num_heads=num_heads,
            head_size=head_size,
            output_size=input_dim,
            dropout=dropout,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            bias_initializer=bias_initializer,
            name=f"{name}_self_attn",
        )
        self.cross_attn = MultiHeadAttention(
            num_heads=num_heads,
            head_size=head_size,
            output_size=input_dim,
            dropout=dropout,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            bias_initializer=bias_initializer,
            name=f"{name}_cross_attn",
        )
        self.mlp = FFNModule(
            input_dim=input_dim,
            fc_factor=fc_factor,
            dropout=dropout,
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            bias_initializer=bias_initializer,
            name=f"{name}_mlp",
        )


    def call(self, inputs, training=False, mask=None):
        query_pos_emb, key_pos_emb, outputs, enc_key_emb, enc_val_emb = inputs

        attn_out = self.self_attn([query_pos_emb, key_pos_emb, outputs], training=training)
        cross_out = self.cross_attn([attn_out, enc_key_emb, enc_val_emb], training=training)
        outputs = self.mlp(cross_out, training=training)

        return outputs

    def compute_output_shape(self, input_shape):
        outputs_shape = input_shape[2] # (batch, target_seq_len, d_model)
        return outputs_shape
    
    def get_config(self):
        config = super(MoonshineDecoderBlock, self).get_config()
        config.update({
            "decoder_num_blocks": self.decoder_num_blocks,
            "num_heads": self.num_heads,
            "head_size": self.head_size,
            "fc_factor": self.fc_factor,
        })
        return config
    

@tf.keras.utils.register_keras_serializable(package=__name__)
class MoonshineDecoder(tf.keras.Model):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 288,
        decoder_num_blocks: int = 6,
        num_heads: int = 6,
        head_size: int = 32,
        fc_factor: int = 2,
        activation: str = "swiglu",
        dropout: float = 0.0,
        kernel_initializer: str = "glorot_uniform",
        kernel_regularizer: str = None,
        bias_regularizer: str = None,
        bias_initializer: str = "zeros",
        name: str = "decoder",
        **kwargs,
    ):
        super(MoonshineDecoder, self).__init__(name=name, **kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.decoder_num_blocks = decoder_num_blocks

        self.text_emb = TextEmbeddings(
            vocab_size=vocab_size,
            d_model=d_model,
            name="text_embeddings",
        )

        self.rotary_pos_emb = RoPEPositionalEncoding(name=f"{name}_rope_pos_emb")

        self.decoder_blocks = [
            MoonshineDecoderBlock(
                input_dim=d_model,
                decoder_num_blocks=decoder_num_blocks,
                num_heads=num_heads,
                head_size=head_size,
                fc_factor=fc_factor,
                activation=activation,
                dropout=dropout,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                bias_initializer=bias_initializer,
                name=f"{name}_{i}",
            )
            for i in range(decoder_num_blocks)
        ]
        self.ln = tf.keras.layers.LayerNormalization(
            name=f"{name}_ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
        )
        self.proj = tf.keras.layers.Dense(
            units=vocab_size,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            bias_initializer=bias_initializer,
            name=f"{name}_proj",
        )
        self.do = tf.keras.layers.Dropout(rate=dropout, name=f"{name}_dropout")

    def call(self, inputs, training=False, mask=None):
        encoder_outputs, *decoder_inputs = inputs

        decoder_tokens, decoder_length = decoder_inputs
        outputs = self.text_emb(
            decoder_tokens, training=training
        )

        encoder_embed = encoder_outputs
        encoder_key_emb = self.rotary_pos_emb(encoder_embed)
        encoder_val_emb = encoder_embed

        for i, block in enumerate(self.decoder_blocks):
            query_pos_emb = self.rotary_pos_emb(outputs)
            key_pos_emb = self.rotary_pos_emb(outputs)
            outputs = block(
                [query_pos_emb, key_pos_emb, outputs, encoder_key_emb, encoder_val_emb],
                training=training,
            )

        outputs = self.ln(outputs)
        outputs = self.do(outputs, training=training)
        outputs = self.proj(outputs, training=training)

        return outputs

    def compute_output_shape(self, input_shape):
        if len(input_shape) >= 2:
            predictions_shape = input_shape[1]
        else:
            predictions_shape = (None, None) 

        batch_dim = predictions_shape[0]
        target_seq_len_dim = predictions_shape[1]
        
        # (batch, target_seq_len, vocab_size)
        return (batch_dim, target_seq_len_dim, self.vocab_size)
    
    def get_config(self):
        conf = super().get_config()
        conf.update(self.text_emb.get_config())
        conf.update(self.rotary_pos_emb.get_config())
        conf.update(self.ln.get_config())
        conf.update(self.proj.get_config())
        for block in self.decoder_blocks:
            conf.update(block.get_config())
        return conf


