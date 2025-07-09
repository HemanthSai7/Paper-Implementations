from src.utils import file_util

import codecs
import unicodedata
import tensorflow as tf
import tensorflow_text as tft

logger = tf.get_logger()

ENGLISH_CHARACTERS = [
    "<blank>",
    " ",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "'",
]

class Tokenizer:
    def __init__(self, decoder_config):
        self.scorer = None
        self.decoder_config = decoder_config
        self.blank = None
        self.tokens2indices = {}
        self.tokens = []
        self.num_classes = None
        self.max_length = 0

    @classmethod
    def corpus_generator(cls, decoder_config):
        for file_path in file_util.preprocess_paths(decoder_config.train_files):
            logger.info(f"Reading {file_path} ...")
            with tf.io.gfile.GFile(file_path, "r") as f:
                temp_lines = f.read().splitlines()
                for line in temp_lines[1:]:  # Skip the header of tsv file
                    data = line.split("\t", 2)[-1]  # get only transcript
                    data = cls.normalize_text(data, decoder_config).numpy()
                    yield data

    @property
    def shape(self) -> list:
        return [self.max_length if self.max_length > 0 else None]

    @property
    def prepand_shape(self) -> list:
        return [self.max_length + 1 if self.max_length > 0 else None]

    def update_length(
        self,
        length: int,
    ):
        self.max_length = max(self.max_length, length)

    def reset_length(self):
        self.max_length = 0

    @classmethod
    def normalize_text(cls, text: tf.Tensor, decoder_config):
        text = tf.strings.regex_replace(text, b"\xe2\x81\x87".decode("utf-8"), "")
        text = tft.normalize_utf8(text, decoder_config["normalization_form"])
        text = tf.strings.regex_replace(text, r"\p{Cc}|\p{Cf}", " ")
        # text = tf.strings.regex_replace(text, decoder_config["unknown_token"], "")
        # text = tf.strings.regex_replace(text, decoder_config["pad_token"], "")
        text = tf.strings.regex_replace(text, r" +", " ")
        text = tf.strings.lower(text, encoding="utf-8")
        text = tf.strings.strip(text)  # remove trailing whitespace
        return text

    def add_scorer(self, scorer: any = None):
        """Add scorer to this instance"""
        self.scorer = scorer

    def normalize_indices(self, indices: tf.Tensor) -> tf.Tensor:
        """
        Remove -1 in indices by replacing them with blanks
        Args:
            indices (tf.Tensor): shape any

        Returns:
            tf.Tensor: normalized indices with shape same as indices
        """
        with tf.name_scope("normalize_indices"):
            minus_one = -1 * tf.ones_like(indices, dtype=tf.int32)
            blank_like = self.blank * tf.ones_like(indices, dtype=tf.int32)
            return tf.where(tf.equal(indices, minus_one), blank_like, indices)

    def prepand_blank(self, text: tf.Tensor) -> tf.Tensor:
        """Prepand blank index for transducer models"""
        return tf.concat([[self.blank], text], 0)

    def tokenize(self, text: str) -> tf.Tensor:
        raise NotImplementedError()

    def detokenize(self, indices: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def detokenize_unicode_points(self, indices: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

class CharacterTokenizer(Tokenizer):
    """
    Extract text feature based on char-level granularity.
    By looking up the vocabulary table, each line of transcript will be
    converted to a sequence of integer indexes.
    """

    def __init__(self, decoder_config):
        print(decoder_config)
        super().__init__(decoder_config)
        lines = []
        if self.decoder_config["vocabulary"] is not None:
            with codecs.open(self.decoder_config["vocabulary"], "r", "utf-8") as fin:
                lines.extend(fin.readlines())
        else:
            lines = ENGLISH_CHARACTERS
        self.blank = self.decoder_config["blank_index"]
        self.tokens = []
        for line in lines:
            line = unicodedata.normalize(self.decoder_config["normalization_form"], line.lower()).strip("\n")
            if line.startswith("#") or not line:
                continue
            if line == "<blank>":
                line = ""
            self.tokens.append(line)
        if self.blank is None:
            self.blank = len(self.tokens)  # blank not at zero
        self.num_classes = len(self.tokens)
        self.indices = tf.range(self.num_classes, dtype=tf.int32)
        self.tokenizer = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys=self.tokens, values=self.indices, key_dtype=tf.string, value_dtype=tf.int32),
            default_value=self.blank,
        )
        self.detokenizer = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys=self.indices, values=self.tokens, key_dtype=tf.int32, value_dtype=tf.string),
            default_value=self.tokens[self.blank],
        )
        self.upoints = tf.strings.unicode_decode(self.tokens, "UTF-8").to_tensor(shape=[None, 1])

    def tokenize(self, text):
        text = self.normalize_text(text, self.decoder_config)
        text = tf.strings.unicode_split(text, "UTF-8")
        return self.tokenizer.lookup(text)

    def detokenize(self, indices: tf.Tensor) -> tf.Tensor:
        """
        Convert list of indices to string
        Args:
            indices: tf.Tensor with dim [B, None]

        Returns:
            transcripts: tf.Tensor of dtype tf.string with dim [B]
        """
        indices = self.normalize_indices(indices)
        # indices = tf.ragged.boolean_mask(indices, tf.not_equal(indices, self.blank))
        tokens = self.detokenizer.lookup(indices)
        tokens = tf.strings.reduce_join(tokens, axis=-1)
        tokens = self.normalize_text(tokens, self.decoder_config)
        return tokens
    
    def get_vocab(self) -> list:
        """
        Get the vocabulary of this tokenizer
        Returns:
            list: vocabulary of this tokenizer
        """
        return self.tokens

    @tf.function(input_signature=[tf.TensorSpec([None], dtype=tf.int32)])
    def detokenize_unicode_points(self, indices: tf.Tensor) -> tf.Tensor:
        """
        Transform Predicted Indices to Unicode Code Points (for using tflite)
        Args:
            indices: tf.Tensor of Classes in shape [None]

        Returns:
            unicode code points transcript with dtype tf.int32 and shape [None]
        """
        with tf.name_scope("indices2upoints"):
            indices = self.normalize_indices(indices)
            upoints = tf.gather_nd(self.upoints, tf.expand_dims(indices, axis=-1))
            return tf.gather_nd(upoints, tf.where(tf.not_equal(upoints, 0)))