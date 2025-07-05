import re

import tensorflow as tf

__all__ = [
    "TokenizerLayer"
]

class Tokenizer:
    def __init__(
        self,
        stop_words: list[str] | None = None,
        allowed_delimiters: str | list[str] | None = None,
    ):
        self._stop_words = set(stop_word.lower() for stop_word in stop_words) if stop_words else set()
        self._allowed_delimiters = set(allowed_delimiters) if allowed_delimiters else set()
    
    @staticmethod
    def _normalize_text(query: str) -> str:
        return query.lower()

    def _get_tokens(self, query: str) -> list[str]:
        query = self._normalize_text(query)
        tokens = re.findall(r"\w+|[^\w\s]", query)
        refined_tokens = []
        for token in tokens:
            if not token.isalnum() and token in self._allowed_delimiters:
                refined_tokens.append(token)
            elif token.isalnum() and token not in self._stop_words:
                refined_tokens.append(token)
        return refined_tokens

    def tokenize(self, query: str) -> list[str]:
        return self._get_tokens(query)

    def tokens_to_indices(self, tokens: list[str], vocab: dict[str, int], unk_token: str = "[UNK]") -> list[int]:
        return [vocab.get(token, vocab.get(unk_token, 0)) for token in tokens]

    def tokens_to_char_indices(self, tokens: list[str], char_vocab: dict[str, int], max_word_len: int = 16, unk_token: str = "[UNK]") -> list[list[int]]:
        # Returns a list of char indices for each token, padded/truncated to max_word_len
        unk_idx = char_vocab.get(unk_token, 0)
        indices = []
        for token in tokens:
            chars = list(token)
            char_ids = [char_vocab.get(c, unk_idx) for c in chars[:max_word_len]]
            # Pad if needed
            if len(char_ids) < max_word_len:
                char_ids += [0] * (max_word_len - len(char_ids))
            indices.append(char_ids)
        return indices

class TokenizerLayer(tf.keras.layers.Layer):
    def __init__(self, tokenizer_args: dict, name="pre-proceesing_tokenizer", **kwargs):
        super(TokenizerLayer, self).__init__(name=name, **kwargs)
        self.tokenizer = Tokenizer(**tokenizer_args)

    def call(self, inputs: list[str]) -> list[list[str]]:
        inputs = [self.tokenizer.tokenize(input) for input in inputs]
        return inputs


# t = TokenizerLayer(tokenizer_args={"stop_words": ["an", "a", "the"], "allowed_delimiters":[",", ".", "'", "?", "!"]})
# print(t(inputs=["The quick brown fox, jumps over the lazy dog.", "hi hello"]))