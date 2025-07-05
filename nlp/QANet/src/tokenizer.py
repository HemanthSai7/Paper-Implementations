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
        if stop_words:
            self._stop_words = set(stop_word.lower() for stop_word in stop_words)
        if allowed_delimiters:
            self._allowed_delimiters = {delimiter for delimiter in allowed_delimiters}
        
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

class TokenizerLayer(tf.keras.layers.Layer):
    def __init__(self, tokenizer_args: dict, name="pre-proceesing_tokenizer", **kwargs):
        super(TokenizerLayer, self).__init__(name=name, **kwargs)
        self.tokenizer = Tokenizer(**tokenizer_args)

    def call(self, inputs: list[str]) -> list[list[str]]:
        inputs = [self.tokenizer.tokenize(input) for input in inputs]
        return inputs


# t = TokenizerLayer(tokenizer_args={"stop_words": ["an", "a", "the"], "allowed_delimiters":[",", ".", "'", "?", "!"]})
# print(t(inputs=["The quick brown fox, jumps over the lazy dog.", "hi hello"]))