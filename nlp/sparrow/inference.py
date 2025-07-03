from model import Sparrow

import tensorflow as tf

def infer(max_tokens: int, query: str, model: Sparrow):
    if max_tokens == 0:
        return
    
    index2word = model.embedding_layer.indexToWord
    logits = model(inputs=[query])
    token = tf.argmax(logits, axis=-1).numpy()[0]
    word = index2word.get(token, "<UNK>")
    print(word, end=" ")
    return infer(max_tokens - 1, query + " " + word, model)

model = Sparrow(
        tokenizer_args={"stop_words": ["an", "a", "the"], "allowed_delimiters":[",", ".", "'", "?", "!"]},
        n_blocks=1,
        n_heads=2,
        n_dims=128,
        max_len=50,
        is_serving=True,
        temperature=0.5
)

print("Hello, How ", end=" ")
infer(10, "Hello, How ", model)