import os
import requests
import chex
from typing import Dict, List, Tuple, Callable, NamedTuple
import jax.numpy as jnp

class Data(NamedTuple):
    train: chex.Array
    val: chex.Array
    chars: List[str]
    vocab_size: int
    encode: Callable[[str], List[str]]
    decode: Callable[[List[int]], str]

def download_data(url: str, file_path: str) -> None:
    """Downloads data if it doesn't exist."""
    if not os.path.exists(file_path):
        print(f"Downloading data to {file_path}...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
    else:
        print("Data already exists.")


def prepare_data(file_path: str) -> Data:
    """Loads and preprocesses the text data."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text_data = f.read()

    chars = sorted(list(set(text_data)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos.get(i, '') for i in l])

    n = len(text_data)
    train_data = text_data[:int(n * 0.9)]
    val_data = text_data[int(n * 0.9):]

    return Data(
        train=jnp.array(encode(train_data), dtype=jnp.uint16),
        val=jnp.array(encode(val_data), dtype=jnp.uint16),
        chars=chars,
        vocab_size=len(chars),
        encode=encode,
        decode=decode
    )
