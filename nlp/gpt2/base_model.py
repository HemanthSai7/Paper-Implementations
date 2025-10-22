from functools import partial
from typing import Dict, Tuple, Callable
from config import Config
from utils import Data

import jax
import jax.numpy as jnp
import chex
import optax
import flax.linen as nn
from flax.training import train_state


class TrainingState(train_state.TrainState):
    """A custom training state"""
    pass

@partial(jax.jit, static_argnames=("model_apply_fn",))
def train_step(
    state: TrainingState,
    batch: Tuple[chex.Array, chex.Array],
    model_apply_fn: Callable,
)-> Tuple[TrainingState, jnp.ndarray]:
    batch_x, batch_y = batch

    def loss_fn(params: Dict) -> chex.Array:
        logits = model_apply_fn({"params": params}, batch_x)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=batch_y
        ).mean()
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@partial(jax.jit, static_argnames=("model_apply_fn",))
def eval_step(
    state: TrainingState,
    batch: Tuple[chex.Array, chex.Array],
    model_apply_fn: Callable,
) -> chex.Array:
    batch_x, batch_y = batch
    logits = model_apply_fn({"params": state.params}, batch_x)
    return optax.softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=batch_y,
    ).mean()

def evalaute_model(
    state: TrainingState,
    val_data: chex.Array,
    data_key: chex.Array,
) -> float:
    total_loss = 0.0
    for i in range(Config.EVAL_STEPS):
        data_key, subkey = jax.random.split(data_key)
        batch_x, batch_y = get_batch(
            val_data, subkey, Config.BATCH_SIZE, Config.BLOCK_SIZE
        )
        loss = eval_step(state, (batch_x, batch_y), state.apply_fn)
        total_loss += loss
    return total_loss / Config.EVAL_STEPS

@partial(jax.jit, static_argnames=("batch_size","block_size"))
def get_batch(
    data: chex.Array,
    key: chex.Array,
    batch_size: int,
    block_size: int
) -> Tuple[chex.Array, chex.Array]:
    idx = jax.random.randint(
        key, (batch_size, ), 0, len(data) - block_size
    )

    sequences = jax.vmap(
        lambda i: jax.lax.dynamic_slice(
            data, (i,), (block_size + 1,)
        )
    )(idx)

    x = sequences[: ,:-1]
    y = sequences[:, 1:]
    return x, y

@partial(jax.jit, static_argnames=('apply_fn', 'max_new_tokens', 'block_size'))
def generate(
    params: Dict, 
    apply_fn: Callable, 
    key: chex.Array,
    max_new_tokens: int, 
    block_size: int
) -> chex.Array:
    """Generates text from the model using a compiled lax.scan loop."""
    initial_tokens = jnp.zeros((1, 1), dtype=jnp.uint16)

    def scan_fn(carry, _):
        """Single generation step for use in lax.scan."""
        key, current_tokens = carry
        
        # Crop context to the last `block_size` tokens
        cond_tokens = current_tokens[:, -block_size:]
        
        logits = apply_fn({'params': params}, cond_tokens)
        logits_last = logits[:, -1, :] # (B, vocab_size)

        # Sample the next token
        key, subkey = jax.random.split(key)
        next_token = jax.random.categorical(subkey, logits_last).astype(jnp.uint16)
        
        # Append the new token to the sequence
        new_tokens = jnp.concatenate(
            [current_tokens[:, 1:], next_token[:, None]], axis=1
        )
        return (key, new_tokens), next_token

    # Run the generation loop efficiently with lax.scan
    (_, _), all_new = jax.lax.scan(
        scan_fn, (key, initial_tokens), None, length=max_new_tokens
    )
    return all_new.flatten()


def create_train_state(
    model_cls: nn.Module,
    key: chex.Array,
    data: Data
) -> TrainingState:
    model = model_cls(
        vocab_size=data.vocab_size,
        n_embed=Config.N_EMBED,
        block_size=Config.BLOCK_SIZE,
        num_heads=Config.NUM_HEADS,
        num_blocks=Config.NUM_BLOCKS,
    )
    init_key, key = jax.random.split(key)
    params = model.init(
        init_key, jnp.ones((1, Config.BLOCK_SIZE), dtype=jnp.uint16)
    )["params"]

    tx = optax.adamw(Config.LEARNING_RATE)

    return TrainingState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

def run_training_loop(
    state: TrainingState,
    data: Data,
    key: chex.Array,
) -> TrainingState:
    for step in range(Config.TRAIN_STEPS):
        key, train_key, eval_key = jax.random.split(key, 3)

        batch_x, batch_y = get_batch(
            data.train, train_key, Config.BATCH_SIZE, Config.BLOCK_SIZE
        )

        state, loss = train_step(state, (batch_x, batch_y), state.apply_fn)

        if step % Config.EVAL_INTERVAL == 0 or step == Config.TRAIN_STEPS - 1:
            val_loss = evalaute_model(state, data.val, eval_key)
            print(
                f"Step {step:4d} | "
                f"Train Loss: {loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )
    return state