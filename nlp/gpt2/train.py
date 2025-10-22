from utils import download_data, prepare_data
from base_model import (
    create_train_state, 
    run_training_loop, 
    generate,
)
from model import GPT2
from config import Config
import jax


def main():
    """Main function to run the data loading, training, and generation."""
    file_path = './data/shakespeare_char/input.txt'
    url = (
        'https://raw.githubusercontent.com/karpathy/char-rnn/'
        'master/data/tinyshakespeare/input.txt'
    )
    download_data(url, file_path)

    # Setup keys for JAX's pseudo-random number generation
    key = jax.random.PRNGKey(0)
    model_key, train_key, gen_key = jax.random.split(key, 3)

    # Prepare data, model, and state
    data = prepare_data(file_path)
    state = create_train_state(GPT2, model_key, data)

    # Run training
    print("Starting training...")
    state = run_training_loop(state, data, train_key)
    print("Training finished.")

    # Generate text from the trained model
    print("\n--- Generated Text ---")
    generated_tokens = generate(
        state.params,
        state.apply_fn,
        gen_key,
        max_new_tokens=Config.GENERATION_TOKENS,
        block_size=Config.BLOCK_SIZE
    )
    print(data.decode(generated_tokens.tolist()))
    print("----------------------\n")


if __name__ == "__main__":
    main()
