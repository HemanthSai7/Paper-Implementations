from src.dataset import get, get_shape
from src.helpers import prepare_featurizers
from src.model import Moonshine

from dotenv import load_dotenv
from omegaconf import DictConfig

import os
import hydra
import tensorflow as tf

load_dotenv()
logger = tf.get_logger()


@hydra.main(config_path="config", config_name="config")
def main(
    config: DictConfig,
    batch_size: int = 2,
    spx: int = None,
):
    
    speech_featurizer, tokenizer = prepare_featurizers(config)

    train_dataset = get(
        speech_featurizer=speech_featurizer,
        tokenizer=tokenizer,
        dataset_config=config.learning_config.train_dataset_config,
    )

    valid_dataset = get(
        speech_featurizer=speech_featurizer,
        tokenizer=tokenizer,
        dataset_config=config.learning_config.eval_dataset_config,
    )

    shapes = get_shape(
        config,
        train_dataset,
        valid_dataset,
        batch_size=config.learning_config.running_config.batch_size,
    )
    
    train_data_loader = train_dataset.create(batch_size=batch_size)
    valid_data_loader = valid_dataset.create(batch_size=batch_size)

    for data in valid_data_loader:
        print(data[0]["inputs"][0].shape)
        break

    try:
        moonshine = Moonshine(**config.model_config, vocab_size=tokenizer.vocab_size)
        moonshine.make(**shapes)

        if config.learning_config.pretrained:
            moonshine.load_weights(config.learning_config.pretrained, by_name=True, skip_mismatch=True)
        moonshine.summary(expand_nested=False)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0001,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9,
        )
        moonshine.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["loss"],
        )
        # moonshine.fit(
        #     train_data_loader,
        #     validation_data=valid_data_loader,
        #     epochs=1,
        #     steps_per_epoch=10,
        #     validation_steps=5,
        # )
    except Exception as e:
        logger.error(f"Error during model call: {e}")
        raise

if __name__ == "__main__":
    main()
