from src.dataset import get, get_shape
from src.helpers import prepare_featurizers, prepare_training_datasets, prepare_training_dataloaders
from src.model.melgan import MelGAN
from src.model.encoder import MelGANGenerator
from src.model.decoder import MelGANMultiScaleDiscriminator
from src.configs import Config
from src.utils import env_util

from omegaconf import DictConfig, OmegaConf
from IPython.display import Audio

import os
import jiwer
import hydra
import tensorflow as tf

logger = tf.get_logger()


@hydra.main(config_path="config", config_name="config")
def main(
    config: DictConfig,
    batch_size: int = None,
    spx: int = 1,
):
    
    tf.keras.backend.clear_session()
    env_util.setup_seed()

    config = Config(OmegaConf.to_container(config, resolve=True), training=True)
    strategy = env_util.setup_strategy(config.learning_config["running_config"]["devices"])
    batch_size = batch_size or config.learning_config["running_config"]["batch_size"]

    speech_featurizer = prepare_featurizers(config)

    train_dataset, valid_dataset = prepare_training_datasets(
        config,
        speech_featurizer=speech_featurizer,
    )

    shapes = get_shape()
    print("Shapes:", shapes)

    train_data_loader, valid_data_loader, global_batch_size = prepare_training_dataloaders(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        strategy=strategy,
        global_batch_size=batch_size,
        shapes=shapes,
    )

    # for batch in train_data_loader:
    #     print(batch)
    #     break

    print(config.model_config)

    with strategy.scope():
        generator = MelGANGenerator(**config.model_config["generator"])
        generator.build(input_shape=[None, None, 80])
        generator.summary(expand_nested=True)

        discriminator = MelGANMultiScaleDiscriminator(**config.model_config["discriminator"])
        discriminator.build(input_shape=[None, None, 1])
        discriminator.summary(expand_nested=True)

        model = MelGAN(generator=generator, discriminator=discriminator)

        if config.learning_config["pretrained"]:
            model.load_weights(config.learning_config["pretrained"], by_name=True)
        model.compile(
            generator_optimizer=tf.keras.optimizers.get(config.learning_config["generator_optimizer_config"]),
            discriminator_optimizer=tf.keras.optimizers.get(config.learning_config["discriminator_optimizer_config"]),
            run_eagerly=False,
        )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(**config.learning_config["running_config"]["checkpoint"], verbose=1),
        tf.keras.callbacks.BackupAndRestore(config.learning_config["running_config"]["states_dir"]),
        tf.keras.callbacks.TensorBoard(**config.learning_config["running_config"]["tensorboard"]),
        tf.keras.callbacks.CSVLogger(config.learning_config["running_config"]["csv_logger"]),
    ]

    # model.fit(
    #     train_data_loader,
    #     epochs=config.learning_config["running_config"]["num_epochs"],
    #     validation_data=valid_data_loader,
    #     steps_per_epoch=train_dataset.total_steps,
    #     validation_steps=valid_dataset.total_steps if valid_data_loader else None,
    #     callbacks=callbacks,
    #     verbose=1,
    # )

if __name__ == "__main__":
    main()