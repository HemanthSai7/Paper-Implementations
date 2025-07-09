from src.utils import env_util
from src.configs import  Config
from src.helpers import (
    prepare_featurizers, 
    prepare_training_datasets, 
    prepare_training_dataloaders
)
from src.losses.ctc_loss import CTCLoss
from src.dataset import get_global_shape
from src.models import BaseModel, DeepSpeech2

from omegaconf import DictConfig, OmegaConf

import hydra
import tensorflow as tf

logger = tf.get_logger()


@hydra.main(config_path="config", config_name="config")
def main(
    config: DictConfig,
    batch_size: int = None,
):
    config = Config(OmegaConf.to_container(config, resolve=True), training=True)

    tf.keras.backend.clear_session()
    env_util.setup_seed()
    strategy = env_util.setup_strategy(config.learning_config["running_config"]["devices"])
    batch_size = batch_size or config.learning_config["running_config"]["batch_size"]

    speech_featurizer, tokenizer = prepare_featurizers(config)

    train_dataset, valid_dataset = prepare_training_datasets(
        config=config,
        speech_featurizer=speech_featurizer,
        tokenizer=tokenizer,
        dataset_type=config.learning_config["running_config"]["dataset_type"]
    )

    shapes = get_global_shape(
        config,
        train_dataset,
        valid_dataset,
    )

    print("Global shapes:", shapes)

    train_data_loader, valid_data_loader, global_batch_size = prepare_training_dataloaders(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        strategy=strategy,
        global_batch_size=batch_size,
        shapes=shapes,
    )

    for batch in train_data_loader:
        print("Batch audio input shape:", batch[0]["inputs"].shape)
        print("Batch audio length shape:", batch[0]["inputs_length"].shape)
        print("Batch text input shape:", batch[0]["predictions"].shape)
        print("Batch text length shape:", batch[0]["predictions_length"].shape)
        break


    with strategy.scope():
        model: BaseModel = DeepSpeech2(**config.model_config)
        model.make(**shapes, batch_size=global_batch_size)
        if config.learning_config["pretrained"]:
            model.load_weights(config.learning_config["pretrained"], by_name=True, skip_mismatch=False)
        model.compile(
            optimizer=tf.keras.optimizers.get(config.learning_config["optimizer_config"]),
            loss=CTCLoss(blank=0, name="ctc_loss"),
            run_eagerly=False,
        )
        model.summary(expand_nested=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(**config.learning_config["running_config"]["checkpoint"], verbose=1),
        tf.keras.callbacks.BackupAndRestore(config.learning_config["running_config"]["states_dir"]),
        tf.keras.callbacks.TensorBoard(**config.learning_config["running_config"]["tensorboard"]),
        tf.keras.callbacks.CSVLogger(config.learning_config["running_config"]["csv_logger"]),
    ]

    model.fit(
        train_data_loader,
        epochs=config.learning_config["running_config"]["num_epochs"],
        validation_data=valid_data_loader,
        steps_per_epoch=train_dataset.total_steps,
        validation_steps=valid_dataset.total_steps if valid_data_loader else None,
        callbacks=callbacks,
        verbose=1,
    )

if __name__ == "__main__":
    main()


    