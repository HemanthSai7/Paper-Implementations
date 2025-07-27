from src.schemas import TrainInput, TargetLabels
from src.configs import DatasetConfig
from src.utils import (
    data_util,
    file_util,
    math_util,
)
from src.speech_featurizer import SpeechFeaturizer

import os
import numpy as np
import tensorflow as tf

logger = tf.get_logger()

def get(
    speech_featurizer: SpeechFeaturizer,
    dataset_config: DatasetConfig,
):
    return MelGANSliceDataset(
        speech_featurizer=speech_featurizer,
        stage=dataset_config["stage"],
        data_paths=list(dataset_config["data_paths"]),
    )

def get_shape():

    mel_input_shape = [None, 80]
    audio_target_shape = [None, 1]

    padded_shapes = (
        TrainInput(
            mel_spectrogram=tf.TensorShape(mel_input_shape),
        ),
        TargetLabels(
            audio_waveform=tf.TensorShape(audio_target_shape),
        )
    )

    return dict(
        mel_input_shape=mel_input_shape,
        audio_target_shape=audio_target_shape,
        padded_shapes=padded_shapes,
    )

BUFFER_SIZE = 100
AUTOTUNE = int(os.environ.get("AUTOTUNE", tf.data.AUTOTUNE))

class BaseDataset:
    def __init__(
        self,
        data_paths: list,
        cache: bool = False,
        shuffle: bool = False,
        buffer_size: int = BUFFER_SIZE,
        indefinite: bool = False,
        drop_remainder: bool = True,
        enabled: bool = True,
        metadata: str = None,
        sample_rate: int = 16000,
        stage: str = "train",
        name: str = "base_dataset",
        **kwargs,
    ):
        self.data_paths = data_paths or []
        if not isinstance(self.data_paths, list):
            raise ValueError("data_paths must be a list of string paths")
        self.cache = cache
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.stage = stage
        self.enabled = enabled
        self.drop_remainder = drop_remainder
        self.indefinite = indefinite
        self.total_steps = None
        self.metadata = metadata
        self.sample_rate = sample_rate
        self.name = name

    def parse(self, *args, **kwargs):
        raise NotImplementedError()
    
    def create(self, *args, **kwargs):
        raise NotImplementedError()
    
class MelGANDataset(BaseDataset):
    def __init__(
        self,
        speech_featurizer: SpeechFeaturizer,
        stage: str,
        data_paths: list,
        cache: bool = False,
        shuffle: bool = False,
        indefinite: bool = True,
        drop_remainder: bool = True,
        enabled: bool = True,
        metadata: str = None,
        buffer_size: int = BUFFER_SIZE,
        sample_rate: int = 16000,
        training=False,
        name: str = "melgan_dataset",
        **kwargs,
    ):
        super(MelGANDataset, self).__init__(
            data_paths=data_paths,
            cache=cache,
            shuffle=shuffle,
            buffer_size=buffer_size,
            indefinite=indefinite,
            drop_remainder=drop_remainder,
            enabled=enabled,
            metadata=metadata,
            sample_rate=sample_rate,
            stage=stage,
            name=name,
        )
        self.entries = []
        self.training = training
        self.speech_featurizer = speech_featurizer

    def read_entries(self):
        if hasattr(self, "entries") and len(self.entries) > 0:
            return
        self.data_paths = file_util.preprocess_paths(self.data_paths, enabled=self.enabled, check_exists=True)
        for file_path in self.data_paths:
            logger.info(f"Reading entries from {file_path}")
            with tf.io.gfile.GFile(file_path, "r") as f:
                for line in f.read().splitlines()[1:]:
                    self.entries.append(line)
        self.entries = np.array(self.entries)
        if self.shuffle:
            np.random.shuffle(self.entries)
        self.total_steps = len(self.entries)
        self.num_entries = self.total_steps
        logger.info(f"Total entries: {self.num_entries}")

    def _process_item(self, path: tf.Tensor, audio: tf.Tensor):
        with tf.device("/CPU:0"):
            audio_inputs = data_util.read_raw_audio(audio, sample_rate=self.sample_rate)
            
            mel_spectrogram_inputs = self.speech_featurizer(audio_inputs)
            audio_waveform_targets = tf.expand_dims(audio_inputs, axis=-1)

        return path, mel_spectrogram_inputs, audio_waveform_targets

    def parse(self, path: tf.Tensor, audio: tf.Tensor):
        (
            _,
            mel_spectrogram_inputs,
            audio_waveform_targets
        ) = self._process_item(path=path, audio=audio)

        return (
            TrainInput(
                mel_spectrogram=mel_spectrogram_inputs,
            ),
            TargetLabels(
                audio_waveform=audio_waveform_targets
            )
        )

    def process(self, dataset: tf.data.Dataset, batch_size: int, shapes=None):
        if self.cache:
            dataset = dataset.cache()

        dataset = dataset.map(self.parse, num_parallel_calls=AUTOTUNE, deterministic=False)
        self.total_steps = math_util.get_num_batches(self.num_entries, batch_size, drop_remainders=self.drop_remainder)

        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=self.buffer_size, reshuffle_each_iteration=True)

        if self.indefinite and self.total_steps:
            dataset = dataset.repeat()

        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=shapes["padded_shapes"],
            padding_values = (
                TrainInput(mel_spectrogram=0.0),
                TargetLabels(audio_waveform=0.0),
            ),
            drop_remainder=self.drop_remainder,
        )

        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset

class MelGANSliceDataset(MelGANDataset):

    @staticmethod
    def load(record: tf.Tensor):
        audio = tf.py_function(
            lambda path: data_util.load_and_convert_to_wav(path.numpy().decode("utf-8")).numpy(),
            inp = [record],
            Tout=tf.string,
        )
        return record, audio

    def create(self, batch_size: int, padded_shapes=None):
        if not self.enabled:
            return None
        self.read_entries()
        if not self.total_steps or self.total_steps == 0:
            return None
        
        dataset = tf.data.Dataset.from_tensor_slices(self.entries)
        # options = tf.data.Options()
        # options.deterministic = False
        # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        # dataset = dataset.with_options(options)
        dataset = dataset.map(self.load, num_parallel_calls=AUTOTUNE, deterministic=False)

        return self.process(dataset, batch_size=batch_size, shapes=padded_shapes)