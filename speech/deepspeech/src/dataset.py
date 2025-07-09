from dataclasses import dataclass, asdict

from src.schema import schemas
from src.configs import Config, DatasetConfig
from src.featurizers import (
    CharacterTokenizer, 
    SpeechFeaturizer,
)
from src.utils import (
    data_util,
    file_util,
    math_util,
)

import os
import json
import tqdm
import numpy as np
import tensorflow as tf

logger = tf.get_logger()

__all__ = [
    "ASR_DATASET_TYPES",
    "get_global_shape",
    "ASRDataset",
    "ASRSliceDataset",
]

@dataclass
class ASR_DATASET_TYPES:
    SLICE: str = "slice"
    GENERATOR: str = "generator"

def get(
    tokenizer: CharacterTokenizer,
    speech_featurizer: SpeechFeaturizer,
    dataset_config: DatasetConfig,
    dataset_type: str,
):
    if dataset_type == ASR_DATASET_TYPES.SLICE:
        return ASRSliceDataset(
            tokenizer=tokenizer, 
            stage=dataset_config["stage"],
            data_paths=list(dataset_config["data_paths"]),
            speech_featurizer=speech_featurizer,
            )
    if dataset_type == ASR_DATASET_TYPES.GENERATOR:
        return ASRDataset(tokenizer=tokenizer, **vars(dataset_config))
    raise ValueError(f"dataset_type must in {asdict(ASR_DATASET_TYPES()).values()}")

def get_global_shape(
    config: Config,
    *datasets,
):

    max_input_length, max_label_length = None, None

    input_shape = [max_input_length]
    prediction_shape = [None]
    label_shape = [max_label_length]

    feature_type = config.speech_config["feature_type"]
    if feature_type == "waveform":
        input_shape.extend([1])
    elif feature_type in ["spectrogram", "log_mel_spectrogram", "mfcc"]:
        input_shape.extend([config.speech_config["num_feature_bins"], 1])
    padded_shapes = (
        schemas.TrainInput(
            inputs=tf.TensorShape(input_shape),
            inputs_length=tf.TensorShape([]),
            predictions=tf.TensorShape(prediction_shape),
            predictions_length=tf.TensorShape([]),
        ),
        schemas.TrainLabel(
            labels=tf.TensorShape(label_shape),
            labels_length=tf.TensorShape([]),
        ),
    )

    return dict(
        input_shape=input_shape,
        prediction_shape=prediction_shape,
        label_shape=label_shape,
        padded_shapes=padded_shapes,
    )


BUFFER_SIZE = 100
TFRECORD_BUFFER_SIZE = 32 * 1024 * 1024
TFRECORD_SHARDS = 16
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
    

class ASRDataset(BaseDataset):
    def __init__(
        self,
        stage: str,
        tokenizer: CharacterTokenizer,
        speech_featurizer: SpeechFeaturizer,
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
        name: str = "asr_dataset",
        **kwargs,
    ):
        super(ASRDataset, self).__init__(
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
        self.tokenizer = tokenizer
        self.speech_featurizer = speech_featurizer
        self.training = training
        self.max_input_length = None
        self.max_label_length = None
        self.load_metadata()

    # METADATA

    def compute_metadata(self):
        self.max_input_length = 0 if self.max_input_length is None else self.max_input_length
        self.max_label_length = 0 if self.max_label_length is None else self.max_label_length
        if self.max_input_length > 0 and self.max_label_length > 0:
            return
        self.read_entries()
        # PATH, DURATION, TRANSCRIPT
        for _, duration, transcript in tqdm.tqdm(self.entries, desc=f"Computing metadata for entries is {self.stage} dataset"):
            input_length = math_util.get_nsamples(duration, self.sample_rate)
            label = self.tokenizer.tokenize(transcript).numpy()
            label_length = len(label)
            self.max_input_length = max(self.max_input_length, input_length)
            self.max_label_length = max(self.max_label_length, label_length)

    def save_metadata(self):
        if self.metadata is None:
            return
        self.metadata = file_util.preprocess_paths(self.metadata)
        if tf.io.gfile.exists(self.metadata):
            with tf.io.gfile.GFile(self.metadata, "r") as f:
                try:
                    content = json.loads(f.read())
                except json.JSONDecodeError as e:
                    raise ValueError(f"File {self.metadata} is currently not in json format. Please update the file") from e
        else:
            content = {}
        content[self.stage] = dict(
            max_input_length=self.max_input_length,
            max_label_length=self.max_label_length,
            num_entries = self.total_steps
        )

    def load_metadata(self):
        if self.metadata is None:
            return
        if not self.enabled:
            return
        content = None
        self.metadata = file_util.preprocess_paths(self.metadata)
        if tf.io.gfile.exists(self.metadata):
            logger.info(f"Loading metadata from {self.metadata} ...")
            with tf.io.gfile.GFile(self.metadata, "r") as f:
                try:
                    content = json.loads(f.read()).get(self.stage, {})
                except json.JSONDecodeError as e:
                    raise ValueError(f"File {self.metadata} must be in json format") from e
        if not content:
            return
        self.max_input_length = content.get("max_input_length")
        self.max_label_length = content.get("max_label_length")
        self.total_steps = int(content.get("num_entries", 0))
        self.num_entries = self.total_steps

    def update_metadata(self):
        self.load_metadata()
        self.compute_metadata()
        self.save_metadata()

    # ENTRIES

    def read_entries(self):
        if hasattr(self, "entries") and len(self.entries) > 0:
            return
        self.data_paths = file_util.preprocess_paths(self.data_paths, enabled=self.enabled, check_exists=True)
        for file_path in self.data_paths:
            logger.info(f"Reading {file_path} ...")
            with tf.io.gfile.GFile(file_path, "r") as f:
                for line in f.read().splitlines()[1:]:  # Skip the header of tsv file
                    self.entries.append(line.split("\t", 2))  # The files is "\t" seperated
        self.entries = np.array(self.entries)
        if self.shuffle:
            np.random.shuffle(self.entries)  # Mix transcripts.tsv
        self.total_steps = len(self.entries)
        self.num_entries = self.total_steps

    # LOAD AND PREPROCESS

    def generator(self):
        # PATH, DURATION, TRANSCRIPT
        for path, _, transcript in self.entries:
            audio = data_util.load_and_convert_to_wav(path, sample_rate=self.sample_rate).numpy()
            yield bytes(path, "utf-8"), audio, bytes(transcript, "utf-8")

    def _process_item(self, path: tf.Tensor, audio: tf.Tensor, transcript: tf.Tensor):
        logger.info(f"Preprocessing using TensorFlow ...")
        inputs = data_util.read_raw_audio(audio)
        inputs_length = tf.cast(tf.shape(inputs)[0], tf.int32)
        inputs, inputs_length = self.speech_featurizer((inputs, inputs_length), training=self.training)

        labels = self.tokenizer.tokenize(transcript)
        labels_length = tf.shape(labels, out_type=tf.int32)[0]

        predictions = self.tokenizer.prepand_blank(labels)
        predictions_length = tf.shape(predictions, out_type=tf.int32)[0]

        return path, inputs, inputs_length, labels, labels_length, predictions, predictions_length
    
    def parse(self, path: tf.Tensor, audio: tf.Tensor, transcript: tf.Tensor):
        (
            _,
            inputs,
            inputs_length,
            labels,
            labels_length,
            predictions,
            predictions_length,
        ) = self._process_item(path=path, audio=audio, transcript=transcript)
        return (
            schemas.TrainInput(
                inputs=inputs, 
                inputs_length=inputs_length, 
                predictions=predictions, 
                predictions_length=predictions_length
            ),
            schemas.TrainLabel(
                labels=labels, 
                labels_length=labels_length
            ),
        )
    
    # CREATION

    def process(self, dataset: tf.data.Dataset, batch_size: int, shapes=None):
        if self.cache:
            dataset = dataset.cache()  # cache original (unchanged data)

        dataset = dataset.map(self.parse, num_parallel_calls=AUTOTUNE, deterministic=False)
        self.total_steps = math_util.get_num_batches(self.num_entries, batch_size, drop_remainders=self.drop_remainder)

        if self.shuffle:
            dataset = dataset.shuffle(self.buffer_size or self.num_entries, reshuffle_each_iteration=True)

        if self.indefinite and self.total_steps:
            dataset = dataset.repeat()

        # PADDED BATCH the dataset
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=shapes["padded_shapes"],
            padding_values=(
                schemas.TrainInput(inputs=0.0, inputs_length=0, predictions=self.tokenizer.blank, predictions_length=0),
                schemas.TrainLabel(labels=self.tokenizer.blank, labels_length=0),
            ),
            drop_remainder=self.drop_remainder,
        )

        # PREFETCH to improve speed of input length
        dataset = dataset.prefetch(AUTOTUNE)
        return dataset
    
    def create(self, batch_size: int, padded_shapes=None):
        if not self.enabled:
            return None
        self.read_entries()
        if not self.total_steps or self.total_steps == 0:
            return None
        dataset = tf.data.Dataset.from_generator(
            self.generator,
            output_types=(tf.string, tf.string, tf.string),
            output_shapes=(tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([])),
        )
        return self.process(dataset, batch_size, padded_shapes=padded_shapes)
    

class ASRSliceDataset(ASRDataset):

    @staticmethod
    def load(record: tf.Tensor):
        audio = tf.py_function(
            lambda path: data_util.load_and_convert_to_wav(path.numpy().decode("utf-8")).numpy(),
            inp=[record[0]],
            Tout=tf.string,
        )

        return record[0], audio, record[2]
    
    def create(self, batch_size: int, padded_shapes=None):
        if not self.enabled:
            return None
        self.read_entries()
        if not self.total_steps or self.total_steps == 0:
            return None
        
        dataset = tf.data.Dataset.from_tensor_slices(self.entries)
        options = tf.data.Options()
        options.deterministic = False
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset = dataset.with_options(options)
        dataset = dataset.map(self.load, num_parallel_calls=AUTOTUNE, deterministic=False)
        return self.process(dataset, batch_size, shapes=padded_shapes)