from src.utils import file_util

from typing import Union

import json
import tensorflow as tf

logger = tf.get_logger()

__all__ = [
    "DecoderConfig",
    "DatasetConfig",
    "DataConfig",
    "LearningConfig",
    "Config",
]

class SpeechConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.sample_rate: str = config.get("sample_rate", 16000)
        self.frame_ms: int = config.get("frame_ms", 25)
        self.stride_ms: int = config.get("stride_ms", 10)
        self.num_feature_bins: int = config.get("num_feature_bins", 80)
        self.feature_type: str = config.get("feature_type", "mfcc")
        self.preemphasis: float = config.get("preemphasis", 0.97)
        self.pad_end: bool = config.get("pad_end", False)
        self.lower_edge_hertz: float = config.get("lower_edge_hertz", 0.0)
        self.upper_edge_hertz: float = config.get("upper_edge_hertz", 8000.0)
        self.output_floor: float = config.get("output_floor", 1e-9)
        self.log_base: str = config.get("log_base", "10")
        self.normalize_signal: bool = config.get("normalize_signal", True)
        self.normalize_zscore: bool = config.get("normalize_zscore",False)
        self.normalize_min_max: bool = config.get("normalize_min_max", False)
        self.padding: float = config.get("padding", 0.0)
        for k, v in config.items():
            setattr(self, k, v)

class DecoderConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.type: str = config.get("type", "wordpiece")

        self.blank_index: int = config.get("blank_index", 0)
        self.pad_token: str = config.get("pad_token", "<pad>")
        self.pad_index: int = config.get("pad_index", -1)
        self.unknown_token: str = config.get("unknown_token", "<unk>")
        self.unknown_index: int = config.get("unknown_index", 0)
        self.bos_token: str = config.get("bos_token", "<bos>")
        self.bos_index: int = config.get("bos_index", -1)
        self.eos_token: str = config.get("eos_token", "<eos>")
        self.eos_index: int = config.get("eos_index", -1)

        self.beam_width: int = config.get("beam_width", 0)
        self.norm_score: bool = config.get("norm_score", True)
        self.lm_config: dict = config.get("lm_config", {})

        self.model_type: str = config.get("model_type", "unigram")
        self.vocabulary: str = file_util.preprocess_paths(config.get("vocabulary", None))
        self.vocab_size: int = config.get("vocab_size", 1000)
        self.max_token_length: int = config.get("max_token_length", 50)
        self.max_unique_chars: int = config.get("max_unique_chars", None)
        self.num_iterations: int = config.get("num_iterations", 4)
        self.reserved_tokens: list = config.get("reserved_tokens", None)
        self.normalization_form: str = config.get("normalization_form", "NFKC")
        self.keep_whitespace: bool = config.get("keep_whitespace", False)
        self.max_sentence_length: int = config.get("max_sentence_length", 1048576)  # bytes
        self.max_sentencepiece_length: int = config.get("max_sentencepiece_length", 16)  # bytes
        self.character_coverage: float = config.get("character_coverage", 1.0)  # 0.9995 for languages with rich character, else 1.0

        self.train_files = config.get("train_files", [])
        self.eval_files = config.get("eval_files", [])

        for k, v in config.items():
            setattr(self, k, v)


class DatasetConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.enabled: bool = config.get("enabled", True)
        self.stage: str = config.get("stage", None)
        self.data_paths = config.get("data_paths", None)
        self.shuffle: bool = config.get("shuffle", False)
        self.cache: bool = config.get("cache", False)
        self.drop_remainder: bool = config.get("drop_remainder", True)
        self.buffer_size: int = config.get("buffer_size", 1000)
        self.metadata: str = config.get("metadata", None)
        self.sample_rate: int = config.get("sample_rate", 16000)
        for k, v in config.items():
            setattr(self, k, v)

class DataConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.train_dataset_config = DatasetConfig(config.get("train_dataset_config", {}))
        self.eval_dataset_config = DatasetConfig(config.get("eval_dataset_config", {}))
        self.test_dataset_configs = DatasetConfig(config.get("test_dataset_configs", {}))

class LearningConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.pretrained = file_util.preprocess_paths(config.get("pretrained", None))
        self.optimizer_config: dict = config.get("optimizer_config", {})
        self.gradn_config = config.get("gradn_config", None)
        self.ga_steps: int = config.get("ga_steps", None)
        self.running_config = config.get("running_config", {})
        # self.callbacks: list = config.get("callbacks", [])
        for k, v in config.items():
            setattr(self, k, v)

class RunningConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.batch_size: int = config.get("batch_size", 32)
        self.num_epochs: int = config.get("num_epochs", 10)
        for k, v in config.items():
            setattr(self, k, v)

class Config:
    def __init__(self, config: dict, training=True, **kwargs):
        self.speech_config = SpeechConfig(config.get("speech_config", {}))
        self.model_config: dict = config.get("model_config", {})
        self.decoder_config = DecoderConfig(config.get("decoder_config", {}))
        self.data_config = DataConfig(config.get("data_config", {}))
        self.learning_config = LearningConfig(config.get("learning_config", {})) if training else None
        for k, v in config.items():
            setattr(self, k, v)
        logger.info(str(self))

    def __str__(self) -> str:
        def default(x):
            try:
                return {k: v for k, v in vars(x).items() if not str(k).startswith("_")}
            except:  # pylint: disable=bare-except
                return str(x)

        return json.dumps(vars(self), indent=2, default=default)
