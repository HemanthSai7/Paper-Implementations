# Moonshine model in Tensorflow from scratch implementation

Trained a small moonshine model coded and trained from scratch in Tensorflow 



[Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608)

## ModelArgs Hyperparameters

| Parameter               | Value                  | Description                                                                 |
|-------------------------|------------------------|-----------------------------------------------------------------------------|
| `batch_size`            | 2                     | The number of samples processed before the model is updated.                |
| `max_lr`                | null                  | Maximum learning rate.                                                      |
| `warmup steps`          | 10000                 | Warmup steps in LR scheduler                                                |
| `dropout`               | 0.1                    | Dropout rate for regularization.                                            |
| `epochs`                | 120                     | Number of training epochs.                                                  |
| `block_size`            | variable                     | Sequence length (number of tokens or time steps).                           |
| `tgt_vocab_size`        | 31     | Size of the target vocabulary.                                              |
| `embeddings_dims`       | 256                    | Dimensionality of token embeddings.                                         |
| `attn_dropout`          | null                    | Dropout rate for attention layers.                                          |
| `no_of_heads`           | 8                      | Number of attention heads in multi-head attention.                          |
| `no_of_decoder_layers`  | 8                      | Number of decoder layers in the model.                                      |
| `log_mel_features`      | 80                     | Number of Mel spectrogram features.                                         |
| `kernel_size`           | 9                      | Kernel size for convolutional layers.                                       |
| `stride`                | 2             | Stride for convolutional layers.                                            |
| `sr`                    | 16000                  | Sampling rate of the audio.                                                 |
| `device`                | `'cuda:0'`             | Device to run the model on (e.g., GPU).                                     |
| `SAMPLING_RATE`         | 16000                  | Sampling rate of the audio.                                                 |
| `N_MELS`                | 80                     | Number of Mel bins in the spectrogram.                                      |
| `WINDOW_DURATION`       | 0.025                  | Duration of the analysis window in seconds (25 ms).                         |
| `STRIDE_DURATION`       | 0.010                  | Stride between consecutive windows in seconds (10 ms).                      |
| `WINDOW`                | hann                   | FFT Window                                                                  |
| `max_t`                 | variable                    | Maximum time steps in the spectrogram.                                      |
| `n_channels`            | 1                     | Number of channels in the input spectrogram.                                |


### Dataset

[LibriSpeech](https://www.openslr.org/12/)


### Frameworks:
**Tensorflow**


### Epochs/Steps
Epochs (train) = 120

Val iterations = every epoch


### Loss Curves

![Train and Val loss curves](../../img/moonshine_loss.png)