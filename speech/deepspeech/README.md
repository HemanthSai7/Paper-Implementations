# DeepSpeech2 Model in TensorFlow

> **"Deep Speech 2: End-to-End Speech Recognition in English and Mandarin"**    
> [arXiv:1512.02595](https://arxiv.org/abs/1512.02595)

## 📝 Abstract
Implemented DeepSpeech2 in TensorFlow 2.x, faithfully reproducing the original architecture designed for scalable end-to-end speech recognition. This version is trained on the LibriSpeech corpus using log-Mel spectrogram features and optimized for robustness and clarity. The model employs convolutional and recurrent layers with optional row convolution for enhanced context modeling.

## 🎯 Objectives
- Implement the DeepSpeech2 model from scratch in TensorFlow.
- Train and evaluate it on LibriSpeech with character-level output.
- Enable optional row convolution for extended context modeling.
- Ensure configurable preprocessing and training options.

## 📦 Model Architecture
- The model uses the following structure:
- **Input**: Log-Mel Spectrograms (80 bins, 25ms window, 10ms stride)
- **Conv Layers**: Two Conv2D layers with stride and padding
- **RNN Layers**: 5 stacked unidirectional LSTM layers
- **Row Convolution**: (width = 5)
- **Fully Connected**: One dense layer before softmax
- **Output**: Character-level vocabulary + CTC blank token

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
| `no_of_encoder_layers`  | 8                      | Number of encoder layers in the model.                                      |
| `log_mel_features`      | 80                     | Number of Mel spectrogram features.                                         |
| `kernel_size`           | [ [ 11, 41 ], [ 11, 21 ] ]                      | Kernel size for convolutional layers.                                       |
| `stride`                | [[ 2, 2 ], [ 1, 2 ] ]          | Stride for convolutional layers.                                            |
| `rnn_layers`            | 5                      | Number of RNN layers in the model.                                         |
| `rnn_units`             | 512                    | Number of units in the RNN layers.                                          |
| `fc_units`              | 1024                   | Number of fully connected units in the model.                               |
| `sr`                    | 16000                  | Sampling rate of the audio.                                                 |
| `device`                | `'cuda:0'` 12 GB            | Device to run the model on (e.g., GPU).                                     |
| `SAMPLING_RATE`         | 16000                  | Sampling rate of the audio.                                                 |
| `WINDOW_DURATION`       | 0.025                  | Duration of the analysis window in seconds (25 ms).                         |
| `STRIDE_DURATION`       | 0.010                  | Stride between consecutive windows in seconds (10 ms).                      |
| `WINDOW`                | hann                   | FFT Window                                                                  |
| `max_t`                 | variable                    | Maximum time steps in the spectrogram.                                      |
| `n_channels`            | 1                     | Number of channels in the input spectrogram.                                |


## 📊 Dataset

- **LibriSpeech 1000h**
- Preprocessing pipeline includes resampling, normalization, and log-mel spectrogram conversion using Hann windowing.

## 🧠 Training Details

- **Framework**: TensorFlow 2.x  
- **Optimizer**: Adam with custom LR scheduler (Noam-style warmup)
- **Device**: Trained on GPU (`cuda:0`)  

---

```bibtex
@misc{amodei2015deepspeech2endtoend,
      title={Deep Speech 2: End-to-End Speech Recognition in English and Mandarin}, 
      author={Dario Amodei and Rishita Anubhai and Eric Battenberg and Carl Case and Jared Casper and Bryan Catanzaro and Jingdong Chen and Mike Chrzanowski and Adam Coates and Greg Diamos and Erich Elsen and Jesse Engel and Linxi Fan and Christopher Fougner and Tony Han and Awni Hannun and Billy Jun and Patrick LeGresley and Libby Lin and Sharan Narang and Andrew Ng and Sherjil Ozair and Ryan Prenger and Jonathan Raiman and Sanjeev Satheesh and David Seetapun and Shubho Sengupta and Yi Wang and Zhiqian Wang and Chong Wang and Bo Xiao and Dani Yogatama and Jun Zhan and Zhenyao Zhu},
      year={2015},
      eprint={1512.02595},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1512.02595}, 
}
```