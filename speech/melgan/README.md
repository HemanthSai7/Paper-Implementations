# MelGAN Implementation in TensorFlow

This is a complete implementation of **MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis** in TensorFlow 2.x.

MelGAN is a non-autoregressive, feed-forward convolutional neural network architecture that generates high-quality audio waveforms from mel-spectrograms using a GAN setup. It's significantly faster than autoregressive models while maintaining high audio quality.

## 📁 Files Overview

- **`melgan_tensorflow.py`** - Complete MelGAN implementation
- **`melgan_example.py`** - Training and inference examples  
- **`melgan_config.yaml`** - Configuration file with hyperparameters
- **`melgan-readme.md`** - This documentation

## 🏗️ Architecture Overview

### Generator Architecture
- **Input**: Mel-spectrograms (80 channels by default)
- **Output**: Raw audio waveforms  
- **Upsampling Factor**: 256x (8×8×2×2)
- **Key Components**:
  - Weight normalization on all convolutional layers
  - Transposed convolutions for upsampling
  - Residual blocks with dilated convolutions (dilations: 1, 3, 9)
  - LeakyReLU activations
  - Tanh output activation

### Discriminator Architecture
- **Multi-scale discriminator** with 3 discriminators
- **Window-based objective** for patch-based discrimination
- **Different temporal resolutions**: 1x, 2x, 4x downsampling
- **Weight normalization** on convolutional layers
- **Feature matching loss** for stable training

### Key Features
- **Weight Normalization**: Custom implementation for improved training stability
- **Residual Connections**: With dilated convolutions for large receptive fields
- **Feature Matching Loss**: Matches intermediate discriminator features
- **Multi-scale Discrimination**: Operates on different audio resolutions

## 🚀 Quick Start

### Installation

```bash
pip install tensorflow>=2.8.0 numpy
# For audio processing (optional):
# pip install librosa soundfile
```

### Basic Usage

```python
from melgan_tensorflow import MelGANGenerator, MelGANMultiScaleDiscriminator
import tensorflow as tf

# Create models
generator = MelGANGenerator()
discriminator = MelGANMultiScaleDiscriminator()

# Generate audio from mel-spectrogram
mel_spectrogram = tf.random.uniform([1, 100, 80])  # [batch, time, mel_channels]
generated_audio = generator(mel_spectrogram)

print(f"Input shape: {mel_spectrogram.shape}")   # (1, 100, 80)
print(f"Output shape: {generated_audio.shape}")  # (1, 25600, 1)
```

## 📚 Training

### Prepare Data

```python
# Example data preprocessing (you'll need to implement based on your dataset)
import librosa
import numpy as np

def preprocess_audio(audio_path):
    # Load audio
    audio, sr = librosa.load(audio_path, sr=22050)
    
    # Compute mel-spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=1024, hop_length=256, 
        win_length=1024, n_mels=80, fmin=80, fmax=7600
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Normalize
    mel_normalized = (mel_db + 80) / 80
    
    return mel_normalized.T, audio
```

### Training Script

```python
from melgan_tensorflow import *
import tensorflow as tf

# Configuration
config = MelGANConfig()  # From melgan_example.py

# Create models
generator = MelGANGenerator()
discriminator = MelGANMultiScaleDiscriminator()

# Setup training
gen_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=0.5, beta_2=0.9)
disc_optimizer = tf.keras.optimizers.Adam(1e-6, beta_1=0.5, beta_2=0.9)
losses = MelGANLosses()

# Training loop (simplified)
for epoch in range(num_epochs):
    for mel_batch, audio_batch in train_dataset:
        gen_loss, disc_loss = train_step(
            generator, discriminator, gen_optimizer, disc_optimizer,
            mel_batch, audio_batch, losses
        )
```

## 🎯 Model Parameters

### Default Configuration

| Component | Parameter | Value |
|-----------|-----------|-------|
| **Audio** | Sample Rate | 22,050 Hz |
| | Hop Length | 256 |
| | Mel Channels | 80 |
| **Generator** | Initial Filters | 512 |
| | Upsample Scales | [8, 8, 2, 2] |
| | Dilations | [1, 3, 9] |
| **Discriminator** | Base Filters | 16 |
| | Num Layers | 4 |
| | Kernel Size | 15 |
| **Training** | Gen Learning Rate | 1e-5 |
| | Disc Learning Rate | 1e-6 |
| | Feature Match λ | 10.0 |

### Memory and Performance

- **Generator Parameters**: ~4.0M
- **Discriminator Parameters**: ~12.6M
- **Inference Speed**: Real-time capable (>100x faster than real-time on GPU)
- **Training**: Stable with feature matching loss

## 🔧 Customization

### Custom Generator

```python
generator = MelGANGenerator(
    mel_channels=80,           # Input mel channels
    out_channels=1,            # Output audio channels
    upsample_scales=[8,8,2,2], # Upsampling stages
    filters=512,               # Initial filter count
    dilations=[1,3,9]          # Residual block dilations
)
```

### Custom Discriminator

```python
discriminator = MelGANMultiScaleDiscriminator(
    num_discriminators=3       # Number of scales
)
```

### Custom Loss Function

```python
losses = MelGANLosses(
    feature_match_lambda=10.0  # Feature matching weight
)
```

## 📊 Loss Functions

### Generator Loss
- **Adversarial Loss**: MSE loss to fool discriminator
- **Feature Matching Loss**: L1 loss between discriminator features
- **Total**: `L_adv + λ * L_fm` (λ = 10.0)

### Discriminator Loss  
- **Real Loss**: MSE loss for real samples (target = 1)
- **Fake Loss**: MSE loss for fake samples (target = 0)
- **Total**: `(L_real + L_fake) / 2`

## 🎵 Audio Quality

### Objective Metrics
- **Mean Opinion Score (MOS)**: 4.06 (reported in paper)
- **Real-time Factor**: 0.03 on CPU, >100x on GPU
- **Parameters**: 4M (generator) vs 28M (WaveNet)

### Training Tips
1. **Stable Training**: Use feature matching loss for stability
2. **Learning Rates**: Generator LR > Discriminator LR
3. **Batch Size**: Use larger batches (16-32) for stable gradients
4. **Warm-up**: Pre-train generator with reconstruction loss

## 🛠️ Implementation Details

### Weight Normalization
- Custom implementation using `tf.nn.l2_normalize`
- Applied to all convolutional layers except discriminator's first layer
- Separates weight magnitude (g) from direction (v)

### Residual Blocks
- LeakyReLU → Conv1D (dilated) → LeakyReLU → Conv1D (1x1)
- Skip connection with optional channel matching
- Exponentially increasing receptive field

### Multi-scale Discrimination
- 3 discriminators at different temporal resolutions
- Average pooling for downsampling (factors: 1x, 2x, 4x)
- Feature matching across all scales

## 📝 Paper Reference

```bibtex
@article{kumar2019melgan,
  title={MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis},
  author={Kumar, Kundan and Kumar, Rithesh and de Boissiere, Thibault and Gestin, Lucas and Teoh, Wei Zhen and Sotelo, Jose and de Brebisson, Alexandre and Bengio, Yoshua and Courville, Aaron},
  journal={Advances in Neural Information Processing Systems},
  volume={32},
  year={2019}
}
```

## 🤝 Contributing

Feel free to:
- Report issues or bugs
- Suggest improvements
- Add new features
- Optimize performance

## 📄 License

This implementation is provided for educational and research purposes. Please refer to the original MelGAN paper and TensorFlow license for usage guidelines.

## 🔗 Related Projects

- **Original MelGAN**: https://github.com/descriptinc/melgan-neurips  
- **TensorFlowTTS**: https://github.com/TensorSpeech/TensorFlowTTS
- **Multi-band MelGAN**: Improved version with sub-band processing

---

**Happy audio generation with MelGAN! 🎵**