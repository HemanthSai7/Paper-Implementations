# IMDB Sentiment Analysis with LoRA Fine-tuning

### LoRA Fine-tuning Performance

| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
|-------|-----------|-----------|-----------|----------|
| 1     | 0.2345    | 90.29%    | 0.1383    | **94.89%** |
| 2     | 0.1573    | 94.15%    | 0.1485    | 94.88%   |
| 3     | 0.1250    | 95.64%    | 0.1382    | **95.06%** |

**Final Test Accuracy: 95.06%**

### Key Metrics

- **Training Time**: ~69 minutes (3 epochs)
- **Peak Test Accuracy**: 95.06%
- **Trainable Parameters**: 2,410,442 (1.93% of total)
- **Total Parameters**: 124,647,170

## Model Comparison

### LoRA vs Full Fine-tuning

| Metric | Full Fine-tuning | LoRA Fine-tuning | Improvement |
|--------|-----------------|------------------|-------------|
| Trainable Parameters | 124.6M (100%) | 2.4M (1.93%) | **98.07% reduction** |
| Test Accuracy | ~95%* | 95.06% | Comparable |
| Training Time/Epoch | ~25-30 min* | ~23 min | **~20% faster** |
| Memory Usage | High | Low | **Significant reduction** |
| Model Size | ~500MB | ~10MB (adapters only) | **98% smaller** |

*Estimated based on typical RoBERTa fine-tuning benchmarks

## Architecture

### LoRA Configuration

```python
LoRA Config:
  - Rank (r): 8
  - Alpha: 8
  - Dropout: 0.1
  - Target modules: query, key, value, dense, word_embeddings
  - Exclude modules: classifier
  - RS-LoRA: Enabled
```

## 🚀 Usage

### Training

```bash
python finetune.py
```

### Inference

```bash
python inference_roberta.py
```

### Inference Samples on LoRA Finetuned Model

```bash
Text: This movie was absolutely fantastic! I loved every minute of it.
Prediction: POSITIVE
Confidence: 0.9952
Probabilities - Negative: 0.0048 | Positive: 0.9952

Text: Terrible film. Complete waste of time and money.
Prediction: NEGATIVE
Confidence: 0.9990
Probabilities - Negative: 0.9990 | Positive: 0.0010

Text: Not bad, but nothing special either.
Prediction: NEGATIVE
Confidence: 0.9604
Probabilities - Negative: 0.9604 | Positive: 0.0396
```

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

## 🎓 Citation

If you use this implementation, please cite the original LoRA paper:

```bibtex
@article{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Yelong, Shen and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal={arXiv preprint arXiv:2106.09685},
  year={2021}
}
```