import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lora import LoraConfig, LoRAModel
from safetensors.torch import load_file

import warnings
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SentimentClassifier:
    def __init__(self, model_path, use_lora=True, device=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_lora = use_lora
        self.labels = ["negative", "positive"]

        self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")

        base_model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-base", num_labels=len(self.labels))

        if use_lora:
            if model_path.endswith(".safetensors"):
                lora_config = LoraConfig(
                    rank=8,
                    target_modules=["query", "key", "value", "dense", "word_embeddings"],
                    exclude_modules=["classifier"],
                    lora_alpha=8,
                    lora_dropout=0.1,
                    bias="lora_only",
                    use_rslora=True
                )
                self.model = LoRAModel(base_model, lora_config)
                
                state_dict = load_file(model_path)
                self.model.load_state_dict(state_dict, strict=False)
            elif model_path.endswith(".pt"):
                state_dict = torch.load(model_path, map_location=self.device)
                base_model.load_state_dict(state_dict)
                self.model = base_model
        else:
            state_dict = load_file(model_path)
            base_model.load_state_dict(state_dict)
            self.model = base_model
        
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!\n")

    def predict(self, text, return_probabilities=False):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_class].item()
        
        predicted_label = self.labels[predicted_class]

        if return_probabilities:
            return {
                "label": predicted_label,
                "confidence": confidence,
                "probabilities": {
                    label: probs[0][i].item() 
                    for i, label in enumerate(self.labels)
                }
            }
        else:
            return predicted_label
        
def main():
    model_path = "imdb_merged.pt"

    use_lora = model_path.endswith(".safetensors")
    classifier = SentimentClassifier(model_path, use_lora=use_lora)

    examples = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Terrible film. Complete waste of time and money.",
        "Not bad, but nothing special either."
    ]
    
    for text in examples:
        result = classifier.predict(text, return_probabilities=True)
        print(f"\nText: {text}")
        print(f"Prediction: {result['label'].upper()}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Probabilities - Negative: {result['probabilities']['negative']:.4f} | "
              f"Positive: {result['probabilities']['positive']:.4f}")
        
if __name__ == "__main__":
    main()

