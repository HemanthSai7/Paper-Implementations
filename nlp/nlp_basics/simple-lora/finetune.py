import os
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass

import torch
import torch.nn as nn
from datasets import load_dataset
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from safetensors.torch import save_file
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

from lora import LoraConfig, LoRAModel

import warnings
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.multiprocessing.set_sharing_strategy("file_system")

@dataclass
class Config:
    num_epochs = 3
    batch_size = 128
    learning_rate = 1e-4
    num_workers = 8
    gradient_checkpointing = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # LoRA Config
    use_lora = True
    train_head_only = False
    target_modules = ["query", "key", "value", "dense", "word_embeddings"]
    exclude_modules = ["classifier"]
    rank = 8
    lora_alpha = 8
    use_rslora = True
    bias = "lora_only"
    lora_dropout = 0.1


print("Loading dataset...")
dataset = load_dataset("imdb")
labels = dataset["train"].features["label"].names

print("Tokenizing...")
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
dataset = dataset.map(lambda e: tokenizer(e["text"], truncation=True), batched=True, remove_columns="text")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, max_length=512)

trainloader = DataLoader(dataset["train"], batch_size=Config.batch_size, collate_fn=data_collator, 
                        shuffle=True, num_workers=Config.num_workers)
testloader = DataLoader(dataset["test"], batch_size=Config.batch_size, collate_fn=data_collator, 
                       shuffle=False, num_workers=Config.num_workers)

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-base", num_labels=len(labels))

if Config.gradient_checkpointing:
    model.gradient_checkpointing_enable()

if Config.use_lora:
    print("Converting to LoRA...")
    lora_config = LoraConfig(
        rank=Config.rank, 
        target_modules=Config.target_modules, 
        exclude_modules=Config.exclude_modules, 
        lora_alpha=Config.lora_alpha, 
        lora_dropout=Config.lora_dropout, 
        bias=Config.bias, 
        use_rslora=Config.use_rslora
    )
    model = LoRAModel(model, lora_config).to(Config.device)
else:
    model = model.to(Config.device)

accuracy_fn = Accuracy(task="multiclass", num_classes=len(labels)).to(Config.device)
params_to_train = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.AdamW(params_to_train, lr=Config.learning_rate)

print(f"\nStarting training on {Config.device}...")
print(f"Epochs: {Config.num_epochs} | Batch Size: {Config.batch_size} | LR: {Config.learning_rate}")
print("="*80)

for epoch in range(Config.num_epochs):
    model.train()
    train_loss = []
    train_acc = []
    accuracy_fn.reset()
    
    pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{Config.num_epochs} [Train]", 
                bar_format='{l_bar}{bar:30}{r_bar}')
    
    for batch in pbar:
        batch = {k: v.to(Config.device) for k, v in batch.items()}
        
        pred = model(**batch)
        loss = pred["loss"]
        
        predicted = pred["logits"].argmax(axis=1)
        acc = accuracy_fn(predicted, batch["labels"])
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        train_loss.append(loss.item())
        train_acc.append(acc.item())
        
        pbar.set_postfix({
            'loss': f'{np.mean(train_loss[-50:]):.4f}',
            'acc': f'{np.mean(train_acc[-50:]):.4f}'
        })
    
    model.eval()
    test_loss = []
    test_acc = []
    accuracy_fn.reset()
    
    pbar = tqdm(testloader, desc=f"Epoch {epoch+1}/{Config.num_epochs} [Eval]", 
                bar_format='{l_bar}{bar:30}{r_bar}')
    
    with torch.no_grad():
        for batch in pbar:
            batch = {k: v.to(Config.device) for k, v in batch.items()}
            
            pred = model(**batch)
            loss = pred["loss"]
            
            predicted = pred["logits"].argmax(axis=1)
            accuracy = accuracy_fn(predicted, batch["labels"])
            
            test_loss.append(loss.item())
            test_acc.append(accuracy.item())
            
            pbar.set_postfix({
                'loss': f'{np.mean(test_loss):.4f}',
                'acc': f'{np.mean(test_acc):.4f}'
            })
    
    epoch_train_loss = np.mean(train_loss)
    epoch_test_loss = np.mean(test_loss)
    epoch_train_acc = np.mean(train_acc)
    epoch_test_acc = np.mean(test_acc)
    
    print(f"\nEpoch {epoch+1}/{Config.num_epochs} Summary:")
    print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
    print(f"  Test Loss:  {epoch_test_loss:.4f} | Test Acc:  {epoch_test_acc:.4f}")
    print("="*80)

print("\nTraining Complete!")

save_dir = os.getcwd()
if Config.use_lora:
    print("Saving LoRA model...")
    model.save_model(os.path.join(save_dir, "imdb.safetensors"))
    model.save_model(os.path.join(save_dir, "imdb_merged.pt"), merge_weights=True)
elif Config.train_head_only:
    print("Saving head-only model...")
    save_file(model.state_dict(), os.path.join(save_dir, "imdb_head_only.safetensors"))
else:
    print("Saving base model...")
    save_file(model.state_dict(), os.path.join(save_dir, "imdb.safetensors"))

print("Done!")