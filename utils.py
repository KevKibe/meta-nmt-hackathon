from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoProcessor
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import numpy as np
import evaluate

def load_dataset(train_path, test_path):
  train_data = pd.read_csv(train_path)
  test_data = pd.read_csv(test_path)
  return train_data, test_data

def load_model(model_name, read_token):
  tokenizer = AutoTokenizer.from_pretrained(model_name, token=read_token)
  model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=read_token)
  processor = AutoProcessor.from_pretrained(model_name, token=read_token)
  return tokenizer, model, processor



class ForDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        input_ids = torch.tensor(self.inputs["input_ids"][index]).squeeze()
        target_ids = torch.tensor(self.targets["input_ids"][index]).squeeze()

        return {"input_ids": input_ids, "labels": target_ids}
    
def split(tokenizer, source, target):
    max = 512
    X_train, X_val, y_train, y_val = train_test_split(source, target, test_size=0.2)
    X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=max, return_tensors="pt")
    y_train_tokenized = tokenizer(y_train, padding=True, truncation=True, max_length=max, return_tensors="pt")
    X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=max, return_tensors="pt")
    y_val_tokenized = tokenizer(y_val, padding=True, truncation=True, max_length=max, return_tensors="pt")

    train_dataset = ForDataset(X_train_tokenized, y_train_tokenized)
    test_dataset = ForDataset(X_val_tokenized, y_val_tokenized)
    return train_dataset, test_dataset
