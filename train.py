import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
import torch
from torch.utils.data import Dataset
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

class PrivacyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class WeightedTrainer(Trainer):
    
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        if self.class_weights is not None:
            # Apply class weights to loss
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = outputs.get('loss')
        
        return (loss, outputs) if return_outputs else loss

def load_and_prepare_data():
    df = pd.read_csv('output_concerning_sentences_CLEANED.csv')
    texts = df['sentence'].tolist()
    labels = df['concern_type'].tolist()
    
    unique_labels = sorted(list(set(labels)))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    label_ids = [label2id[label] for label in labels]
    
    return texts, label_ids, label2id, id2label

def create_class_weights(labels):
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_labels,
        y=labels
    )
    
    return torch.FloatTensor(class_weights).to('cpu')

def train_improved_model():
    texts, label_ids, label2id, id2label = load_and_prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, label_ids, 
        test_size=0.2, 
        random_state=42,
        stratify=label_ids
    )
    
    model_name = 'distilbert-base-uncased'
    device = torch.device('cpu')
    
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    
    model = model.to(device)
    
    train_dataset = PrivacyDataset(X_train, y_train, tokenizer)
    test_dataset = PrivacyDataset(X_test, y_test, tokenizer)
    
    class_weights = create_class_weights(y_train)
    
    # Training configuration - keep it simple and stable
    training_args = TrainingArguments(
        output_dir='./privacy_bert_model',
        num_train_epochs=3,                    # Train for 3 rounds
        per_device_train_batch_size=4,         # Process 4 samples at a time
        learning_rate=3e-5,                    # How fast the model learns
        save_strategy="epoch",                 # Save after each training round
        evaluation_strategy="epoch",           # Check progress after each round
        load_best_model_at_end=True,          # Keep the best version
        seed=42,                              # For reproducible results
        no_cuda=True,                         # Use CPU only
    )
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_weighted = f1_score(labels, predictions, average='weighted')
        
        return {
            'f1': f1_macro,
            'f1_weighted': f1_weighted,
            'accuracy': (predictions == labels).mean()
        }
    
    class_weights = class_weights.to(device)
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    trainer.train()
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    
    model_save_path = './privacy_bert_model'
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    with open(os.path.join(model_save_path, 'label_mappings.json'), 'w') as f:
        json.dump({'label2id': label2id, 'id2label': id2label}, f, indent=2)
    
    return trainer, test_results

if __name__ == "__main__":
    train_improved_model()