"""
Pet Adoption Speed Prediction using Transformer-based Text Representations
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class PetAdoptionDataset(Dataset):
    """Dataset for Pet Adoption with text metadata"""
    
    def __init__(self, dataframe, metadata_dir, tokenizer, max_length=128):
        self.data = dataframe.reset_index(drop=True)
        self.metadata_dir = metadata_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pet_id = row['PetID']
        
        # Extract text from metadata
        text = self.extract_text_from_metadata(pet_id)
        
        # Add description if available
        if pd.notna(row.get('Description')):
            text = f"{row['Description']} {text}"
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(row['AdoptionSpeed'], dtype=torch.long)
        }
    
    def extract_text_from_metadata(self, pet_id):
        """Extract text descriptions from metadata JSON files"""
        texts = []
        
        # Check for metadata files with this PetID
        for i in range(1, 20):  # Max 20 images per pet
            metadata_file = os.path.join(self.metadata_dir, f"{pet_id}-{i}.json")
            
            if not os.path.exists(metadata_file):
                break
                
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Extract label annotations (descriptions)
                if 'labelAnnotations' in metadata:
                    descriptions = [label['description'] for label in metadata['labelAnnotations'][:5]]
                    texts.extend(descriptions)
                
            except Exception as e:
                continue
        
        return ' '.join(texts) if texts else 'no description'


class TransformerPetClassifier(nn.Module):
    """Transformer-based classifier for pet adoption speed prediction"""
    
    def __init__(self, model_name='bert-base-uncased', num_classes=5, dropout=0.3):
        super(TransformerPetClassifier, self).__init__()
        
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # Get hidden size from transformer config
        hidden_size = self.transformer.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits


def load_data(train_csv_path):
    """Load training data"""
    print("Loading training data...")
    df = pd.read_csv(train_csv_path)
    print(f"Loaded {len(df)} training samples")
    print(f"AdoptionSpeed distribution:\n{df['AdoptionSpeed'].value_counts().sort_index()}")
    return df


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(outputs, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    kappa = cohen_kappa_score(true_labels, predictions, weights='quadratic')
    
    return avg_loss, accuracy, kappa


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    kappa = cohen_kappa_score(true_labels, predictions, weights='quadratic')
    
    return avg_loss, accuracy, kappa, predictions, true_labels


def main():
    # Configuration
    TRAIN_CSV = 'data/train/train.csv'
    METADATA_DIR = 'data/train_metadata'
    MODEL_NAME = 'bert-base-uncased'  # or 'distilbert-base-uncased' for faster training
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 5  # AdoptionSpeed: 0-4
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    df = load_data(TRAIN_CSV)
    
    # Split data
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42,
        stratify=df['AdoptionSpeed']
    )
    
    print(f"\nTrain size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    
    # Initialize tokenizer
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = PetAdoptionDataset(train_df, METADATA_DIR, tokenizer, MAX_LENGTH)
    val_dataset = PetAdoptionDataset(val_df, METADATA_DIR, tokenizer, MAX_LENGTH)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    print(f"\nInitializing model: {MODEL_NAME}")
    model = TransformerPetClassifier(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)
    
    best_kappa = 0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc, train_kappa = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Validate
        val_loss, val_acc, val_kappa, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Kappa: {train_kappa:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Kappa: {val_kappa:.4f}")
        
        # Save best model
        if val_kappa > best_kappa:
            best_kappa = val_kappa
            torch.save(model.state_dict(), 'best_transformer_model.pth')
            print(f"✓ Saved best model (Kappa: {best_kappa:.4f})")
    
    # Final evaluation
    print("\n" + "="*50)
    print("Final Evaluation on Validation Set")
    print("="*50)
    
    # Load best model
    model.load_state_dict(torch.load('best_transformer_model.pth'))
    val_loss, val_acc, val_kappa, val_preds, val_labels = evaluate(
        model, val_loader, criterion, device
    )
    
    print(f"\nBest Validation Results:")
    print(f"Loss: {val_loss:.4f}")
    print(f"Accuracy: {val_acc:.4f}")
    print(f"Cohen's Kappa (Quadratic): {val_kappa:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=[f'Speed {i}' for i in range(5)]))
    
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = main()
