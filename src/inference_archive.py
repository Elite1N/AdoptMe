import os
import torch
import pandas as pd
import numpy as np
import joblib
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm

# This script is for inference usage in predicting using trained models (ensemble) 
# .pth checkpoints & joblib xgboost

# TODO: -- BELOW CODE is GENERATED for drafting how we might execute this shi. --
import torch.nn as nn

# --- Configuration --- # TODO: recheck the path
TABULAR_MODEL_PATH = '../models/xgboost_tabular.joblib'
IMAGE_MODEL_PATH = '../models/resnet_image.pth'
TEXT_MODEL_PATH = '../models/distilbert_text.pth'
TEST_DATA_PATH = '../data/test/test.csv'
IMAGE_DIR = '../data/test_images/'
OUTPUT_PATH = '../submission.csv'
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Custom Dataset ---
class PetDataset(Dataset):
    def __init__(self, df, img_dir, tokenizer, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Image
        img_name = os.path.join(self.img_dir, row['Id'] + '.jpg')
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            # Handle missing images with a black placeholder
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
        
        # Text
        desc = str(row['Description']) if pd.notna(row['Description']) else ""
        text_encoding = self.tokenizer(
            desc,
            add_special_tokens=True,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'input_ids': text_encoding['input_ids'].flatten(),
            'attention_mask': text_encoding['attention_mask'].flatten(),
            'tabular_features': torch.tensor(row[tabular_cols].values.astype(np.float32))
        }

# --- Model Definitions (Simplified for Inference) ---
class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1) # Regression output

    def forward(self, x):
        return self.resnet(x)

class TextModel(nn.Module):
    def __init__(self):
        super(TextModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.regressor = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.regressor(cls_token)

# --- Feature Engineering Helpers ---
# (Assuming preprocessing logic matches training exactly)
def preprocess_tabular(df):
    # Dummy preprocessing: replace with actual logic used during training
    # For example: One-hot encoding, scaling, filling NaNs
    # Here we just select numeric columns for demonstration
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Fill NA for inference safety
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df, numeric_cols

# --- Main Inference Routine ---
if __name__ == "__main__":
    print(f"Loading test data from {TEST_DATA_PATH}...")
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    # Preprocess Tabular Data
    test_df_processed, tabular_cols = preprocess_tabular(test_df.copy())
    
    # 1. Load Models
    print("Loading models...")
    
    # XGBoost (Tabular)
    xgb_model = joblib.load(TABULAR_MODEL_PATH)
    
    # PyTorch Image Model
    img_model = ImageModel().to(DEVICE)
    img_model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=DEVICE))
    img_model.eval()
    
    # PyTorch Text Model
    text_model = TextModel().to(DEVICE)
    text_model.load_state_dict(torch.load(TEXT_MODEL_PATH, map_location=DEVICE))
    text_model.eval()
    
    # Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # 2. Prepare DataLoader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = PetDataset(test_df_processed, IMAGE_DIR, tokenizer, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 3. Prediction Loop
    print("Running inference...")
    
    img_preds = []
    text_preds = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            imgs = batch['image'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            masks = batch['attention_mask'].to(DEVICE)
            
            # Image Prediction
            img_out = img_model(imgs)
            img_preds.extend(img_out.cpu().numpy().flatten())
            
            # Text Prediction
            text_out = text_model(input_ids, masks)
            text_preds.extend(text_out.cpu().numpy().flatten())

    # Tabular Prediction (XGBoost expects numpy array or pandas DF)
    # Ensure correct column ordering matching training
    tabular_features = test_df_processed[tabular_cols].values
    tab_preds = xgb_model.predict(tabular_features)

    # 4. Ensemble (Simple Weighted Average) #TODO: use j4's generated dense layer for training combining inference
    print("Ensembling predictions...")
    # TODO: change this into a linear layer
    # Weights usually determined via validation set optimization
    w_tab, w_img, w_text = 0.5, 0.3, 0.2 
    
    final_preds = (w_tab * tab_preds) + \
                  (w_img * np.array(img_preds)) + \
                  (w_text * np.array(text_preds))

    # 5. Save Submission
    submission = pd.DataFrame({
        'Id': test_df['Id'],
        'AdoptionSpeed': final_preds
    })
    
    # Optional: If target is categorical, round or clip results
    # submission['AdoptionSpeed'] = np.round(submission['AdoptionSpeed']).astype(int)
    
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"Submission saved to {OUTPUT_PATH}")
