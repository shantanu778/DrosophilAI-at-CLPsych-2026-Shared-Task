import torch
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from dataset_v1 import PresenceRatingDataset, CLPsychDataLoader

class MentalRoBERTa_PresenceRating(nn.Module):
    """
    Mental-RoBERTa with dual expert heads for presence rating
    Similar to MoE but simpler: shared encoder + two specialized heads
    """
    
    def __init__(self, model_name='mental/mental-roberta-base', dropout=0.3):
        super().__init__()
        
        # Shared encoder (Mental-RoBERTa)
        self.roberta = AutoModel.from_pretrained(model_name)
        
        # Adaptive expert head
        self.adaptive_expert = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        # Maladaptive expert head
        self.maladaptive_expert = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        # Shared encoding
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled = outputs.last_hidden_state[:, 0, :]  # [batch, 768]
        
        # Expert predictions (raw logits)
        adaptive_raw = self.adaptive_expert(pooled)
        maladaptive_raw = self.maladaptive_expert(pooled)
        
        # Constrain to [1, 5] range
        adaptive_score = torch.sigmoid(adaptive_raw) * 4 + 1
        maladaptive_score = torch.sigmoid(maladaptive_raw) * 4 + 1
        
        return adaptive_score.squeeze(), maladaptive_score.squeeze()
    




# ========== Training Functions ==========
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    adaptive_losses = []
    maladaptive_losses = []
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        adaptive_target = batch['adaptive_score'].to(device)
        maladaptive_target = batch['maladaptive_score'].to(device)
        
        # Forward
        adaptive_pred, maladaptive_pred = model(input_ids, attention_mask)
        
        # Separate losses
        loss_adaptive = criterion(adaptive_pred, adaptive_target)
        loss_maladaptive = criterion(maladaptive_pred, maladaptive_target)
        loss = loss_adaptive + loss_maladaptive
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        adaptive_losses.append(loss_adaptive.item())
        maladaptive_losses.append(loss_maladaptive.item())
    
    return {
        'total_loss': total_loss / len(dataloader),
        'adaptive_loss': np.mean(adaptive_losses),
        'maladaptive_loss': np.mean(maladaptive_losses)
    }

def evaluate(model, dataloader, device):
    model.eval()
    
    all_adaptive_pred = []
    all_adaptive_true = []
    all_maladaptive_pred = []
    all_maladaptive_true = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            adaptive_pred, maladaptive_pred = model(input_ids, attention_mask)
            
            all_adaptive_pred.extend(adaptive_pred.cpu().numpy())
            all_adaptive_true.extend(batch['adaptive_score'].numpy())
            all_maladaptive_pred.extend(maladaptive_pred.cpu().numpy())
            all_maladaptive_true.extend(batch['maladaptive_score'].numpy())
    
    # Calculate metrics
    adaptive_mae = mean_absolute_error(all_adaptive_true, all_adaptive_pred)
    adaptive_rmse = np.sqrt(mean_squared_error(all_adaptive_true, all_adaptive_pred))
    adaptive_corr, _ = pearsonr(all_adaptive_true, all_adaptive_pred)
    
    maladaptive_mae = mean_absolute_error(all_maladaptive_true, all_maladaptive_pred)
    maladaptive_rmse = np.sqrt(mean_squared_error(all_maladaptive_true, all_maladaptive_pred))
    maladaptive_corr, _ = pearsonr(all_maladaptive_true, all_maladaptive_pred)
    
    # Accuracy within ±1
    adaptive_acc_1 = np.mean(np.abs(np.array(all_adaptive_true) - np.array(all_adaptive_pred)) <= 1)
    maladaptive_acc_1 = np.mean(np.abs(np.array(all_maladaptive_true) - np.array(all_maladaptive_pred)) <= 1)
    
    return {
        'adaptive_mae': float(adaptive_mae),
        'adaptive_rmse': float(adaptive_rmse),
        'adaptive_corr': float(adaptive_corr),
        'adaptive_acc_1': float(adaptive_acc_1),
        'maladaptive_mae': float(maladaptive_mae),
        'maladaptive_rmse': float(maladaptive_rmse),
        'maladaptive_corr': float(maladaptive_corr),
        'maladaptive_acc_1': float(maladaptive_acc_1),
        'predictions': {
            'adaptive_pred': all_adaptive_pred,
            'adaptive_true': all_adaptive_true,
            'maladaptive_pred': all_maladaptive_pred,
            'maladaptive_true': all_maladaptive_true
        }
    }

if __name__ == "__main__":

    #!/usr/bin/env python3
    """
    Train Mental-RoBERTa for Task 1.2 (Presence Rating)
    """

    print("="*60)
    print("Mental-RoBERTa Presence Rating - Task 1.2")
    print("="*60)

    # ========== Load Data ==========
    print("\nLoading data...")
    train_loader_data = CLPsychDataLoader('tasks12/', split='train')
    val_loader_data = CLPsychDataLoader('tasks12/', split='val')

    train_df = train_loader_data.load()
    val_df = val_loader_data.load()

    train_loader_data.verify_order()
    val_loader_data.verify_order()

    # ========== Prepare Datasets ==========
    print("\nPreparing datasets...")
    tokenizer = AutoTokenizer.from_pretrained('mental/mental-roberta-base')

    train_dataset = PresenceRatingDataset(train_df, tokenizer)
    val_dataset = PresenceRatingDataset(val_df, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16)

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # ========== Initialize Model ==========
    print("\nInitializing model...")
    model = MentalRoBERTa_PresenceRating()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ========== Training Setup ==========
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.MSELoss()

    # ========== Training Loop ==========
    num_epochs = 10
    best_avg_mae = float('inf')

    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)

    # for epoch in range(num_epochs):

    #     print(f"\nEpoch {epoch+1}/{num_epochs}")
    #     print("-" * 60)
        
    #     # Train
    #     train_metrics = train_epoch(model, train_dataloader, optimizer, criterion, device)
        
    #     # Evaluate
    #     val_metrics = evaluate(model, val_dataloader, device)
        
    #     # Print results
    #     print(f"\nTrain Loss:")
    #     print(f"  Total:       {train_metrics['total_loss']:.4f}")
    #     print(f"  Adaptive:    {train_metrics['adaptive_loss']:.4f}")
    #     print(f"  Maladaptive: {train_metrics['maladaptive_loss']:.4f}")
        
    #     print(f"\nValidation Metrics:")
    #     print(f"  Adaptive:")
    #     print(f"    MAE:      {val_metrics['adaptive_mae']:.4f}")
    #     print(f"    RMSE:     {val_metrics['adaptive_rmse']:.4f}")
    #     print(f"    Corr:     {val_metrics['adaptive_corr']:.4f}")
    #     print(f"    Acc(±1):  {val_metrics['adaptive_acc_1']:.4f}")
    #     print(f"  Maladaptive:")
    #     print(f"    MAE:      {val_metrics['maladaptive_mae']:.4f}")
    #     print(f"    RMSE:     {val_metrics['maladaptive_rmse']:.4f}")
    #     print(f"    Corr:     {val_metrics['maladaptive_corr']:.4f}")
    #     print(f"    Acc(±1):  {val_metrics['maladaptive_acc_1']:.4f}")
        
    #     # Save best model
    #     avg_mae = (val_metrics['adaptive_mae'] + val_metrics['maladaptive_mae']) / 2
    #     if avg_mae < best_avg_mae:
    #         best_avg_mae = avg_mae
    #         torch.save({
    #             'epoch': epoch,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'metrics': val_metrics
    #         }, 'mental_roberta_presence_best.pt')
    #         print(f"\n✅ Saved best model (Avg MAE: {avg_mae:.4f})")

    # print("\n" + "="*60)
    # print("Training Complete!")
    # print(f"Best Average MAE: {best_avg_mae:.4f}")
    # print("="*60)

    # # ========== Save Final Results ==========
    # with open('mental_roberta_results.json', 'w') as f:
    #     json.dump({
    #         'best_avg_mae': best_avg_mae,
    #         'final_val_metrics': {k: v for k, v in val_metrics.items() if k != 'predictions'}
    #     }, f, indent=2)

    # print("\n✅ Results saved to mental_roberta_results.json")


    #!/usr/bin/env python3
    """
    Generate predictions on validation set
    """

    # Load model
    checkpoint = torch.load('mental_roberta_presence_best.pt', weights_only=False)
    model = MentalRoBERTa_PresenceRating()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained('mental/mental-roberta-base')

    def predict_presence(text):
        """Predict presence scores for a single post"""
        
        encoding = tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            adaptive, maladaptive = model(input_ids, attention_mask)
        
        return {
            'adaptive_score': round(adaptive.item(), 2),
            'maladaptive_score': round(maladaptive.item(), 2)
        }

    # Test
    test_post = "I'm searching for someone who could help keep me accountable with daily exercise..."
    result = predict_presence(test_post)
    print(f"Prediction:")
    print(f"  Adaptive:    {result['adaptive_score']}")
    print(f"  Maladaptive: {result['maladaptive_score']}")





