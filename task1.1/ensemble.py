import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ========== Parse JSON to Feature Vectors ==========
def parse_json_to_features(prediction, flag=True):
    """
    Convert JSON prediction to feature vector
    
    Returns:
        abcd_vector: [12] binary vector (6 dims × 2 polarities)
        presence_vector: [2] continuous values (adaptive, maladaptive presence scores)
    """
    if flag:
        prediction = prediction['prediction']
    else:
        prediction = prediction['ground_truth']
    
    dimensions = ['A', 'B-S', 'B-O', 'C-S', 'C-O', 'D']
    
    # Task 1.1: ABCD binary vector (12 dimensions)
    abcd_vector = np.zeros(12)
    # Adaptive state (even indices: 0, 2, 4, 6, 8, 10)
    if prediction and 'adaptive-state' in prediction and prediction['adaptive-state'] is not None:
        for i, dim in enumerate(dimensions):
            if dim in prediction['adaptive-state'] and dim != 'Presence':
                # print(dim,  prediction['adaptive-state'])
                abcd_vector[i*2] = 1
    
    # Maladaptive state (odd indices: 1, 3, 5, 7, 9, 11)
    if prediction and 'maladaptive-state' in prediction and prediction['maladaptive-state'] is not None:
        for i, dim in enumerate(dimensions):
            if dim in prediction['maladaptive-state'] and dim != 'Presence':
                abcd_vector[i*2 + 1] = 1
    
    # print(abcd_vector)
    # Task 1.2: Presence scores (2 values)
    presence_vector = np.zeros(2)
    
    if prediction and 'adaptive-state' in prediction and 'Presence' in prediction['adaptive-state']:
        presence_vector[0] = prediction['adaptive-state']['Presence']
    else:
        presence_vector[0] = 1  # Default: not present
    
    if prediction and 'maladaptive-state' in prediction and 'Presence' in prediction['maladaptive-state']:
        presence_vector[1] = prediction['maladaptive-state']['Presence']
    else:
        presence_vector[1] = 1  # Default: not present
    
    return abcd_vector, presence_vector


# ========== Example Usage ==========
# example_pred = {
#     "timeline_id": "91b6a42835",
#     "post_id": "28641e5b6d",
#     "adaptive-state": {
#         "Presence": 5,
#         "B-S": {"subelement": 1},
#         "B-O": {"subelement": 1},
#         "C-S": {"subelement": 1},
#         "D": {"subelement": 3}
#     },
#     "maladaptive-state": {
#         "Presence": 2,
#         "C-S": {"subelement": 2}
#     }
# }

# abcd, presence = parse_json_to_features(example_pred)
# print(f"ABCD vector: {abcd}")  # [0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
# print(f"Presence vector: {presence}")  # [5, 2]


# Load predictions from Phi, QWEN, Llama
with open('val_predictions.json') as f:
    phi_predictions = json.load(f)

with open('val_predictions.json') as f:
    qwen_predictions = json.load(f)

with open('val_predictions.json') as f:
    llama_predictions = json.load(f)

# Load ground truth
with open('val_predictions.json') as f:
    ground_truth = json.load(f)

print(f"Loaded {len(phi_predictions)} predictions from each model")

# ========== Convert All Predictions to Feature Vectors ==========
phi_abcd = []
phi_presence = []

qwen_abcd = []
qwen_presence = []

llama_abcd = []
llama_presence = []

gt_abcd = []
gt_presence = []

for i in range(len(phi_predictions)):
    # Phi
    abcd, pres = parse_json_to_features(phi_predictions[i])
    phi_abcd.append(abcd)
    phi_presence.append(pres)
    
    # QWEN
    abcd, pres = parse_json_to_features(qwen_predictions[i])
    qwen_abcd.append(abcd)
    qwen_presence.append(pres)
    
    # Llama
    abcd, pres = parse_json_to_features(llama_predictions[i])
    llama_abcd.append(abcd)
    llama_presence.append(pres)
    
    # Ground Truth
    abcd, pres = parse_json_to_features(ground_truth[i], flag=False)
    gt_abcd.append(abcd)
    gt_presence.append(pres)

# Convert to numpy
phi_abcd = np.array(phi_abcd)  # [N, 12]
phi_presence = np.array(phi_presence)  # [N, 2]

qwen_abcd = np.array(qwen_abcd)
qwen_presence = np.array(qwen_presence)

llama_abcd = np.array(llama_abcd)
llama_presence = np.array(llama_presence)

gt_abcd = np.array(gt_abcd)
gt_presence = np.array(gt_presence)

print(f"\nFeature shapes:")
print(f"ABCD: {phi_abcd.shape}")
print(f"Presence: {phi_presence.shape}")


class EnsembleDataset(Dataset):
    """
    Dataset combining predictions from Phi, QWEN, Llama
    """
    
    def __init__(self, phi_abcd, phi_pres, qwen_abcd, qwen_pres, 
                 llama_abcd, llama_pres, gt_abcd, gt_pres):
        
        self.phi_abcd = torch.FloatTensor(phi_abcd)
        self.phi_pres = torch.FloatTensor(phi_pres)
        
        self.qwen_abcd = torch.FloatTensor(qwen_abcd)
        self.qwen_pres = torch.FloatTensor(qwen_pres)
        
        self.llama_abcd = torch.FloatTensor(llama_abcd)
        self.llama_pres = torch.FloatTensor(llama_pres)
        
        self.gt_abcd = torch.FloatTensor(gt_abcd)
        self.gt_pres = torch.FloatTensor(gt_pres)
    
    def __len__(self):
        return len(self.phi_abcd)
    
    def __getitem__(self, idx):
        return {
            # ABCD predictions from 3 models
            'phi_abcd': self.phi_abcd[idx],
            'qwen_abcd': self.qwen_abcd[idx],
            'llama_abcd': self.llama_abcd[idx],
            
            # Presence predictions from 3 models
            'phi_pres': self.phi_pres[idx],
            'qwen_pres': self.qwen_pres[idx],
            'llama_pres': self.llama_pres[idx],
            
            # Ground truth
            'target_abcd': self.gt_abcd[idx],
            'target_pres': self.gt_pres[idx]
        }

# Create dataset
from sklearn.model_selection import train_test_split

# Split indices
indices = np.arange(len(phi_abcd))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

# Create train/val datasets
train_dataset = EnsembleDataset(
    phi_abcd[train_idx], phi_presence[train_idx],
    qwen_abcd[train_idx], qwen_presence[train_idx],
    llama_abcd[train_idx], llama_presence[train_idx],
    gt_abcd[train_idx], gt_presence[train_idx]
)

val_dataset = EnsembleDataset(
    phi_abcd[val_idx], phi_presence[val_idx],
    qwen_abcd[val_idx], qwen_presence[val_idx],
    llama_abcd[val_idx], llama_presence[val_idx],
    gt_abcd[val_idx], gt_presence[val_idx]
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

print(f"\nTrain samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")

print(train_dataset[1])

class JointMetaLearner(nn.Module):
    """
    Meta-learner that combines:
    - Task 1.1: ABCD classification (12 binary outputs)
    - Task 1.2: Presence rating (2 regression outputs)
    """
    
    def __init__(self):
        super().__init__()
        
        # ========== Shared Feature Extraction ==========
        # Input: 3 models × (12 ABCD + 2 presence) = 42 features
        self.shared_encoder = nn.Sequential(
            nn.Linear(42, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # ========== Task 1.1: ABCD Classification Head ==========
        self.abcd_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 12),
            nn.Sigmoid()  # Binary classification for 12 dimensions
        )
        
        # ========== Task 1.2: Presence Rating Head ==========
        self.presence_head = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 2)  # 2 outputs: adaptive, maladaptive
        )
    
    def forward(self, phi_abcd, phi_pres, qwen_abcd, qwen_pres, llama_abcd, llama_pres):
        """
        Args:
            phi_abcd: [batch, 12]
            phi_pres: [batch, 2]
            qwen_abcd: [batch, 12]
            qwen_pres: [batch, 2]
            llama_abcd: [batch, 12]
            llama_pres: [batch, 2]
        
        Returns:
            abcd_pred: [batch, 12] - binary predictions
            pres_pred: [batch, 2] - presence scores (1-5)
        """
        
        # Concatenate all features: [batch, 42]
        combined = torch.cat([
            phi_abcd, phi_pres,
            qwen_abcd, qwen_pres,
            llama_abcd, llama_pres
        ], dim=1)
        
        # Shared encoding
        shared_features = self.shared_encoder(combined)
        
        # Task-specific predictions
        abcd_pred = self.abcd_head(shared_features)
        
        # Presence scores: constrain to [1, 5]
        pres_raw = self.presence_head(shared_features)
        pres_pred = torch.sigmoid(pres_raw) * 4 + 1  # Maps to [1, 5]
        
        return abcd_pred, pres_pred
    

#!/usr/bin/env python3
"""
Train Joint Meta-Learner for Tasks 1.1 and 1.2
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score, mean_absolute_error
import numpy as np

print("="*60)
print("Training Joint Ensemble Meta-Learner")
print("="*60)

# Initialize model
model = JointMetaLearner()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(f"\nDevice: {device}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Loss functions
abcd_criterion = nn.BCELoss()  # Binary cross-entropy for ABCD
presence_criterion = nn.MSELoss()  # MSE for presence scores

# Loss weights
abcd_weight = 0.6
presence_weight = 0.4

# ========== Training Loop ==========
num_epochs = 50
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # ===== Train =====
    model.train()
    train_loss = 0
    train_abcd_loss = 0
    train_pres_loss = 0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        # Move to device
        phi_abcd = batch['phi_abcd'].to(device)
        phi_pres = batch['phi_pres'].to(device)
        qwen_abcd = batch['qwen_abcd'].to(device)
        qwen_pres = batch['qwen_pres'].to(device)
        llama_abcd = batch['llama_abcd'].to(device)
        llama_pres = batch['llama_pres'].to(device)
        
        target_abcd = batch['target_abcd'].to(device)
        target_pres = batch['target_pres'].to(device)
        
        # Forward
        abcd_pred, pres_pred = model(
            phi_abcd, phi_pres,
            qwen_abcd, qwen_pres,
            llama_abcd, llama_pres
        )
        
        # Losses
        loss_abcd = abcd_criterion(abcd_pred, target_abcd)
        loss_pres = presence_criterion(pres_pred, target_pres)
        
        # Combined loss
        loss = abcd_weight * loss_abcd + presence_weight * loss_pres
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss += loss.item()
        train_abcd_loss += loss_abcd.item()
        train_pres_loss += loss_pres.item()
    
    # ===== Validate =====
    model.eval()
    val_loss = 0
    val_abcd_loss = 0
    val_pres_loss = 0
    
    all_abcd_pred = []
    all_abcd_true = []
    all_pres_pred = []
    all_pres_true = []
    
    with torch.no_grad():
        for batch in val_loader:
            phi_abcd = batch['phi_abcd'].to(device)
            phi_pres = batch['phi_pres'].to(device)
            qwen_abcd = batch['qwen_abcd'].to(device)
            qwen_pres = batch['qwen_pres'].to(device)
            llama_abcd = batch['llama_abcd'].to(device)
            llama_pres = batch['llama_pres'].to(device)
            
            target_abcd = batch['target_abcd'].to(device)
            target_pres = batch['target_pres'].to(device)
            
            # Predict
            abcd_pred, pres_pred = model(
                phi_abcd, phi_pres,
                qwen_abcd, qwen_pres,
                llama_abcd, llama_pres
            )
            
            # Losses
            loss_abcd = abcd_criterion(abcd_pred, target_abcd)
            loss_pres = presence_criterion(pres_pred, target_pres)
            loss = abcd_weight * loss_abcd + presence_weight * loss_pres
            
            val_loss += loss.item()
            val_abcd_loss += loss_abcd.item()
            val_pres_loss += loss_pres.item()
            
            # Collect predictions
            all_abcd_pred.append((abcd_pred > 0.5).cpu().numpy())
            all_abcd_true.append(target_abcd.cpu().numpy())
            all_pres_pred.append(pres_pred.cpu().numpy())
            all_pres_true.append(target_pres.cpu().numpy())
    
    # Calculate metrics
    all_abcd_pred = np.vstack(all_abcd_pred)
    all_abcd_true = np.vstack(all_abcd_true)
    all_pres_pred = np.vstack(all_pres_pred)
    all_pres_true = np.vstack(all_pres_true)
    
    # ABCD F1
    f1_micro = f1_score(all_abcd_true, all_abcd_pred, average='micro')
    
    # Presence MAE
    mae_adaptive = mean_absolute_error(all_pres_true[:, 0], all_pres_pred[:, 0])
    mae_maladaptive = mean_absolute_error(all_pres_true[:, 1], all_pres_pred[:, 1])
    mae_avg = (mae_adaptive + mae_maladaptive) / 2
    
    # Print results
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {train_loss/len(train_loader):.4f} (ABCD: {train_abcd_loss/len(train_loader):.4f}, Pres: {train_pres_loss/len(train_loader):.4f})")
    print(f"  Val Loss:   {val_loss/len(val_loader):.4f} (ABCD: {val_abcd_loss/len(val_loader):.4f}, Pres: {val_pres_loss/len(val_loader):.4f})")
    print(f"  Val F1 (ABCD): {f1_micro:.4f}")
    print(f"  Val MAE (Presence): {mae_avg:.4f} (Adaptive: {mae_adaptive:.4f}, Maladaptive: {mae_maladaptive:.4f})")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'f1_micro': f1_micro,
            'mae_avg': mae_avg
        }, 'ensemble_meta_learner_best.pt')
        print(f"  ✅ Saved best model!")

print("\n" + "="*60)
print("Training Complete!")
print("="*60)

#!/usr/bin/env python3
"""
Use trained meta-learner for ensemble predictions
"""

# Load best model
checkpoint = torch.load('ensemble_meta_learner_best.pt')
model = JointMetaLearner()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded best model from epoch {checkpoint['epoch']}")
print(f"Val F1: {checkpoint['f1_micro']:.4f}")
print(f"Val MAE: {checkpoint['mae_avg']:.4f}")

# Predict on test set
def predict_ensemble(phi_pred, qwen_pred, llama_pred):
    """Generate ensemble prediction from 3 model outputs"""
    
    # Parse to features
    phi_abcd, phi_pres = parse_json_to_features(phi_pred)
    qwen_abcd, qwen_pres = parse_json_to_features(qwen_pred)
    llama_abcd, llama_pres = parse_json_to_features(llama_pred)
    
    # Convert to tensors
    phi_abcd = torch.FloatTensor(phi_abcd).unsqueeze(0)
    phi_pres = torch.FloatTensor(phi_pres).unsqueeze(0)
    qwen_abcd = torch.FloatTensor(qwen_abcd).unsqueeze(0)
    qwen_pres = torch.FloatTensor(qwen_pres).unsqueeze(0)
    llama_abcd = torch.FloatTensor(llama_abcd).unsqueeze(0)
    llama_pres = torch.FloatTensor(llama_pres).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        abcd_pred, pres_pred = model(
            phi_abcd, phi_pres,
            qwen_abcd, qwen_pres,
            llama_abcd, llama_pres
        )
    
    # Convert back to JSON format
    ensemble_output = binary_to_json(
        abcd_pred.squeeze().numpy(),
        pres_pred.squeeze().numpy()
    )
    
    return ensemble_output

def binary_to_json(abcd_vector, presence_vector):
    """Convert binary vector back to JSON format"""
    
    dimensions = ['A', 'B-S', 'B-O', 'C-S', 'C-O', 'D']
    
    output = {
        "adaptive-state": {"Presence": int(round(presence_vector[0]))},
        "maladaptive-state": {"Presence": int(round(presence_vector[1]))}
    }
    
    # ABCD elements
    for i, dim in enumerate(dimensions):
        # Adaptive (even indices)
        if abcd_vector[i*2] > 0.5:
            output["adaptive-state"][dim] = {"subelement": 1}
        
        # Maladaptive (odd indices)
        if abcd_vector[i*2 + 1] > 0.5:
            output["maladaptive-state"][dim] = {"subelement": 1}
    
    return output

# Example usage
ensemble_pred = predict_ensemble(
    phi_predictions[0],
    qwen_predictions[0],
    llama_predictions[0]
)

print(json.dumps(ensemble_pred, indent=2))