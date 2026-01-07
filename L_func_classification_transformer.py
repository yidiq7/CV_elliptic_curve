import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torch.amp import autocast, GradScaler
import time
import argparse
import math
import os

# --- 1. Configuration and Hyperparameters ---
parser = argparse.ArgumentParser(description='L-function Classification with CNN or Transformer')
parser.add_argument('--model', type=str, default='transformer', choices=['cnn', 'transformer'], help='Model architecture')
parser.add_argument('--image_size', type=int, default=300, help='Input image size')
# Transformer Hyperparameters (Tuned for H200)
parser.add_argument('--patch_size', type=int, default=15, help='Patch size (15 for 300x300 -> 20x20 grid)')
parser.add_argument('--dim', type=int, default=512, help='Embedding dimension')
parser.add_argument('--depth', type=int, default=8, help='Transformer depth')
parser.add_argument('--heads', type=int, default=8, help='Attention heads')
parser.add_argument('--mlp_dim', type=int, default=2048, help='MLP dimension')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
# Training Hyperparameters
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=1024, help='Batch size (high for H200)')
parser.add_argument('--lr', type=float, default=1e-3, help='Peak learning rate')
parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay (AdamW)')
parser.add_argument('--warmup_epochs', type=int, default=5, help='Linear warmup epochs')
parser.add_argument('--use_conv_stem', action='store_true', help='Use a Convolutional Stem (Hybrid architecture)')
parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')

args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# File paths
IMAGE_SIZE = args.image_size
REAL_DATA_PATH = f'combined_twisted_arrays_{IMAGE_SIZE}.npy'
FAKE_DATA_PATH = f'combined_twisted_arrays_fake_{IMAGE_SIZE}.npy'
CLASS_WEIGHT_RATIO = 3.0

# --- 2. Data Loading ---

print(f"Looking for data files with size {IMAGE_SIZE}...")
if not os.path.exists(REAL_DATA_PATH):
    print(f"Error: {REAL_DATA_PATH} not found. Please generate data with SIZE={IMAGE_SIZE}.")
    exit()

try:
    real_data_mmap = np.load(REAL_DATA_PATH, mmap_mode='r')
    fake_data_mmap = np.load(FAKE_DATA_PATH, mmap_mode='r')
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

print(f"Real data: {real_data_mmap.shape} | Fake data: {fake_data_mmap.shape}")

class LFunctionDataset(Dataset):
    def __init__(self, data_mmap, label_value):
        self.data = data_mmap
        self.label = label_value

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = np.array(self.data[idx])
        feature_tensor = torch.FloatTensor(feature).permute(2, 0, 1) # (C, H, W)
        label_tensor = torch.FloatTensor([self.label])
        return feature_tensor, label_tensor

real_dataset = LFunctionDataset(real_data_mmap, label_value=1)
fake_dataset = LFunctionDataset(fake_data_mmap, label_value=0)
full_dataset = ConcatDataset([real_dataset, fake_dataset])

# Split
g = torch.Generator().manual_seed(42)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=g)

# DataLoader (High num_workers for high throughput)
num_workers = 16 
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                          num_workers=num_workers, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                        num_workers=num_workers, pin_memory=True, persistent_workers=True)

print(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")
print(f"Batch Size: {args.batch_size} | Steps per epoch: {len(train_loader)}")

# --- 3. Model Definition ---

class LFunctionTransformer(nn.Module):
    def __init__(self, image_size=IMAGE_SIZE, patch_size=args.patch_size, dim=args.dim, 
                 depth=args.depth, heads=args.heads, mlp_dim=args.mlp_dim, channels=2, dropout=args.dropout, 
                 use_conv_stem=False):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        
        num_patches = (image_size // patch_size) ** 2
        
        if use_conv_stem:
            # Hybrid Architecture: Small CNN stem before tokenization
            # This extracts local features (3x3 convs) before projecting to tokens
            self.to_patch_embedding = nn.Sequential(
                # Layer 1: Extract low-level features
                nn.Conv2d(channels, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                # Layer 2: Refine features
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                # Layer 3: Project to embeddings (using stride=patch_size to maintain sequence length)
                nn.Conv2d(64, dim, kernel_size=patch_size, stride=patch_size),
                nn.Flatten(2), # (B, dim, num_patches)
            )
        else:
            # Standard ViT: Linear projection of flattened patches
            self.to_patch_embedding = nn.Sequential(
                nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
                nn.Flatten(2), # (B, dim, num_patches)
            )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, 
                                                 dropout=dropout, activation='gelu', batch_first=True,
                                                 norm_first=True) # Pre-Norm is generally better
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        x = x.transpose(1, 2) # (B, N, D)
        
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x[:, 0] # CLS token
        return self.mlp_head(x)

class LFunctionCNN(nn.Module):
    # (Kept for compatibility, mostly unchanged)
    def __init__(self):
        super(LFunctionCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.classifier(self.conv_layers(x))

if args.model == 'transformer':
    stem_type = "Convolutional" if args.use_conv_stem else "Standard Linear"
    print(f"\nInitializing Transformer ({stem_type} Stem, Patch: {args.patch_size}, Dim: {args.dim}, Depth: {args.depth}, Heads: {args.heads})")
    model = LFunctionTransformer(use_conv_stem=args.use_conv_stem).to(DEVICE)
else:
    print("\nInitializing CNN")
    model = LFunctionCNN().to(DEVICE)

# --- 4. Training Setup with Scheduler ---

pos_weight = torch.tensor([CLASS_WEIGHT_RATIO]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scaler = GradScaler('cuda')

# Checkpoint Loading logic
START_EPOCH = 0
best_val_f1 = 0.0

if args.resume:
    if os.path.isfile(args.resume):
        print(f"\nLoading checkpoint from '{args.resume}'...")
        checkpoint = torch.load(args.resume, map_location=DEVICE)
        
        # Check if it's a full checkpoint or just model weights
        if 'epoch' in checkpoint:
            START_EPOCH = checkpoint['epoch']
            best_val_f1 = checkpoint.get('best_val_f1_real', 0.0)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Resuming from epoch {START_EPOCH} (Best F1: {best_val_f1:.4f})")
        else:
            # Assume it's just the model state dict
            print("Checkpoint appears to be model weights only (no epoch/optimizer state).")
            print("Loading weights and starting fine-tuning/training from Epoch 0.")
            model.load_state_dict(checkpoint)
            START_EPOCH = 0
            best_val_f1 = 0.0 # Reset best f1 tracking
            
    else:
        print(f"\nCheckpoint '{args.resume}' not found. Starting from scratch.")

# Learning Rate Scheduler (Linear Warmup + Cosine Decay)
def get_scheduler(optimizer, warmup_epochs, max_epochs, steps_per_epoch, last_epoch=-1):
    total_steps = max_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    # Calculate last_epoch for the scheduler based on the resumed epoch
    scheduler_last_epoch = (last_epoch * steps_per_epoch) - 1 if last_epoch > 0 else -1
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=scheduler_last_epoch)

scheduler = get_scheduler(optimizer, args.warmup_epochs, args.epochs, len(train_loader), last_epoch=START_EPOCH)

# --- 5. Helper Metrics ---
def calculate_metrics(predictions, labels):
    y_true = labels.flatten().long()
    y_pred = predictions.flatten().long()
    indices = 2 * y_true + y_pred
    cm = torch.bincount(indices, minlength=4)
    tn, fp, fn, tp = cm[0].item(), cm[1].item(), cm[2].item(), cm[3].item()
    eps = 1e-7
    
    p_real = tp / (tp + fp + eps)
    r_real = tp / (tp + fn + eps)
    f1_real = 2 * (p_real * r_real) / (p_real + r_real + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    return {'f1_real': f1_real, 'acc': acc, 'p_real': p_real, 'r_real': r_real}

# --- 6. Training Loop ---

print("\nStarting training...")
print("-" * 100)
print(f"{ 'Epoch':^7} | { 'Loss':^8} | { 'LR':^9} | { 'Train F1':^10} | { 'Val F1':^10} | { 'Val Acc':^10} | { 'Time':^8}")
print("-" * 100)

best_val_f1 = 0.0
start_time = time.time()

for epoch in range(args.epochs):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    
    all_preds = []
    all_labels = []

    for features, labels in train_loader:
        labels = labels.to(DEVICE) # keep (B, 1)
        features = features.to(DEVICE)

        optimizer.zero_grad()
        
        with autocast('cuda'):
            outputs = model(features)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        running_loss += loss.item()
        
        with torch.no_grad():
            preds = torch.round(torch.sigmoid(outputs))
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # Train Metrics
    train_metrics = calculate_metrics(torch.cat(all_preds), torch.cat(all_labels))
    avg_loss = running_loss / len(train_loader)
    current_lr = scheduler.get_last_lr()[0]

    # Validation
    model.eval()
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(DEVICE)
            with autocast('cuda'):
                outputs = model(features)
            preds = torch.round(torch.sigmoid(outputs))
            val_preds.append(preds.cpu())
            val_labels.append(labels)
            
    val_metrics = calculate_metrics(torch.cat(val_preds), torch.cat(val_labels))
    
    epoch_time = time.time() - epoch_start
    
    print(f"{epoch+1:^7} | {avg_loss:^8.4f} | {current_lr:^9.2e} | {train_metrics['f1_real']:^10.4f} | {val_metrics['f1_real']:^10.4f} | {val_metrics['acc']:^10.4f} | {epoch_time:^8.1f}")
    
    # Save regular checkpoint
    save_path = f'L_function_{args.model}_{IMAGE_SIZE}_checkpoint.pth'
    checkpoint = {
        'epoch': epoch + 1, # Save next epoch to start from
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_f1_real': best_val_f1,
        'args': vars(args)
    }
    torch.save(checkpoint, save_path)
    print(f"\nCheckpoint saved to {save_path}")

    # Save best model (full checkpoint now)
    if val_metrics['f1_real'] > best_val_f1:
        best_val_f1 = val_metrics['f1_real']
        # Update best val f1 in checkpoint dict before saving
        checkpoint['best_val_f1_real'] = best_val_f1
        torch.save(checkpoint, f'best_transformer_{IMAGE_SIZE}.pth')
        print(f"  -> New best model saved (F1: {best_val_f1:.4f})")

print(f"\nTraining finished in {(time.time() - start_time)/60:.1f} minutes.")
print(f"Best Val F1 (Real): {best_val_f1:.4f}")