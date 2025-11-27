import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torch.amp import autocast, GradScaler
import time

# --- 1. Configuration and Hyperparameters ---
# Decide which device to use (GPU if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 256
EPOCHS = 150
TRAIN_VAL_SPLIT_RATIO = 0.8 # 80% for training, 20% for validation

# File paths for your data
IMAGE_SIZE = 200
REAL_DATA_PATH = f'combined_twisted_arrays_{IMAGE_SIZE}.npy'
FAKE_DATA_PATH = f'combined_twisted_arrays_fake_{IMAGE_SIZE}.npy'

# Class imbalance ratio (fake:real = 10:1)
CLASS_WEIGHT_RATIO = 3.0
OPTIMAL_THRESHOLD = 0.5

# Resume training configuration
RESUME_TRAINING = True  # Set to False to start fresh

# --- 2. Data Loading and Preprocessing ---

print("Loading and preprocessing data using memory-mapping...")

try:
    real_data_mmap = np.load(REAL_DATA_PATH, mmap_mode='r')
    fake_data_mmap = np.load(FAKE_DATA_PATH, mmap_mode='r')
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please make sure the .npy files are in the same directory as the script.")
    exit()

print(f"Real data shape: {real_data_mmap.shape}")
print(f"Fake data shape: {fake_data_mmap.shape}")
print(f"Class imbalance ratio (fake:real): {len(fake_data_mmap)/len(real_data_mmap):.1f}:1")

# --- 3. PyTorch Dataset and DataLoaders ---

class LFunctionDataset(Dataset):
    """
    Custom PyTorch Dataset for L-function data.
    Now designed to work with memory-mapped files.
    """
    def __init__(self, data_mmap, label_value):
        self.data = data_mmap
        self.label = label_value

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 1. Get the raw data slice
        raw_feature = self.data[idx]
        
        # 2. Remove the constant middle channel
        feature_sliced = raw_feature[:, :, [0, 2]]
        
        # 3. Convert to torch tensor and permute dimensions
        feature_tensor = torch.FloatTensor(feature_sliced).permute(2, 0, 1)
        
        # 4. Create the label
        label_tensor = torch.FloatTensor([self.label]).unsqueeze(0)
        
        return feature_tensor, label_tensor

real_dataset = LFunctionDataset(real_data_mmap, label_value=1)
fake_dataset = LFunctionDataset(fake_data_mmap, label_value=0)

# Use ConcatDataset to create a single logical dataset without loading everything to memory
full_dataset = ConcatDataset([real_dataset, fake_dataset])


# Split into training and validation sets
# Note: We create a generator for reproducibility in splitting
g = torch.Generator().manual_seed(42)
train_size = int(TRAIN_VAL_SPLIT_RATIO * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=g)

# Create DataLoaders
# The DataLoader will handle shuffling the combined dataset
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*10, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

print(f"Created {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
print("Data is being loaded from disk in batches, not all at once.")

# --- 4. CNN Model Definition ---

class LFunctionCNN(nn.Module):
    def __init__(self):
        super(LFunctionCNN, self).__init__()
        # Input: (Batch, 2, 100, 100)
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), # ADDED: Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (16, 50, 50)

            # Block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), # ADDED: Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (32, 25, 25)

            # Block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), # ADDED: Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (64, 12, 12)

            # Block 4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), # ADDED: Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (128, 6, 6)

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (512, 6, 6)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

# Instantiate the model
model = LFunctionCNN().to(DEVICE)
print("\nModel Architecture:")
print(model)

scaler = GradScaler('cuda')

# --- 5. Helper function for calculating metrics ---

def calculate_metrics(predictions, labels):
    """
    Efficient calculation of precision, recall, and F1 scores for binary classification.
    Uses vectorized operations to compute confusion matrix in one pass.
    """
    # Keep computations on the same device as inputs
    device = predictions.device
    
    # Ensure tensors are 1D (flatten if needed)
    y_true = labels.flatten().long()
    y_pred = predictions.flatten().long()
    
    # Create unique indices for each confusion matrix cell
    # This maps: (true=0,pred=0)→0, (true=0,pred=1)→1, (true=1,pred=0)→2, (true=1,pred=1)→3
    indices = 2 * y_true + y_pred
    
    # Count occurrences of each combination using bincount
    cm = torch.bincount(indices, minlength=4)
    tn, fp, fn, tp = cm[0].item(), cm[1].item(), cm[2].item(), cm[3].item()
    
    # Compute all metrics using the confusion matrix values
    # Use epsilon to avoid division by zero
    eps = 1e-7
    
    # Metrics for Real class (label=1)
    precision_real = tp / (tp + fp + eps)
    recall_real = tp / (tp + fn + eps)
    f1_real = 2 * (precision_real * recall_real) / (precision_real + recall_real + eps)
    
    # Metrics for Fake class (label=0)
    precision_fake = tn / (tn + fn + eps)
    recall_fake = tn / (tn + fp + eps)
    f1_fake = 2 * (precision_fake * recall_fake) / (precision_fake + recall_fake + eps)
    
    # Overall accuracy
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / (total + eps)
    
    return {
        'accuracy': accuracy,
        'precision_real': precision_real,
        'recall_real': recall_real,
        'f1_real': f1_real,
        'precision_fake': precision_fake,
        'recall_fake': recall_fake,
        'f1_fake': f1_fake,
        'confusion_matrix': {
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
    }

# --- 6. Training and Validation ---

# Loss function with class weight and optimizer
# pos_weight gives weight to positive class (real data, label=1)
# Since we have 10x more fake data, we weight the real class by 10
pos_weight = torch.tensor([CLASS_WEIGHT_RATIO]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Load checkpoint if it exists ---
START_EPOCH = 0
best_val_f1_real = 0.0
best_model_state = None

if RESUME_TRAINING:
    try:
        checkpoint = torch.load('L_function_classifier_checkpoint.pth', map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        START_EPOCH = checkpoint['epoch']
        best_val_f1_real = checkpoint['best_val_f1_real']
        best_model_state = checkpoint['model_state_dict']
        print(f"\nResumed training from epoch {START_EPOCH}")
        print(f"Best validation F1(Real) so far: {best_val_f1_real:.4f}")
    except FileNotFoundError:
        print("\nNo checkpoint found. Starting training from scratch.")
        START_EPOCH = 0
        best_val_f1_real = 0.0
        best_model_state = None
else:
    print("\nStarting training from scratch.")
    START_EPOCH = 0
    best_val_f1_real = 0.0
    best_model_state = None

print(f"\nUsing weighted BCE loss with pos_weight={pos_weight} to handle class imbalance")
print("\nStarting training...")
print("-" * 120)
print(f"{'Epoch':^7} | {'Loss':^8} | {'Train P(Real)':^13} | {'Train R(Real)':^13} | {'Train F1(Real)':^14} | {'Val P(Real)':^11} | {'Val R(Real)':^11} | {'Val F1(Real)':^12} | {'Train Time':^11}")
print("-" * 120)

start_time = time.time()

for epoch in range(START_EPOCH, START_EPOCH + EPOCHS):

    epoch_start_time = time.time()

    # --- Training phase ---
    model.train()
    running_loss = 0.0
    
    # Collect all predictions and labels for metric calculation
    all_train_preds = []
    all_train_labels = []
    
    for features, labels in train_loader:
        labels = labels.squeeze(1)
        features, labels = features.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        #with autocast('cuda'):  # Use mixed precision
        #    outputs = model(features)
        #    loss = criterion(outputs, labels)

        #scaler.scale(loss).backward()
        #scaler.step(optimizer)
        #scaler.update()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        # Collect predictions for metric calculation
        with torch.no_grad():
            preds = torch.round(torch.sigmoid(outputs))
            all_train_preds.append(preds)
            all_train_labels.append(labels)

    # Concatenate all predictions and labels
    all_train_preds = torch.cat(all_train_preds)
    all_train_labels = torch.cat(all_train_labels)
    
    # Calculate training metrics
    train_metrics = calculate_metrics(all_train_preds, all_train_labels)

    # --- Validation phase ---
    model.eval()
    all_val_preds = []
    all_val_labels = []
    
    with torch.no_grad():
        for features, labels in val_loader:
            labels = labels.squeeze(1)
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            outputs = model(features)
            preds = torch.round(torch.sigmoid(outputs))
            all_val_preds.append(preds)
            all_val_labels.append(labels)
    
    # Concatenate all predictions and labels
    all_val_preds = torch.cat(all_val_preds)
    all_val_labels = torch.cat(all_val_labels)
    
    # Calculate validation metrics
    val_metrics = calculate_metrics(all_val_preds, all_val_labels)

    # Synchronize CUDA before measuring time
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    epoch_time = time.time() - epoch_start_time
    
    # --- Epoch Summary ---
    avg_train_loss = running_loss / len(train_loader)
    
    # Print metrics (focusing on Real class as it's the minority class)
    print(f"{epoch+1:^7} | {avg_train_loss:^8.4f} | {train_metrics['precision_real']:^13.4f} | {train_metrics['recall_real']:^13.4f} | {train_metrics['f1_real']:^14.4f} | {val_metrics['precision_real']:^11.4f} | {val_metrics['recall_real']:^11.4f} | {val_metrics['f1_real']:^12.4f} | {epoch_time:^12.4f}")

    # Save best model based on validation F1 score for Real class
    if val_metrics['f1_real'] > best_val_f1_real:
        best_val_f1_real = val_metrics['f1_real']
        best_model_state = model.state_dict().copy()
        print(f"  -> New best model saved with validation F1(Real): {best_val_f1_real:.4f}")
        
end_time = time.time()
print("-" * 100)
print(f"\nTraining finished in {(end_time - start_time):.2f} seconds.")

# --- Save the trained model with checkpoint ---
checkpoint = {
    'epoch': START_EPOCH + EPOCHS,
    'model_state_dict': best_model_state if best_model_state is not None else model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_val_f1_real': best_val_f1_real,
}
torch.save(checkpoint, 'L_function_classifier_checkpoint.pth')
print("\nCheckpoint saved to L_function_classifier_checkpoint.pth")

# --- Final Evaluation with Confusion Matrix ---
print("\n" + "="*50)
print("FINAL VALIDATION SET EVALUATION")
print("="*50)

# Load best model state for final evaluation
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("Loaded best model for final evaluation")

for OPTIMAL_THRESHOLD in (0.3, 0.5, 0.7, 0.9):
    model.eval()

    # Collect all predictions for final evaluation
    all_final_preds = []
    all_final_labels = []

    with torch.no_grad():
        for features, labels in val_loader:
            labels = labels.squeeze(1)
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(features)
            probs = torch.sigmoid(outputs)
            preds = (probs > OPTIMAL_THRESHOLD).float()
            #preds = torch.round(torch.sigmoid(outputs))
            all_final_preds.append(preds)
            all_final_labels.append(labels)

    # Concatenate and calculate final metrics
    all_final_preds = torch.cat(all_final_preds)
    all_final_labels = torch.cat(all_final_labels)
    final_metrics = calculate_metrics(all_final_preds, all_final_labels)

    # Print confusion matrix
    cm = final_metrics['confusion_matrix']
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                 Fake  Real")
    print(f"Actual Fake    [{cm['tn']:5d} {cm['fp']:5d}]")
    print(f"       Real    [{cm['fn']:5d} {cm['tp']:5d}]")

    # Print detailed metrics
    print("\n" + "-"*50)
    print("Summary Statistics:")
    print(f"Total validation samples: {len(all_final_labels)}")
    print(f"Overall accuracy: {final_metrics['accuracy']:.4f}")
    print(f"\nPer-class metrics:")
    print(f"Fake (0): Precision={final_metrics['precision_fake']:.4f}, Recall={final_metrics['recall_fake']:.4f}, F1={final_metrics['f1_fake']:.4f}")
    print(f"Real (1): Precision={final_metrics['precision_real']:.4f}, Recall={final_metrics['recall_real']:.4f}, F1={final_metrics['f1_real']:.4f}")
    print("-"*50)

    # Print class distribution in validation set
    real_count = (all_final_labels == 1).sum().item()
    fake_count = (all_final_labels == 0).sum().item()
    print(f"\nValidation set class distribution:")
    print(f"Real samples: {real_count} ({real_count/len(all_final_labels)*100:.1f}%)")
    print(f"Fake samples: {fake_count} ({fake_count/len(all_final_labels)*100:.1f}%)")
