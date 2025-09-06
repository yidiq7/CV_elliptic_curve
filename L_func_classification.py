import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import time

# --- 1. Configuration and Hyperparameters ---

# Decide which device to use (GPU if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Hyperparameters
LEARNING_RATE = 0.004
BATCH_SIZE = 128 # Increased for A100 GPU performance
EPOCHS = 100 # This task might converge quickly, 15 is a solid starting point
TRAIN_VAL_SPLIT_RATIO = 0.8 # 80% for training, 20% for validation

# File paths for your data
REAL_DATA_PATH = 'combined_twisted_arrays.npy'
FAKE_DATA_PATH = 'combined_twisted_arrays_fake.npy'


# --- 2. Data Loading and Preprocessing ---

print("Loading and preprocessing data using memory-mapping...")

# --- CHANGE 2: Load arrays using mmap_mode ---
# This does NOT load the data into RAM. It creates a pointer to the file on disk.
try:
    real_data_mmap = np.load(REAL_DATA_PATH, mmap_mode='r')
    fake_data_mmap = np.load(FAKE_DATA_PATH, mmap_mode='r')
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please make sure the .npy files are in the same directory as the script.")
    exit()

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
        # --- CHANGE 3: Slicing and processing happens here, on-the-fly ---
        # Only this single item is loaded from disk into RAM.
        
        # 1. Get the raw data slice
        raw_feature = self.data[idx]
        
        # 2. Remove the constant middle channel
        feature_sliced = raw_feature[:, :, [0, 2]]
        
        # 3. Convert to torch tensor and permute dimensions
        feature_tensor = torch.FloatTensor(feature_sliced).permute(2, 0, 1)
        
        # 4. Create the label
        label_tensor = torch.FloatTensor([self.label]).unsqueeze(0)
        
        return feature_tensor, label_tensor

# --- CHANGE 4: Create two separate datasets and combine them ---
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
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: (128, 6, 6)
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

# --- 5. Training and Validation ---

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss() # Numerically stable for binary classification
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


print("\nStarting training...")
start_time = time.time()

best_val_accuracy = 0.0
best_model_state = None

for epoch in range(EPOCHS):
    # --- Training phase ---
    model.train()
    running_loss = 0.0
    # ADDED: Initialize counters for training accuracy
    train_total_correct = 0
    train_total_samples = 0
    
    for features, labels in train_loader:
        labels = labels.squeeze(1)
        features, labels = features.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        # ADDED: Calculate training accuracy for the batch
        with torch.no_grad(): # No need to track gradients for accuracy calculation
            preds = torch.round(torch.sigmoid(outputs))
            train_total_correct += (preds == labels).sum().item()
            train_total_samples += labels.size(0)

    # --- Validation phase ---
    model.eval()
    val_total_correct = 0
    val_total_samples = 0
    with torch.no_grad():
        for features, labels in val_loader:
            labels = labels.squeeze(1)
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            outputs = model(features)
            preds = torch.round(torch.sigmoid(outputs))
            val_total_correct += (preds == labels).sum().item()
            val_total_samples += labels.size(0)

    # --- Epoch Summary ---
    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = train_total_correct / train_total_samples # ADDED
    val_accuracy = val_total_correct / val_total_samples
    
    # MODIFIED: Updated print statement
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | Val Acc: {val_accuracy:.4f}")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_state = model.state_dict().copy() # Use .copy() to avoid saving a reference
        print(f"  -> New best model saved with validation accuracy: {best_val_accuracy:.4f}")
        
end_time = time.time()
print(f"\nTraining finished in {(end_time - start_time):.2f} seconds.")

# --- You can now save the trained model if desired ---
torch.save(model.state_dict(), 'L_function_classifier.pth')
print("\nModel saved to L_function_classifier.pth")



# --- ADDED: Confusion Matrix Calculation ---
print("\n" + "="*50)
print("VALIDATION SET CONFUSION MATRIX")
print("="*50)

# Load best model state for final evaluation
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("Loaded best model for final evaluation")

model.eval()

# Initialize confusion matrix components
true_positives = 0  # Real correctly classified as Real (label=1, pred=1)
true_negatives = 0  # Fake correctly classified as Fake (label=0, pred=0)
false_positives = 0 # Fake incorrectly classified as Real (label=0, pred=1)
false_negatives = 0 # Real incorrectly classified as Fake (label=1, pred=0)

with torch.no_grad():
    for features, labels in val_loader:
        labels = labels.squeeze(1)
        features, labels = features.to(DEVICE), labels.to(DEVICE)
        
        outputs = model(features)
        preds = torch.round(torch.sigmoid(outputs))
        
        # Calculate confusion matrix components
        for pred, label in zip(preds, labels):
            if label == 1 and pred == 1:
                true_positives += 1
            elif label == 0 and pred == 0:
                true_negatives += 1
            elif label == 0 and pred == 1:
                false_positives += 1
            elif label == 1 and pred == 0:
                false_negatives += 1

# Print confusion matrix
print("\nConfusion Matrix:")
print("                 Predicted")
print("                 Fake  Real")
print(f"Actual Fake    [{true_negatives:5d} {false_positives:5d}]")
print(f"       Real    [{false_negatives:5d} {true_positives:5d}]")

# Calculate and print metrics
total = true_positives + true_negatives + false_positives + false_negatives
accuracy = (true_positives + true_negatives) / total if total > 0 else 0

# Precision, Recall, F1 for Real class (label=1)
precision_real = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall_real = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1_real = 2 * (precision_real * recall_real) / (precision_real + recall_real) if (precision_real + recall_real) > 0 else 0

# Precision, Recall, F1 for Fake class (label=0)
precision_fake = true_negatives / (true_negatives + false_negatives) if (true_negatives + false_negatives) > 0 else 0
recall_fake = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
f1_fake = 2 * (precision_fake * recall_fake) / (precision_fake + recall_fake) if (precision_fake + recall_fake) > 0 else 0

print("\n" + "-"*50)
print("Summary Statistics:")
print(f"Total validation samples: {total}")
print(f"Overall accuracy: {accuracy:.4f}")
print("\nPer-class metrics:")
print(f"Fake (0): Precision={precision_fake:.4f}, Recall={recall_fake:.4f}, F1={f1_fake:.4f}")
print(f"Real (1): Precision={precision_real:.4f}, Recall={recall_real:.4f}, F1={f1_real:.4f}")
print("-"*50)
