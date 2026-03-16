import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

LEARNING_RATE = 0.001
BATCH_SIZE = 256
EPOCHS = 1  # ONLY RUN 1 EPOCH
TRAIN_VAL_SPLIT_RATIO = 0.8

IMAGE_SIZE = 100
REAL_DATA_PATH = f'combined_twisted_arrays_{IMAGE_SIZE}.npy'
FAKE_DATA_PATH = f'combined_twisted_arrays_fake_{IMAGE_SIZE}.npy'
CLASS_WEIGHT_RATIO = 3.0

print("Loading data...")
try:
    real_data_mmap = np.load(REAL_DATA_PATH, mmap_mode='r')
    fake_data_mmap = np.load(FAKE_DATA_PATH, mmap_mode='r')
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    exit()

class LFunctionDataset(Dataset):
    def __init__(self, data_mmap, label_value):
        self.data = data_mmap
        self.label = label_value

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = np.array(self.data[idx])
        feature_tensor = torch.FloatTensor(feature).permute(2, 0, 1)
        label_tensor = torch.FloatTensor([self.label]).unsqueeze(0)
        return feature_tensor, label_tensor

real_dataset = LFunctionDataset(real_data_mmap, label_value=1)
fake_dataset = LFunctionDataset(fake_data_mmap, label_value=0)
full_dataset = ConcatDataset([real_dataset, fake_dataset])

g = torch.Generator().manual_seed(42)
train_size = int(TRAIN_VAL_SPLIT_RATIO * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=g)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

class LFunctionCNN(nn.Module):
    def __init__(self):
        super(LFunctionCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

model = LFunctionCNN().to(DEVICE)
pos_weight = torch.tensor([CLASS_WEIGHT_RATIO]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def calculate_metrics_step(predictions, labels):
    y_true = labels.flatten().long()
    y_pred = predictions.flatten().long()
    indices = 2 * y_true + y_pred
    cm = torch.bincount(indices, minlength=4)
    tn, fp, fn, tp = cm[0].item(), cm[1].item(), cm[2].item(), cm[3].item()
    eps = 1e-7
    precision_real = tp / (tp + fp + eps)
    recall_real = tp / (tp + fn + eps)
    f1_real = 2 * (precision_real * recall_real) / (precision_real + recall_real + eps)
    return precision_real, recall_real, f1_real

print("\nStarting training (Intra-Epoch Reporting every 100 steps)...")
print("-" * 120)
print(f"{'Step':^7} | {'Loss':^8} | {'Train P(Real)':^13} | {'Train R(Real)':^13} | {'Train F1(Real)':^14} | {'Val P(Real)':^11} | {'Val R(Real)':^11} | {'Val F1(Real)':^12} | {'Time':^11}")
print("-" * 120)

start_time = time.time()
step_start_time = time.time()

model.train()
running_loss = 0.0
all_train_preds = []
all_train_labels = []
step = 0

for features, labels in train_loader:
    step += 1
    labels = labels.squeeze(1)
    features, labels = features.to(DEVICE), labels.to(DEVICE)

    optimizer.zero_grad()
    outputs = model(features)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    running_loss += loss.item()
    
    with torch.no_grad():
        preds = torch.round(torch.sigmoid(outputs))
        all_train_preds.append(preds)
        all_train_labels.append(labels)
        
    if step % 100 == 0:
        # Calculate training metrics for the last 100 steps
        train_preds_tensor = torch.cat(all_train_preds)
        train_labels_tensor = torch.cat(all_train_labels)
        train_p, train_r, train_f1 = calculate_metrics_step(train_preds_tensor, train_labels_tensor)
        avg_train_loss = running_loss / 100
        
        # Evaluate on FULL validation set
        model.eval()
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for v_features, v_labels in val_loader:
                v_labels = v_labels.squeeze(1)
                v_features, v_labels = v_features.to(DEVICE), v_labels.to(DEVICE)
                v_outputs = model(v_features)
                v_preds = torch.round(torch.sigmoid(v_outputs))
                all_val_preds.append(v_preds)
                all_val_labels.append(v_labels)
                
        val_preds_tensor = torch.cat(all_val_preds)
        val_labels_tensor = torch.cat(all_val_labels)
        val_p, val_r, val_f1 = calculate_metrics_step(val_preds_tensor, val_labels_tensor)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        step_time = time.time() - step_start_time
        
        # Print matching original format
        print(f"{step:^7} | {avg_train_loss:^8.4f} | {train_p:^13.4f} | {train_r:^13.4f} | {train_f1:^14.4f} | {val_p:^11.4f} | {val_r:^11.4f} | {val_f1:^12.4f} | {step_time:^11.4f}")
        
        # Reset training accumulators for the next 100 steps
        model.train()
        running_loss = 0.0
        all_train_preds = []
        all_train_labels = []
        step_start_time = time.time()

print("-" * 120)
print(f"Epoch finished in {time.time() - start_time:.2f} seconds.")
