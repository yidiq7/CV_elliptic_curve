import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import csv
import ast
import sympy
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
import argparse

# --- Configuration ---
SEQ_LEN = 500  # Number of primes to use
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_PLOT = 'murmuration_plot_fake_predicted.png'

print(f"Using device: {DEVICE}")

# --- 1. Data Loading and Preprocessing ---

def get_primes(n):
    print(f"Generating first {n} primes...")
    return np.array([sympy.prime(i) for i in range(1, n + 1)])

def load_real_data(filepath, max_len):
    print(f"Loading real data from {filepath}...")
    X = []
    y = []
    
    try:
        csv.field_size_limit(sys.maxsize)
    except:
        pass

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in tqdm(reader, desc="Parsing Real Data"):
            if not row or (row[0].strip() and not row[0].lstrip('-').replace('.', '', 1).isdigit()):
                continue
            
            try:
                # Format: [conductor, rank, "[ap_list]"]
                if len(row) < 3: continue
                
                rank = int(row[1])
                if rank not in [0, 1, 2]:
                    continue
                
                ap_str = row[2].strip()
                if not (ap_str.startswith('[') and ap_str.endswith(']')):
                    continue
                    
                aps = ast.literal_eval(ap_str)
                if len(aps) < max_len:
                    padded = np.zeros(max_len)
                    padded[:len(aps)] = aps
                    aps = padded
                else:
                    aps = aps[:max_len]
                
                X.append(aps)
                y.append(rank)
            except (ValueError, SyntaxError):
                continue
                
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.longlong)

def load_fake_data(filepath, max_len):
    print(f"Loading fake data from {filepath}...")
    X = []
    
    try:
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for row in tqdm(reader, desc="Parsing Fake Data"):
                if not row: continue
                
                try:
                    # Handle header or comments
                    if not row[0].lstrip('-').replace('.', '', 1).isdigit():
                        continue
                        
                    aps = np.array([float(x) for x in row], dtype=np.float32)
                    
                    if len(aps) < max_len:
                        padded = np.zeros(max_len)
                        padded[:len(aps)] = aps
                        aps = padded
                    else:
                        aps = aps[:max_len]
                        
                    X.append(aps)
                except ValueError:
                    continue
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return np.array([])
        
    return np.array(X, dtype=np.float32)

def calculate_accuracy(y_true, y_pred):
    if y_pred.dim() > 1:
        _, predicted = torch.max(y_pred, 1)
    else:
        predicted = y_pred
    correct = (predicted == y_true).sum().item()
    return correct / y_true.size(0)

# --- 2. Model Definition ---

class CNN(nn.Module):
    def __init__(self, num_classes, input_length):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, padding=1)

        # Dummy input to calculate flattened_size
        dummy_input = torch.zeros(1, 1, input_length)
        self.flattened_size = self._get_flattened_size(dummy_input)

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.5)

    def _get_flattened_size(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))           
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- 3. Main Execution ---

def main():
    parser = argparse.ArgumentParser(description='Train CNN on real data and predict ranks for fake data.')
    parser.add_argument('--real_csv', type=str, default='ap.csv', help='Path to real data CSV')
    parser.add_argument('--fake_csv', type=str, default='fake_ap.csv', help='Path to fake data CSV')
    args = parser.parse_args()

    if not os.path.exists(args.real_csv):
        print(f"Error: {args.real_csv} not found.")
        return

    # 1. Prepare Data
    primes = get_primes(SEQ_LEN)
    sqrt_primes = np.sqrt(primes).astype(np.float32)
    
    # Load Real Data
    X_real_raw, y_real = load_real_data(args.real_csv, SEQ_LEN)
    if len(X_real_raw) == 0:
        print("No valid real data found.")
        return
        
    print(f"Loaded {len(X_real_raw)} real samples.")
    
    # Normalize: a_p / sqrt(p)
    X_real_norm = X_real_raw / sqrt_primes
    
    # Convert to Tensor
    X_tensor = torch.tensor(X_real_norm).unsqueeze(1) # (N, 1, SEQ_LEN)
    y_tensor = torch.tensor(y_real)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    # 2. Train Model
    print("\nTraining Classifier...")
    num_classes = 3 # Ranks 0, 1, 2
    model = CNN(num_classes, SEQ_LEN).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_accuracies = []
    test_accuracies = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Evaluation
        model.eval()
        with torch.no_grad():
            # Calculate Training Accuracy (approximation from last batch to save time/memory or full pass)
            # Doing full pass for accuracy as per user snippet structure preference
            train_correct = 0
            train_total = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                train_correct += (outputs.argmax(1) == labels).sum().item()
                train_total += labels.size(0)
            train_acc = train_correct / train_total
            train_accuracies.append(train_acc)

            # Calculate Validation Accuracy
            val_correct = 0
            val_total = 0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
            val_acc = val_correct / val_total
            test_accuracies.append(val_acc)

        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}, '
              f'Train Accuracy: {train_acc:.4f}, Test Accuracy: {val_acc:.4f}')

    # 3. Process Fake Data
    if not os.path.exists(args.fake_csv):
        print(f"\n{args.fake_csv} not found. Skipping prediction.")
        return

    X_fake_raw = load_fake_data(args.fake_csv, SEQ_LEN)
    if len(X_fake_raw) == 0:
        print("No valid fake data found.")
        return
        
    print(f"\nPredicting ranks for {len(X_fake_raw)} fake samples...")
    
    # Normalize Fake Data
    X_fake_norm = X_fake_raw / sqrt_primes
    X_fake_tensor = torch.tensor(X_fake_norm).unsqueeze(1)
    
    fake_dataset = TensorDataset(X_fake_tensor)
    fake_loader = DataLoader(fake_dataset, batch_size=BATCH_SIZE * 2)
    
    all_preds = []
    model.eval()
    with torch.no_grad():
        for (X_batch,) in fake_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.append(predicted.cpu().numpy())
            
    all_preds = np.concatenate(all_preds)
    
    print(f"Predicted distribution: {np.bincount(all_preds, minlength=3)}")
    
    # 4. Plot Murmuration
    print("\nGenerating Murmuration Plot for Fake Data...")
    
    rank_groups = {0: [], 1: [], 2: []}
    
    # Group original (unnormalized) data by predicted rank
    for i, rank in enumerate(all_preds):
        rank_groups[rank].append(X_fake_raw[i])
        
    plt.figure(figsize=(12, 6))
    cmap = plt.get_cmap('tab10')
    
    for rank in [0, 1, 2]:
        data_list = rank_groups[rank]
        if not data_list:
            print(f"No fake samples predicted as Rank {rank}")
            continue
            
        data_arr = np.array(data_list)
        # Compute average per prime
        avg_ap = np.mean(data_arr, axis=0)
        
        count = len(data_list)
        label = f'Pred Rank {rank} (N={count})'
        color = cmap(rank)
        
        plt.plot(primes, avg_ap, '.', markersize=2, alpha=0.5, color=color, label=label)
        
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Prime p')
    plt.ylabel('Average a_p')
    plt.title(f'Murmuration of Fake Elliptic Curves (by Predicted Rank)\nModel trained on normalized $a_p / \sqrt{{p}}$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Plot saved to {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()
