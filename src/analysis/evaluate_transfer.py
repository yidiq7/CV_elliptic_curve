import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR, RESULTS_DIR
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os
import time

# --- 1. Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 2. PyTorch Dataset ---
class LFunctionDataset(Dataset):
    """
    Custom PyTorch Dataset for L-function data.
    Designed to work with memory-mapped files.
    """
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

# --- 3. CNN Model Definition ---
class LFunctionCNN(nn.Module):
    def __init__(self):
        super(LFunctionCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
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

# --- 4. Metrics Helper ---
def calculate_metrics(predictions, labels):
    y_true = labels.flatten().long()
    y_pred = predictions.flatten().long()
    indices = 2 * y_true + y_pred
    cm = torch.bincount(indices, minlength=4)
    tn, fp, fn, tp = cm[0].item(), cm[1].item(), cm[2].item(), cm[3].item()
    
    eps = 1e-7
    precision_real = tp / (tp + fp + eps)
    recall_real = tp / (tp + fn + eps)
    f1_real = 2 * (precision_real * recall_real) / (precision_real + recall_real + eps)
    
    precision_fake = tn / (tn + fn + eps)
    recall_fake = tn / (tn + fp + eps)
    f1_fake = 2 * (precision_fake * recall_fake) / (precision_fake + recall_fake + eps)
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    
    return {
        'accuracy': accuracy,
        'precision_real': precision_real,
        'recall_real': recall_real,
        'f1_real': f1_real,
        'precision_fake': precision_fake,
        'recall_fake': recall_fake,
        'f1_fake': f1_fake,
        'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
    }

# --- 5. Evaluation Loop ---
def evaluate_model(size):
    print(f"\n{'='*60}")
    print(f"Evaluating Transfer Learning for N = {size}")
    print(f"{'='*60}")
    
    # Paths assuming script is run from within `transfer_learning` directory
    checkpoint_path = os.path.join(RESULTS_DIR, f'L_function_classifier_{size}_checkpoint.pth')
    real_data_path = os.path.join(DATA_DIR, f'combined_twisted_arrays_{size}.npy')
    fake_data_path = os.path.join(DATA_DIR, f'combined_twisted_arrays_fake_{size}.npy')
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    if not os.path.exists(real_data_path) or not os.path.exists(fake_data_path):
        print(f"Data files not found for N={size} in the current directory.")
        return

    print("Loading data...")
    real_data_mmap = np.load(real_data_path, mmap_mode='r')
    fake_data_mmap = np.load(fake_data_path, mmap_mode='r')
    
    real_dataset = LFunctionDataset(real_data_mmap, label_value=1)
    fake_dataset = LFunctionDataset(fake_data_mmap, label_value=0)
    full_dataset = ConcatDataset([real_dataset, fake_dataset])
    
    loader = DataLoader(full_dataset, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)
    
    print("Loading model...")
    model = LFunctionCNN().to(DEVICE)
    # Using weights_only=False to suppress the warning if it was saved natively,
    # but weights_only=True is safer. Reverting to standard torch.load.
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Evaluating...")
    start_time = time.time()
    with torch.no_grad():
        for features, labels in loader:
            labels = labels.squeeze(1)
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(features)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.append(preds)
            all_labels.append(labels)
            
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    metrics = calculate_metrics(all_preds, all_labels)
    cm = metrics['confusion_matrix']
    
    print(f"\nEvaluation finished in {time.time() - start_time:.2f} seconds.")
    print(f"Total samples: {len(all_labels)}")
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                 Fake  Real")
    print(f"Actual Fake    [{cm['tn']:5d} {cm['fp']:5d}]")
    print(f"       Real    [{cm['fn']:5d} {cm['tp']:5d}]")
    
    print("\nPer-class metrics:")
    print(f"Fake (0): Precision={metrics['precision_fake']:.4f}, Recall={metrics['recall_fake']:.4f}, F1={metrics['f1_fake']:.4f}")
    print(f"Real (1): Precision={metrics['precision_real']:.4f}, Recall={metrics['recall_real']:.4f}, F1={metrics['f1_real']:.4f}")

if __name__ == '__main__':
    for size in [100, 200, 300]:
        evaluate_model(size)
