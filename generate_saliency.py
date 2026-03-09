import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import csv
from torch.utils.data import DataLoader, Dataset

# --- 1. Model Architecture (Must match L_func_classification.py) ---
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

# --- 2. Simplified Dataset for Saliency ---
class SaliencyDataset(Dataset):
    def __init__(self, data_path, indices=None, num_samples=None):
        self.data = np.load(data_path, mmap_mode='r')
        if indices is None:
            max_len = len(self.data)
            self.indices = list(range(num_samples if num_samples is not None else max_len))
        else:
            self.indices = indices[:num_samples] if num_samples is not None else indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        feature = np.array(self.data[real_idx])
        feature_tensor = torch.FloatTensor(feature).permute(2, 0, 1)
        return feature_tensor

def compute_average_saliency(model, loader, device, num_samples=None):
    model.eval()
    avg_saliency = None
    samples_processed = 0
    
    for batch in loader:
        batch = batch.to(device)
        batch.requires_grad = True
        
        outputs = model(batch)
        score = outputs.sum()
        model.zero_grad()
        score.backward()
        
        saliency = batch.grad.data.abs()
        batch_avg = saliency.sum(dim=0)
        
        if avg_saliency is None:
            avg_saliency = batch_avg
        else:
            avg_saliency += batch_avg
            
        samples_processed += len(batch)
        if num_samples is not None and samples_processed >= num_samples:
            break
            
    return (avg_saliency / samples_processed).cpu().numpy()

import random

def get_rank_indices(csv_path):
    rank0_indices = []
    rank1_indices = []
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Cannot separate ranks.")
        return None, None
        
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for i, row in enumerate(reader):
            try:
                rank = int(row[1])
                if rank == 0:
                    rank0_indices.append(i)
                elif rank == 1:
                    rank1_indices.append(i)
            except (ValueError, IndexError):
                continue
                
    # Randomly shuffle indices to avoid bias (e.g., if ap.csv is sorted by conductor)
    random.seed(42)
    random.shuffle(rank0_indices)
    random.shuffle(rank1_indices)
    
    return rank0_indices, rank1_indices

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMAGE_SIZE = 100 
    CHECKPOINT_PATH = f'L_function_classifier_{IMAGE_SIZE}_checkpoint.pth'
    REAL_DATA_PATH = f'combined_twisted_arrays_{IMAGE_SIZE}.npy'
    FAKE_DATA_PATH = f'combined_twisted_arrays_fake_{IMAGE_SIZE}.npy'
    AP_CSV_PATH = 'ap.csv'
    NUM_SAMPLES = None # Set to None to use the entire dataset
    
    model = LFunctionCNN().to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {CHECKPOINT_PATH}")
    else:
        print(f"Checkpoint {CHECKPOINT_PATH} not found!")
        return

    # Get indices for ranks
    print("Reading ap.csv to separate ranks...")
    rank0_indices, rank1_indices = get_rank_indices(AP_CSV_PATH)
    
    if rank0_indices is None:
        print("Falling back to unseparated real data.")
        real_ds = SaliencyDataset(REAL_DATA_PATH, num_samples=NUM_SAMPLES)
        real_loader = DataLoader(real_ds, batch_size=256, shuffle=False)
        rank0_saliency = compute_average_saliency(model, real_loader, DEVICE, NUM_SAMPLES)
        rank1_saliency = rank0_saliency # Just duplicate if we can't separate
    else:
        print(f"Found {len(rank0_indices)} Rank 0 curves and {len(rank1_indices)} Rank 1 curves.")
        # Rank 0 Saliency
        r0_ds = SaliencyDataset(REAL_DATA_PATH, indices=rank0_indices, num_samples=NUM_SAMPLES)
        r0_loader = DataLoader(r0_ds, batch_size=256, shuffle=False)
        print(f"Computing Rank 0 saliency ({len(r0_ds)} samples)...")
        rank0_saliency = compute_average_saliency(model, r0_loader, DEVICE, NUM_SAMPLES)
        
        # Rank 1 Saliency
        r1_ds = SaliencyDataset(REAL_DATA_PATH, indices=rank1_indices, num_samples=NUM_SAMPLES)
        r1_loader = DataLoader(r1_ds, batch_size=256, shuffle=False)
        print(f"Computing Rank 1 saliency ({len(r1_ds)} samples)...")
        rank1_saliency = compute_average_saliency(model, r1_loader, DEVICE, NUM_SAMPLES)
    
    # Fake Saliency
    fake_ds = SaliencyDataset(FAKE_DATA_PATH, num_samples=NUM_SAMPLES)
    fake_loader = DataLoader(fake_ds, batch_size=256, shuffle=False)
    print(f"Computing Fake saliency ({len(fake_ds)} samples)...")
    fake_saliency = compute_average_saliency(model, fake_loader, DEVICE, NUM_SAMPLES)
    
    # Create output directory
    output_dir = 'saliency_maps'
    os.makedirs(output_dir, exist_ok=True)

    # Save raw data
    np.save(os.path.join(output_dir, 'rank0_saliency_avg.npy'), rank0_saliency)
    np.save(os.path.join(output_dir, 'rank1_saliency_avg.npy'), rank1_saliency)
    np.save(os.path.join(output_dir, 'fake_saliency_avg.npy'), fake_saliency)
    
    # Average across channels (Real/Imag parts)
    r0_map = rank0_saliency.mean(axis=0)
    r1_map = rank1_saliency.mean(axis=0)
    fake_map = fake_saliency.mean(axis=0)
    
    # Plotting 2D Heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    im0 = axes[0, 0].imshow(r0_map, cmap='hot')
    axes[0, 0].set_title('Average Rank 0 Saliency')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(r1_map, cmap='hot')
    axes[0, 1].set_title('Average Rank 1 Saliency')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[1, 0].imshow(fake_map, cmap='hot')
    axes[1, 0].set_title('Average Fake Saliency')
    plt.colorbar(im2, ax=axes[1, 0])
    
    diff_map = r0_map - fake_map
    im3 = axes[1, 1].imshow(diff_map, cmap='coolwarm')
    axes[1, 1].set_title('Rank 0 - Fake Saliency')
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'saliency_maps_2d.png'), dpi=300)
    print(f"2D Saliency maps saved to {output_dir}/saliency_maps_2d.png")

    # 1D Projections
    # 1. Primes (average over twists/characters -> axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(r0_map.mean(axis=1), label='Rank 0 (Average over twists)')
    plt.plot(r1_map.mean(axis=1), label='Rank 1 (Average over twists)')
    plt.plot(fake_map.mean(axis=1), label='Fake (Average over twists)', linestyle='--')
    plt.title('Saliency Projection onto Prime Index')
    plt.xlabel('Prime index (p)')
    plt.ylabel('Saliency')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'saliency_1d_primes.png'))
    
    # 2. Twists (average over primes -> axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(r0_map.mean(axis=0), label='Rank 0 (Average over primes)')
    plt.plot(r1_map.mean(axis=0), label='Rank 1 (Average over primes)')
    plt.plot(fake_map.mean(axis=0), label='Fake (Average over primes)', linestyle='--')
    plt.title('Saliency Projection onto Twist Index')
    plt.xlabel('Dirichlet character index')
    plt.ylabel('Saliency')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'saliency_1d_twists.png'))

if __name__ == "__main__":
    main()
