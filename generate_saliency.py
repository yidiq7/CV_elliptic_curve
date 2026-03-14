import os
import csv
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
            self.indices = list(range(len(self.data)))
        else:
            self.indices = indices
            
        if num_samples is not None:
            self.indices = self.indices[:num_samples]

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
        
        # Saliency is the absolute value of the gradient
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

def get_rank_indices(csv_path):
    rank0_indices = []
    rank1_indices = []
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Cannot separate ranks.")
        return None, None
        
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader) # Skip header
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
        real_saliency = compute_average_saliency(model, real_loader, DEVICE, NUM_SAMPLES)
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

        # Real Saliency (Full Dataset)
        real_ds = SaliencyDataset(REAL_DATA_PATH, num_samples=NUM_SAMPLES)
        real_loader = DataLoader(real_ds, batch_size=256, shuffle=False)
        print(f"Computing Real saliency (Full Dataset, {len(real_ds)} samples)...")
        real_saliency = compute_average_saliency(model, real_loader, DEVICE, NUM_SAMPLES)
    
    # Fake Saliency
    fake_ds = SaliencyDataset(FAKE_DATA_PATH, num_samples=NUM_SAMPLES)
    fake_loader = DataLoader(fake_ds, batch_size=256, shuffle=False)
    print(f"Computing Fake saliency ({len(fake_ds)} samples)...")
    fake_saliency = compute_average_saliency(model, fake_loader, DEVICE, NUM_SAMPLES)
    
    # Create output directory
    output_dir = f'saliency_maps_{IMAGE_SIZE}'
    os.makedirs(output_dir, exist_ok=True)

    # Save raw data
    np.save(os.path.join(output_dir, 'real_saliency_avg.npy'), real_saliency)
    np.save(os.path.join(output_dir, 'fake_saliency_avg.npy'), fake_saliency)
    
    if rank0_indices is not None:
        np.save(os.path.join(output_dir, 'rank0_saliency_avg.npy'), rank0_saliency)
        np.save(os.path.join(output_dir, 'rank1_saliency_avg.npy'), rank1_saliency)
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    def plot_enhanced_heatmap(map_data, title, filename_suffix, cmap='hot'):
        # Axis 0 (Rows) = Primes, Axis 1 (Columns) = Twists
        std_across_twists = map_data.std(axis=1)
        mean_across_twists = map_data.mean(axis=1)
        
        std_across_primes = map_data.std(axis=0)
        mean_across_primes = map_data.mean(axis=0)

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.suptitle(f'{title} (N={IMAGE_SIZE})', fontsize=16, y=0.98)
        
        # Clip to a high percentile to prevent a single pixel from washing out the colormap
        vmax = np.percentile(np.abs(map_data), 99.5)
        vmin = -vmax if cmap == 'coolwarm' else 0
        
        im = ax.imshow(map_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_ylabel('Primes ($p$)', fontsize=12)
        ax.set_xlabel('Twists ($\chi$)', fontsize=12)

        # Create dividers for marginal plots
        divider = make_axes_locatable(ax)
        
        # Marginal plot for Primes (Right side) - Projection onto Primes
        ax_prime = divider.append_axes("right", size="20%", pad=0.1)
        ax_prime.plot(mean_across_twists, range(IMAGE_SIZE), color='red', label='Mean Saliency')
        ax_prime.fill_betweenx(range(IMAGE_SIZE), 
                               mean_across_twists - std_across_twists, 
                               mean_across_twists + std_across_twists, 
                               color='red', alpha=0.2, label='Std Dev')
        ax_prime.invert_yaxis()  # Match image coordinates
        ax_prime.set_ylim(IMAGE_SIZE - 0.5, -0.5) 
        ax_prime.margins(y=0) # Remove blanks on top/bottom
        ax_prime.set_xlabel('Abs Grad', fontsize=10)
        ax_prime.set_yticks([])
        ax_prime.grid(True, alpha=0.3)
        ax_prime.legend(loc='upper right', fontsize=8)
        
        # Marginal plot for Twists (Top side) - Projection onto Twists
        ax_twist = divider.append_axes("top", size="20%", pad=0.1)
        ax_twist.plot(range(IMAGE_SIZE), mean_across_primes, color='blue', label='Mean Saliency')
        ax_twist.fill_between(range(IMAGE_SIZE), 
                              mean_across_primes - std_across_primes, 
                              mean_across_primes + std_across_primes, 
                              color='blue', alpha=0.2, label='Std Dev')
        ax_twist.set_xlim(-0.5, IMAGE_SIZE - 0.5) 
        ax_twist.margins(x=0) # Remove blanks on left/right
        ax_twist.set_ylabel('Abs Grad', fontsize=10)
        ax_twist.set_xticks([])
        ax_twist.grid(True, alpha=0.3)
        ax_twist.legend(loc='upper right', fontsize=8)

        # Use rect to ensure suptitle is not clipped or overlapped
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f'enhanced_marginal_{filename_suffix}.png'), dpi=300)
        plt.close()

    print("Generating enhanced marginal heatmaps for Real Channel, Imag Channel, and Average...")
    
    # Generate plots for Real Channel (index 0), Imag Channel (index 1), and Average
    for channel_idx, channel_name in [(0, 'real_channel'), (1, 'imag_channel'), (None, 'average')]:
        if channel_idx is not None:
            c_real_map = real_saliency[channel_idx]
            c_fake_map = fake_saliency[channel_idx]
            c_title_prefix = f"[{channel_name.replace('_', ' ').title()}]"
        else:
            c_real_map = real_saliency.mean(axis=0)
            c_fake_map = fake_saliency.mean(axis=0)
            c_title_prefix = "[Average Channels]"
            
        c_diff_map = c_real_map - c_fake_map

        plot_enhanced_heatmap(c_real_map, f'{c_title_prefix} Genuine Curve Saliency', f'genuine_{channel_name}', 'hot')
        plot_enhanced_heatmap(c_fake_map, f'{c_title_prefix} Fake Curve Saliency', f'fake_{channel_name}', 'hot')
        plot_enhanced_heatmap(c_diff_map, f'{c_title_prefix} Genuine - Fake Diff', f'diff_{channel_name}', 'coolwarm')
        
        if rank0_indices is not None:
            if channel_idx is not None:
                c_r0_map = rank0_saliency[channel_idx]
                c_r1_map = rank1_saliency[channel_idx]
            else:
                c_r0_map = rank0_saliency.mean(axis=0)
                c_r1_map = rank1_saliency.mean(axis=0)
                
            plot_enhanced_heatmap(c_r0_map, f'{c_title_prefix} Rank 0 Curve Saliency', f'rank0_{channel_name}', 'hot')
            plot_enhanced_heatmap(c_r1_map, f'{c_title_prefix} Rank 1 Curve Saliency', f'rank1_{channel_name}', 'hot')

    print(f"All plots saved to {output_dir}/")

if __name__ == "__main__":
    main()
