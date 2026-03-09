import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
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
    def __init__(self, data_path, num_samples=1000, offset=0):
        self.data = np.load(data_path, mmap_mode='r')
        self.num_samples = min(num_samples, len(self.data) - offset)
        self.offset = offset

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        feature = np.array(self.data[idx + self.offset])
        feature_tensor = torch.FloatTensor(feature).permute(2, 0, 1)
        return feature_tensor

def compute_average_saliency(model, loader, device, num_samples):
    model.eval()
    avg_saliency = None
    
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        batch.requires_grad = True
        
        outputs = model(batch)
        # We want the gradient of the score (logit) with respect to the input
        # For binary classification, we can just use the sum of outputs
        score = outputs.sum()
        model.zero_grad()
        score.backward()
        
        # Saliency is the absolute value of the gradient
        saliency = batch.grad.data.abs()
        
        # Aggregate across the batch and the non-constant channels
        # Shape of saliency is (Batch, Channels, Height, Width)
        batch_avg = saliency.sum(dim=0)
        
        if avg_saliency is None:
            avg_saliency = batch_avg
        else:
            avg_saliency += batch_avg
            
        if (i+1) * loader.batch_size >= num_samples:
            break
            
    return (avg_saliency / num_samples).cpu().numpy()

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMAGE_SIZE = 100 # Adjust if necessary
    CHECKPOINT_PATH = f'L_function_classifier_{IMAGE_SIZE}_checkpoint.pth'
    REAL_DATA_PATH = f'combined_twisted_arrays_{IMAGE_SIZE}.npy'
    FAKE_DATA_PATH = f'combined_twisted_arrays_fake_{IMAGE_SIZE}.npy'
    
    # NOTE: You'll need to provide logic to separate Rank 0 and Rank 1 
    # if they are mixed in the REAL_DATA_PATH. 
    # For now, I'll generate for the first 1000 'Real' and first 1000 'Fake'.
    
    model = LFunctionCNN().to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {CHECKPOINT_PATH}")
    else:
        print(f"Checkpoint {CHECKPOINT_PATH} not found!")
        return

    # Real Saliency (First 1000)
    real_ds = SaliencyDataset(REAL_DATA_PATH, num_samples=1000)
    real_loader = DataLoader(real_ds, batch_size=32, shuffle=False)
    print("Computing Real saliency...")
    real_saliency = compute_average_saliency(model, real_loader, DEVICE, 1000)
    
    # Fake Saliency (First 1000)
    fake_ds = SaliencyDataset(FAKE_DATA_PATH, num_samples=1000)
    fake_loader = DataLoader(fake_ds, batch_size=32, shuffle=False)
    print("Computing Fake saliency...")
    fake_saliency = compute_average_saliency(model, fake_loader, DEVICE, 1000)
    
    # Create output directory
    output_dir = 'saliency_maps'
    os.makedirs(output_dir, exist_ok=True)

    # Save raw data
    np.save(os.path.join(output_dir, 'real_saliency_avg.npy'), real_saliency)
    np.save(os.path.join(output_dir, 'fake_saliency_avg.npy'), fake_saliency)
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # We have 2 channels (Real part and Imaginary part of twisted a_p)
    # Let's average them for the final 2D map
    real_map = real_saliency.mean(axis=0)
    fake_map = fake_saliency.mean(axis=0)
    diff_map = real_map - fake_map
    
    im0 = axes[0].imshow(real_map, cmap='hot')
    axes[0].set_title('Average Real Saliency')
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(fake_map, cmap='hot')
    axes[1].set_title('Average Fake Saliency')
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(diff_map, cmap='coolwarm')
    axes[2].set_title('Real - Fake Saliency')
    plt.colorbar(im2, ax=axes[2])
    
    plt.savefig(os.path.join(output_dir, 'saliency_maps.png'), dpi=300)
    print(f"Saliency maps saved to {output_dir}/saliency_maps.png")

    # 1D Projections (Important for mathematical patterns)
    plt.figure(figsize=(10, 6))
    plt.plot(real_map.mean(axis=0), label='Real (Average over twists)')
    plt.plot(fake_map.mean(axis=0), label='Fake (Average over twists)')
    plt.title('Saliency Projection onto Prime Index')
    plt.xlabel('Prime index (p)')
    plt.ylabel('Saliency')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'saliency_1d_primes.png'))
    
    plt.figure(figsize=(10, 6))
    plt.plot(real_map.mean(axis=1), label='Real (Average over primes)')
    plt.plot(fake_map.mean(axis=1), label='Fake (Average over primes)')
    plt.title('Saliency Projection onto Twist Index')
    plt.xlabel('Dirichlet character index')
    plt.ylabel('Saliency')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'saliency_1d_twists.png'))

if __name__ == "__main__":
    main()
