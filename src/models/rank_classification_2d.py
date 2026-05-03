import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR, RESULTS_DIR
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import csv
import random
import time
import os
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- 1. Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

LEARNING_RATE = 0.001
BATCH_SIZE = 256
EPOCHS = 100
TRAIN_VAL_SPLIT_RATIO = 0.8
IMAGE_SIZE = 100
NUM_CLASSES = 3  # Ranks 0, 1, 2

AP_CSV_PATH = os.path.join(DATA_DIR, 'ap_nocm.csv')

# --- 2. Data Loading ---

def get_ranks_from_csv(csv_path):
    """Read ranks from ap_nocm.csv. Row i (after header) corresponds to npy index i."""
    ranks = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            try:
                rank = int(row[1])
                ranks.append(rank)
            except (ValueError, IndexError):
                ranks.append(-1)  # Mark invalid
    return np.array(ranks, dtype=np.int64)


def get_rank_indices(csv_path):
    """Read ap_nocm.csv and return indices grouped by rank."""
    rank_indices = {0: [], 1: [], 2: []}
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found.")
        return None

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for i, row in enumerate(reader):
            try:
                rank = int(row[1])
                if rank in rank_indices:
                    rank_indices[rank].append(i)
            except (ValueError, IndexError):
                continue

    random.seed(42)
    for r in rank_indices:
        random.shuffle(rank_indices[r])

    return rank_indices


class RankDataset(Dataset):
    """Dataset that loads 2D twisted images and rank labels."""
    def __init__(self, data_mmap, ranks, valid_indices):
        self.data = data_mmap
        self.ranks = ranks
        self.valid_indices = valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        feature = np.array(self.data[real_idx])
        feature_tensor = torch.FloatTensor(feature).permute(2, 0, 1)  # (2, H, W)
        label = self.ranks[real_idx]
        return feature_tensor, label


class SaliencyDataset(Dataset):
    """Dataset for saliency computation (no labels needed)."""
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


# --- 3. Model Definition ---

class RankCNN(nn.Module):
    """2D CNN for rank classification. Uses MaxPool after conv layers,
    AdaptiveAvgPool (global average pool) in the classifier."""
    def __init__(self, num_classes=3):
        super(RankCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pool
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x


# --- 4. Saliency Computation ---

def compute_class_saliency(model, loader, device, class_idx, num_samples=None):
    """Compute average saliency w.r.t. a specific output class logit."""
    model.eval()
    avg_saliency = None
    samples_processed = 0

    for batch in loader:
        batch = batch.to(device)
        batch.requires_grad = True

        outputs = model(batch)
        score = outputs[:, class_idx].sum()
        model.zero_grad()
        score.backward()

        saliency = batch.grad.data.abs()
        batch_avg = saliency.sum(dim=0).cpu()

        if avg_saliency is None:
            avg_saliency = batch_avg
        else:
            avg_saliency += batch_avg

        samples_processed += len(batch)
        if num_samples is not None and samples_processed >= num_samples:
            break

    return (avg_saliency / samples_processed).numpy()


# --- 5. Plotting ---

def plot_enhanced_heatmap(map_data, title, filepath, image_size, cmap='hot'):
    std_across_twists = map_data.std(axis=1)
    mean_across_twists = map_data.mean(axis=1)
    std_across_primes = map_data.std(axis=0)
    mean_across_primes = map_data.mean(axis=0)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(f'{title} (N={image_size})', fontsize=16, y=0.98)

    vmax = np.percentile(np.abs(map_data), 99.5)
    vmin = -vmax if cmap == 'coolwarm' else 0

    im = ax.imshow(map_data, cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax)
    ax.set_ylabel('Primes ($p$)', fontsize=12)
    ax.set_xlabel('Twists ($\\chi$)', fontsize=12)

    divider = make_axes_locatable(ax)

    # Marginal: projection onto primes (right side)
    ax_prime = divider.append_axes("right", size="20%", pad=0.1)
    ax_prime.plot(mean_across_twists, range(image_size), color='red', label='Mean')
    ax_prime.fill_betweenx(range(image_size),
                           mean_across_twists - std_across_twists,
                           mean_across_twists + std_across_twists,
                           color='red', alpha=0.2, label='Std')
    ax_prime.invert_yaxis()
    ax_prime.set_ylim(image_size - 0.5, -0.5)
    ax_prime.margins(y=0)
    ax_prime.set_xlabel('Abs Grad', fontsize=10)
    ax_prime.set_yticks([])
    ax_prime.grid(True, alpha=0.3)
    ax_prime.legend(loc='upper right', fontsize=8)

    # Colorbar to the right of the marginal plot
    ax_cbar = divider.append_axes("right", size="3%", pad=0.15)
    plt.colorbar(im, cax=ax_cbar)

    # Marginal: projection onto twists (top side)
    ax_twist = divider.append_axes("top", size="20%", pad=0.1)
    ax_twist.plot(range(image_size), mean_across_primes, color='blue', label='Mean')
    ax_twist.fill_between(range(image_size),
                          mean_across_primes - std_across_primes,
                          mean_across_primes + std_across_primes,
                          color='blue', alpha=0.2, label='Std')
    ax_twist.set_xlim(-0.5, image_size - 0.5)
    ax_twist.margins(x=0)
    ax_twist.set_ylabel('Abs Grad', fontsize=10)
    ax_twist.set_xticks([])
    ax_twist.grid(True, alpha=0.3)
    ax_twist.legend(loc='upper right', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


# --- 6. Main ---

def main():
    parser = argparse.ArgumentParser(description='Train 2D CNN to predict elliptic curve rank and generate saliency maps.')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--image_size', type=int, default=IMAGE_SIZE)
    args = parser.parse_args()

    image_size = args.image_size
    real_data_path = os.path.join(DATA_DIR, f'combined_twisted_arrays_{image_size}.npy')

    # Load data
    if not os.path.exists(real_data_path):
        print(f"Error: {real_data_path} not found.")
        return
    if not os.path.exists(AP_CSV_PATH):
        print(f"Error: {AP_CSV_PATH} not found.")
        return

    print("Loading data...")
    data_mmap = np.load(real_data_path, mmap_mode='r')
    ranks = get_ranks_from_csv(AP_CSV_PATH)

    print(f"Data shape: {data_mmap.shape}")
    print(f"Ranks loaded: {len(ranks)}")

    # Filter to valid ranks (0, 1, 2)
    valid_indices = [i for i in range(min(len(data_mmap), len(ranks))) if ranks[i] in (0, 1, 2)]
    print(f"Valid samples (rank 0/1/2): {len(valid_indices)}")

    # Print rank distribution
    rank_counts = np.bincount([ranks[i] for i in valid_indices], minlength=3)
    for r in range(NUM_CLASSES):
        print(f"  Rank {r}: {rank_counts[r]} samples")

    # Create dataset and split
    dataset = RankDataset(data_mmap, ranks, valid_indices)

    g = torch.Generator().manual_seed(42)
    train_size = int(TRAIN_VAL_SPLIT_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=g)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=8, pin_memory=True, persistent_workers=True)

    print(f"Training samples: {train_size}, Validation samples: {val_size}")

    # Model, loss, optimizer
    model = RankCNN(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("\nModel Architecture:")
    print(model)

    best_val_acc = 0.0
    best_model_state = None

    # =====================
    # Training loop
    # =====================
    print("\nStarting training...")
    print("-" * 110)
    print(f"{'Epoch':^7} | {'Loss':^8} | {'Train Acc':^10} | {'Val Acc':^8} | "
          f"{'Val Acc R0':^10} | {'Val Acc R1':^10} | {'Val Acc R2':^10} | {'Time':^8}")
    print("-" * 110)

    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # --- Training ---
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for features, labels in train_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total

        # --- Validation ---
        model.eval()
        val_correct = 0
        val_total = 0
        per_class_correct = [0] * NUM_CLASSES
        per_class_total = [0] * NUM_CLASSES

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                outputs = model(features)
                preds = outputs.argmax(1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                for c in range(NUM_CLASSES):
                    mask = labels == c
                    per_class_correct[c] += (preds[mask] == labels[mask]).sum().item()
                    per_class_total[c] += mask.sum().item()

        val_acc = val_correct / val_total
        per_class_acc = [per_class_correct[c] / max(per_class_total[c], 1) for c in range(NUM_CLASSES)]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_time = time.time() - epoch_start

        avg_loss = running_loss / len(train_loader)
        print(f"{epoch+1:^7} | {avg_loss:^8.4f} | {train_acc:^10.4f} | {val_acc:^8.4f} | "
              f"{per_class_acc[0]:^10.4f} | {per_class_acc[1]:^10.4f} | {per_class_acc[2]:^10.4f} | {epoch_time:^8.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"  -> New best model (val acc: {best_val_acc:.4f})")

    total_time = time.time() - start_time
    print("-" * 110)
    print(f"\nTraining finished in {total_time:.1f}s. Best val acc: {best_val_acc:.4f}")

    # =====================
    # Final Evaluation
    # =====================
    print("\n" + "=" * 60)
    print("FINAL VALIDATION SET EVALUATION")
    print("=" * 60)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            outputs = model(features)
            preds = outputs.argmax(1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Confusion matrix
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    for t, p in zip(all_labels, all_preds):
        confusion[t, p] += 1

    print("\nConfusion Matrix:")
    print("              Predicted")
    print("            ", end="")
    for c in range(NUM_CLASSES):
        print(f"  R{c:d}  ", end="")
    print()
    for t in range(NUM_CLASSES):
        print(f"Actual R{t:d}  ", end="")
        for p in range(NUM_CLASSES):
            print(f"{confusion[t, p]:5d} ", end="")
        print()

    # Per-class metrics
    print(f"\nOverall accuracy: {(all_preds == all_labels).float().mean():.4f}")
    print("\nPer-class metrics:")
    for c in range(NUM_CLASSES):
        tp = confusion[c, c].item()
        total_actual = confusion[c, :].sum().item()
        total_pred = confusion[:, c].sum().item()
        precision = tp / max(total_pred, 1)
        recall = tp / max(total_actual, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-7)
        print(f"  Rank {c}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f} "
              f"(N_actual={total_actual}, N_pred={total_pred})")

    # =====================
    # Saliency Maps
    # =====================
    print("\n" + "=" * 60)
    print("GENERATING SALIENCY MAPS")
    print("=" * 60)

    rank_indices = get_rank_indices(AP_CSV_PATH)
    if rank_indices is None:
        print("Cannot generate saliency maps without ap_nocm.csv.")
        return

    for r in range(NUM_CLASSES):
        print(f"  Rank {r}: {len(rank_indices[r])} curves")

    output_dir = f'saliency_maps_rank_{image_size}'
    os.makedirs(output_dir, exist_ok=True)

    # Compute saliency for each rank group w.r.t. each class logit
    saliency_results = {}

    for rank in range(NUM_CLASSES):
        indices = rank_indices[rank]
        if not indices:
            print(f"No Rank {rank} curves found, skipping.")
            continue

        ds = SaliencyDataset(real_data_path, indices=indices)
        loader = DataLoader(ds, batch_size=256, shuffle=False)

        for target_class in range(NUM_CLASSES):
            key = f'rank{rank}_wrt_class{target_class}'
            print(f"Computing saliency for Rank {rank} curves w.r.t. class {target_class} logit ({len(ds)} samples)...")
            sal = compute_class_saliency(model, loader, DEVICE, target_class)
            saliency_results[key] = sal
            np.save(os.path.join(output_dir, f'{key}_saliency_avg.npy'), sal)

    # Compute combined saliency: weighted average of per-rank saliencies
    # (each curve backpropagated w.r.t. its own rank's logit)
    rank_counts = [len(rank_indices[r]) for r in range(NUM_CLASSES)]
    total = sum(rank_counts)
    combined_sal = None
    for r in range(NUM_CLASSES):
        key = f'rank{r}_wrt_class{r}'
        if key in saliency_results and rank_counts[r] > 0:
            weighted = saliency_results[key] * (rank_counts[r] / total)
            combined_sal = weighted if combined_sal is None else combined_sal + weighted
    if combined_sal is not None:
        saliency_results['all_combined'] = combined_sal
        np.save(os.path.join(output_dir, os.path.join(RESULTS_DIR, 'all_combined_saliency_avg.npy')), combined_sal)
        print(f"Combined saliency computed (weighted by rank counts: {rank_counts})")

    # Generate plots
    print("\nGenerating saliency heatmaps...")

    channel_configs = [(0, 'real_channel'), (1, 'imag_channel'), (None, 'average')]

    # Per-rank plots: rank R curves, gradient w.r.t. class R logit
    for rank in range(NUM_CLASSES):
        key_own = f'rank{rank}_wrt_class{rank}'
        if key_own not in saliency_results:
            continue

        sal = saliency_results[key_own]

        for ch_idx, ch_name in channel_configs:
            if ch_idx is not None:
                sal_map = sal[ch_idx]
            else:
                sal_map = sal.mean(axis=0)

            ch_label = ch_name.replace('_', ' ').title()
            title = f'[{ch_label}] Rank {rank} Saliency (w.r.t. class {rank})'
            fname = os.path.join(output_dir, f'enhanced_marginal_rank{rank}_{ch_name}.png')
            plot_enhanced_heatmap(sal_map, title, fname, image_size, 'hot')

    # All-ranks-combined plot (each curve w.r.t. its own rank)
    if 'all_combined' in saliency_results:
        sal = saliency_results['all_combined']
        for ch_idx, ch_name in channel_configs:
            sal_map = sal[ch_idx] if ch_idx is not None else sal.mean(axis=0)
            ch_label = ch_name.replace('_', ' ').title()
            title = f'[{ch_label}] All Curves Combined Saliency'
            fname = os.path.join(output_dir, f'enhanced_marginal_all_combined_{ch_name}.png')
            plot_enhanced_heatmap(sal_map, title, fname, image_size, 'hot')

    # Difference maps between rank pairs
    for r_a, r_b in [(0, 1), (0, 2), (1, 2)]:
        key_a = f'rank{r_a}_wrt_class{r_a}'
        key_b = f'rank{r_b}_wrt_class{r_b}'
        if key_a not in saliency_results or key_b not in saliency_results:
            continue

        for ch_idx, ch_name in channel_configs:
            if ch_idx is not None:
                diff_map = saliency_results[key_a][ch_idx] - saliency_results[key_b][ch_idx]
            else:
                diff_map = saliency_results[key_a].mean(axis=0) - saliency_results[key_b].mean(axis=0)

            ch_label = ch_name.replace('_', ' ').title()
            title = f'[{ch_label}] Rank {r_a} - Rank {r_b} Saliency Diff'
            fname = os.path.join(output_dir, f'enhanced_marginal_diff_r{r_a}_r{r_b}_{ch_name}.png')
            plot_enhanced_heatmap(diff_map, title, fname, image_size, 'coolwarm')

    print(f"\nAll saliency plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
