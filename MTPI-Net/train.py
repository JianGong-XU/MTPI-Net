import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from MTPI_Net import MTPI_Net
from CO_Loss import CollaborativeLoss, compute_temporal_pairs

# ================================================================
# Dataset Definition
# ================================================================
class SentinelDualPolDataset(Dataset):
    """
    Custom dataset for real Sentinel-1 dual-pol despeckling.
    Each sample contains:
      - tar  : target tensor (C,H,W)
      - prev : previous temporal frame (C,H,W)
      - next : next temporal frame (C,H,W)
      - mask : change detection mask (1,H,W)
    Expected input channels: 7 (C11, C22, Re(C12), Im(C12), H, A, α)
    """
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        # item: dictionary or preloaded tensor triplet
        return {
            "tar": torch.tensor(item["tar"], dtype=torch.float32),
            "prev": torch.tensor(item["prev"], dtype=torch.float32),
            "next": torch.tensor(item["next"], dtype=torch.float32),
            "mask": torch.tensor(item.get("mask", 1.0), dtype=torch.float32).unsqueeze(0)
        }

# ================================================================
# Training Function
# ================================================================
def train_mtpi(
    train_dataset,
    val_dataset=None,
    batch_size=8,
    epochs=60,
    lr=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    ckpt_dir="./checkpoints",
):
    # Model, loss, optimizer
    model = MTPI_Net(in_ch=7, base_ch=64, n_ddcr=3, spp_filter="FrGT").to(device)
    criterion = CollaborativeLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    os.makedirs(ckpt_dir, exist_ok=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    print(f"🚀 Training MTPI-Net on {device}")
    print(f"Dataset size: {len(train_dataset)} samples | Batch size: {batch_size}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", ncols=100)
        for batch in pbar:
            I_tar = batch["tar"].to(device)
            I_prev = batch["prev"].to(device)
            I_next = batch["next"].to(device)
            mask = batch["mask"].to(device)

            # Forward
            despeckled = model(I_tar, I_prev, I_next)

            # Temporal pairs (for optional L_t)
            temporal_pairs = compute_temporal_pairs([I_prev, I_tar, I_next])

            # Compute collaborative loss
            loss, parts = criterion(despeckled, I_tar, mask=mask, temporal_pairs=temporal_pairs)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({
                "Total": f"{loss.item():.4f}",
                "L_n": f"{parts['L_n']:.3f}",
                "L_e": f"{parts['L_e']:.3f}",
                "L_t": f"{parts['L_t']:.3f}",
            })

        scheduler.step()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1:03d} | Avg Loss: {avg_loss:.6f}")

        # Save checkpoint
        ckpt_path = os.path.join(ckpt_dir, f"mtpi_epoch_{epoch+1:03d}.pth")
        torch.save({
            "epoch": epoch+1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, ckpt_path)

    print("✅ Training complete. Checkpoints saved in:", ckpt_dir)
    return model

# ================================================================
# Example Usage
# ================================================================
if __name__ == "__main__":
    # Example dummy dataset
    num_samples = 10
    data_list = []
    for _ in range(num_samples):
        H = W = 128
        data_list.append({
            "tar": torch.randn(7, H, W).numpy(),
            "prev": torch.randn(7, H, W).numpy(),
            "next": torch.randn(7, H, W).numpy(),
            "mask": torch.ones(H, W).numpy()
        })

    train_set = SentinelDualPolDataset(data_list)
    train_mtpi(train_set, epochs=2, batch_size=2)

