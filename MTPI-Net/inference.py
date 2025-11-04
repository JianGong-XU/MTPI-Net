import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from MTPI_Net import MTPI_Net

# ================================================================
# Inference Dataset
# ================================================================
class InferenceDataset(Dataset):
    """
    Inference dataset for MTPI-Net.
    Each item should include:
        - tar  : target frame tensor (C,H,W)
        - prev : previous temporal frame tensor (C,H,W)
        - next : next temporal frame tensor (C,H,W)
    Expected channel order: [C11, C22, Re(C12), Im(C12), H, A, α]
    """
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        return {
            "tar": torch.tensor(item["tar"], dtype=torch.float32),
            "prev": torch.tensor(item["prev"], dtype=torch.float32),
            "next": torch.tensor(item["next"], dtype=torch.float32)
        }

# ================================================================
# Inference Function
# ================================================================
def inference_mtpi(
    model_path,
    data_list,
    save_dir="./results",
    device="cuda" if torch.cuda.is_available() else "cpu",
    patch_size=None,
):
    os.makedirs(save_dir, exist_ok=True)

    # Load model
    model = MTPI_Net(in_ch=7, base_ch=64, n_ddcr=3, spp_filter="FrGT").to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    print(f"✅ Loaded model from {model_path}")

    # Prepare data
    dataset = InferenceDataset(data_list)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"🚀 Running inference on {len(dataset)} samples...")
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            I_tar = batch["tar"].to(device)
            I_prev = batch["prev"].to(device)
            I_next = batch["next"].to(device)

            despeckled = model(I_tar, I_prev, I_next)
            despeckled_np = despeckled.squeeze(0).cpu().numpy()

            save_path = os.path.join(save_dir, f"despeckled_{idx:03d}.npy")
            np.save(save_path, despeckled_np)
            print(f"Saved: {save_path}")

    print(f"✅ Inference complete. Results stored in: {save_dir}")

# ================================================================
# Example Usage
# ================================================================
if __name__ == "__main__":
    # Example dummy input (replace with preprocessed Sentinel-1 data)
    H = W = 128
    example_data = [
        {
            "tar": np.random.randn(7, H, W),
            "prev": np.random.randn(7, H, W),
            "next": np.random.randn(7, H, W),
        },
        {
            "tar": np.random.randn(7, H, W),
            "prev": np.random.randn(7, H, W),
            "next": np.random.randn(7, H, W),
        }
    ]

    model_ckpt = "./checkpoints/mtpi_epoch_060.pth"
    inference_mtpi(model_ckpt, example_data, save_dir="./results")

