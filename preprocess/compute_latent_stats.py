import os, torch
from glob import glob
from tqdm import tqdm

latent_dir = "/data0/determined/users/andywu/Audio-CALM-v2/data/latents/train/LibriTTS_R"  # 改成你的训练 latent 根目录
max_files = None   # 可设为整数抽样
reduce_dim = True  # True 得到标量 mean/std；False 得到按维度 [D] 统计

files = sorted(glob(os.path.join(latent_dir, "**", "*.pt"), recursive=True))
if max_files: files = files[:max_files]
assert files, "No .pt found"

sum1 = None
sum2 = None
count = 0

for f in tqdm(files, desc="scanning"):
    payload = torch.load(f, map_location="cpu", weights_only=True)
    lat = payload.get("latent", payload) if isinstance(payload, dict) else payload  # (T,D) 或 (D,T)
    if lat.dim() == 2 and lat.shape[0] in (64, 80, 128, 192):  # 如果是 (D,T) 则转置
        lat = lat.transpose(0, 1)  # -> (T,D)
    lat = lat.float()
    if reduce_dim:
        lat = lat.contiguous().reshape(-1)  # 标量统计
    else:
        lat = lat.contiguous()  # (T,D) 按维统计

    if sum1 is None:
        sum1 = lat.sum(dim=0)
        sum2 = (lat * lat).sum(dim=0)
        count = lat.shape[0]
    else:
        sum1 += lat.sum(dim=0)
        sum2 += (lat * lat).sum(dim=0)
        count += lat.shape[0]

mean = sum1 / count
var = sum2 / count - mean * mean
std = torch.sqrt(var.clamp(min=1e-12))

if reduce_dim:
    print(f"latent_mean: {mean.item():.6f}")
    print(f"latent_std : {std.item():.6f}")
else:
    print("latent_mean shape:", mean.shape)
    print("latent_std  shape:", std.shape)
    torch.save({"mean": mean, "std": std}, "latent_stats.pt")
    print("saved to latent_stats.pt")