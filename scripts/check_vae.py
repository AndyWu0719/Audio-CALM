import torch, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from glob import glob
from models.modeling_vae import AcousticVAE
from eval.eval_calm import Vocoder
import torchaudio  # 提前 import，避免循环 import

device = "cuda:0"
vae_ckpt = "outputs/checkpoints/vae_4x_128_5e-4/checkpoint-17350"
latents_root = "data/latents/train/LibriTTS_R/train-clean-100"

vae = AcousticVAE.from_pretrained(vae_ckpt).to(device)
vae.eval().requires_grad_(False)
vocoder = Vocoder(device)

pt_files = sorted(glob(os.path.join(latents_root, "**", "*.pt"), recursive=True))
os.makedirs("tmp_vae_wavs", exist_ok=True)

for i, pt in enumerate(pt_files[:10]):
    payload = torch.load(pt, map_location="cpu")
    z = payload.get("latent", payload)
    if z.dim() == 2 and z.shape[0] in (64, 80, 128, 192):
        z = z.transpose(0, 1)              # [T, C]
    z = z.unsqueeze(0).to(device)          # [1, T, C]
    z = z.transpose(1, 2).to(vae.dtype)    # [1, C, T]

    mel = vae.decode(z)                    # [1, 80, T']
    wav = vocoder.decode(mel).cpu()        # 可能是 [T'] 或 [1, T'] 或 [1, 1, T']

    # 规范成 [channels, time]
    if wav.dim() == 3:
        wav = wav.squeeze(0)               # [1, T]
    if wav.dim() == 1:
        wav_to_save = wav.unsqueeze(0)     # [1, T]
    elif wav.dim() == 2:
        wav_to_save = wav                  # [C, T]
    else:
        raise RuntimeError(f"Unexpected wav shape: {wav.shape}")

    torchaudio.save(f"tmp_vae_wavs/{i}.wav", wav_to_save, 16000)
    print("saved", i)