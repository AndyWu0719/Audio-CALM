import os
import sys
import torch
import torchaudio
import matplotlib.pyplot as plt
import argparse
import gradio as gr
from rich.console import Console

if not hasattr(torchaudio, "list_audio_backends"):
    def _list_audio_backends():
        return ["soundfile"]
    torchaudio.list_audio_backends = _list_audio_backends

console = Console()
sys.path.append(os.getcwd())

from models.modeling_vae import AcousticVAE
from preprocess.prepare_mel import MelExtractor

# 尝试导入 HiFiGAN (保持原有兼容性)
try:
    from speechbrain.inference.vocoders import HifiGAN
except ImportError:
    try:
        from speechbrain.inference.vocoders import HIFIGAN as HifiGAN
    except ImportError:
        import speechbrain.inference.vocoders as mod
        available = [x for x in dir(mod) if 'GAN' in x and 'Base' not in x]
        if available:
            HifiGAN = getattr(mod, available[0])
        else:
            raise ImportError("Could not find HifiGAN class!")

def load_models(checkpoint_path, device):
    console.print(f"[bold]Loading VAE from {checkpoint_path}...[/bold]")
    vae_model = AcousticVAE.from_pretrained(checkpoint_path).to(device)
    vae_model.eval()

    console.print("[bold]Loading HiFi-GAN...[/bold]")
    hifi_gan = HifiGAN.from_hparams(
        source="speechbrain/tts-hifigan-libritts-16kHz", 
        savedir="./outputs/eval/tmpdir_vocoder",
        run_opts={"device": device}
    )
    mel_extractor = MelExtractor().to(device)
    return vae_model, hifi_gan, mel_extractor

def run_inference(audio_path, vae_model, hifi_gan, extractor, device):
    wav, sr = torchaudio.load(audio_path)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    
    # Normalize
    wav = wav / (torch.max(torch.abs(wav)) + 1e-8) * 0.95
    wav = wav.to(device)
    
    with torch.no_grad():
        gt_mel = extractor(wav)
        outputs = vae_model(gt_mel)
        recon_mel = outputs['recon_mel']
        
        # Vocoder
        wav_recon = hifi_gan.decode_batch(recon_mel)
        wav_oracle = hifi_gan.decode_batch(gt_mel)
        
    return wav.cpu(), wav_recon.cpu(), wav_oracle.cpu(), gt_mel, recon_mel

def create_demo(checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae_model, hifi_gan, extractor = load_models(checkpoint_path, device)

    def process_audio(audio_file):
        if audio_file is None: return None, None, None
        
        orig, recon, oracle, _, _ = run_inference(audio_file, vae_model, hifi_gan, extractor, device)
        
        # Save temp files for Gradio
        os.makedirs("outputs/demo_tmp", exist_ok=True)
        path_orig = "outputs/demo_tmp/orig.wav"
        path_recon = "outputs/demo_tmp/recon.wav"
        path_oracle = "outputs/demo_tmp/oracle.wav"
        
        torchaudio.save(path_orig, orig, 16000)
        torchaudio.save(path_recon, recon.squeeze(0), 16000)
        torchaudio.save(path_oracle, oracle.squeeze(0), 16000)
        
        return path_orig, path_oracle, path_recon

    with gr.Blocks(title="Audio VAE Demo") as demo:
        gr.Markdown("# Audio VAE Reconstruction Demo")
        with gr.Row():
            inp = gr.Audio(type="filepath", label="Input Audio")
        
        btn = gr.Button("Reconstruct")
        
        with gr.Row():
            out_orig = gr.Audio(label="Original (Resampled)")
            out_oracle = gr.Audio(label="Oracle (Mel -> HiFiGAN)")
            out_recon = gr.Audio(label="VAE Reconstructed")
            
        btn.click(process_audio, inputs=inp, outputs=[out_orig, out_oracle, out_recon])
    
    return demo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--audio_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs/eval_results_vae")
    parser.add_argument("--web_demo", action="store_true", help="Launch Gradio Web UI")
    args = parser.parse_args()
    
    if args.web_demo:
        demo = create_demo(args.checkpoint)
        demo.launch(server_name="0.0.0.0", share=True)
    else:
        # CLI Mode (Original Logic)
        if not args.audio_path:
            console.print("[red]Please provide --audio_path for CLI mode[/red]")
            return
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        os.makedirs(args.output_dir, exist_ok=True)
        vae_model, hifi_gan, extractor = load_models(args.checkpoint, device)
        
        console.print(f"Processing {args.audio_path}...")
        orig, recon, oracle, gt_mel, recon_mel = run_inference(args.audio_path, vae_model, hifi_gan, extractor, device)
        
        # Calc Metrics
        mse = torch.nn.functional.mse_loss(recon_mel, gt_mel).item()
        console.print(f"MSE Loss: {mse:.6f}")
        
        # Save
        torchaudio.save(os.path.join(args.output_dir, "vae_recon.wav"), recon.squeeze(0), 16000)
        console.print(f"[green]Saved to {args.output_dir}[/green]")

if __name__ == "__main__":
    main()