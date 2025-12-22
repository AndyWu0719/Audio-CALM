import gradio as gr
import torch
import torchaudio
import numpy as np
import os
from transformers import AutoTokenizer
from models.modeling_calm import QwenCALM, QwenCALMConfig
from peft import PeftModel

# === Config ===
# 填入训练好的 checkpoint 路径
CHECKPOINT_PATH = "./outputs/calm-experiment-001/checkpoint-final"
QWEN_PATH = "/root/autodl-tmp/qwen2_7B_Instruct" 
VAE_PATH = "/root/autodl-tmp/checkpoints/audio_vae_4x"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading from {CHECKPOINT_PATH}...")

# Load Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
config = QwenCALMConfig(qwen_path=QWEN_PATH, vae_path=VAE_PATH, use_precomputed_latents=False)
model = QwenCALM(config)

# Load Weights
if os.path.exists(os.path.join(CHECKPOINT_PATH, "adapter_model.bin")):
    model.llm = PeftModel.from_pretrained(model.llm, CHECKPOINT_PATH)
    model.llm = model.llm.merge_and_unload()

# Load Projector
model.input_proj.load_state_dict(torch.load(os.path.join(CHECKPOINT_PATH, "input_proj.bin"), map_location="cpu"))
model.output_head.load_state_dict(torch.load(os.path.join(CHECKPOINT_PATH, "output_head.bin"), map_location="cpu"))

model.to(DEVICE).eval().to(torch.bfloat16)
model.vae.to(torch.float32)

def tts_fn(text):
    if not text: return None
    print(f"Generating TTS for: {text}")
    # 这里填入你的 generate_audio_from_text 逻辑
    # 模拟返回：生成 1秒静音
    return (16000, np.zeros(16000)) 

def asr_fn(audio_path):
    if not audio_path: return ""
    print(f"Transcribing: {audio_path}")
    # 这里填入你的 transcribe_audio 逻辑
    return "This is a placeholder ASR result."

# === UI ===
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎵 Audio-CALM Interactive Demo")
    
    with gr.Tab("🗣️ Text-to-Speech"):
        t_input = gr.Textbox(label="Input Text", value="Hello world.")
        t_btn = gr.Button("Generate", variant="primary")
        t_out = gr.Audio(label="Output Audio")
        t_btn.click(tts_fn, inputs=t_input, outputs=t_out)
        
    with gr.Tab("👂 ASR (Speech-to-Text)"):
        a_input = gr.Audio(label="Microphone / Upload", sources=["microphone", "upload"], type="filepath")
        a_btn = gr.Button("Transcribe", variant="primary")
        a_out = gr.Textbox(label="Transcription")
        a_btn.click(asr_fn, inputs=a_input, outputs=a_out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)