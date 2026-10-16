"""Fetch only required, revision-pinned official weights into a portable directory."""
import os
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
BASE_PATTERNS = ['model_index.json', 'scheduler/*.json', 'text_encoder/*.json',
                 'text_encoder/*.fp16.safetensors', 'tokenizer/*', 'unet/*.json',
                 'unet/*.fp16.safetensors', 'vae/*.json', 'vae/*.fp16.safetensors']
MODELS = {
 'sdxl': ('stabilityai/stable-diffusion-xl-base-1.0', '462165984030d82259a11f4367a4eed129e94a7b', BASE_PATTERNS + ['text_encoder_2/*.json','text_encoder_2/*.fp16.safetensors','tokenizer_2/*']),
 'sd15': ('stable-diffusion-v1-5/stable-diffusion-v1-5', '451f4fe16113bff5a5d2269ed5ad43b0592e9a14', BASE_PATTERNS),
 'controlnet': ('lllyasviel/control_v11p_sd15_canny','115a470d547982438f70198e353a921996e2e819',['config.json','diffusion_pytorch_model.fp16.safetensors']),
 'adapter': ('TencentARC/t2iadapter_canny_sd15v2','a18baf4f0ff002f34dc2f19c4b93fe00d4cce9ee',['config.json','diffusion_pytorch_model.bin']),
 'lcm': ('latent-consistency/lcm-lora-sdxl','a18548dd4956b174ec5b0d78d340c8dae0a129cd',['pytorch_lora_weights.safetensors']),
 'lightning': ('ByteDance/SDXL-Lightning','c9a24f48e1c025556787b0c58dd67a091ece2e44',[f'sdxl_lightning_{n}step_unet.safetensors' for n in [2,4,8]]),
 'clip': ('openai/clip-vit-base-patch32','3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',['*.json','*.txt','pytorch_model.bin']),
}

def fetch(item):
 key, (repo, rev, patterns) = item
 print('DOWNLOAD', key, flush=True)
 snapshot_download(repo, revision=rev, allow_patterns=patterns, local_dir=ROOT/'models'/key, max_workers=3)
 print('READY', key, flush=True)

if __name__ == '__main__':
 (ROOT/'configs/model_revisions.json').write_text(json.dumps(MODELS,indent=2))
 with ThreadPoolExecutor(max_workers=3) as pool:
  list(pool.map(fetch, MODELS.items()))
