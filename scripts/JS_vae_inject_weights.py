# Copied almost entirely from https://gist.github.com/ProGamerGov/70061a08e3a2da6e9ed83e145ea24a70

import copy
import torch

# Path to model and VAE files that you want to merge
vae_file_path = r'C:\Users\jeremy\Python_Projects\SDV1\vae_finetune\AllWikiArt_GN0p0887_imgsize(256)_epochs(10)_bs(16)_accum(1)_kl(0.0)_lpips(1.0)_lr(0.02)_ema(0.0)\lightning_logs\version_0\checkpoints\epoch=9-step=64540.ckpt'
model_file_path = r'C:\Users\jeremy\Python_Projects\SDV1\checkpoints\stable-diffusion-v1-5\v1-5-pruned-emaonly.ckpt'

# Name to use for new model file
new_model_name = r'C:\Users\jeremy\Python_Projects\SDV1\checkpoints\sdv15_finetuned_vae\v1-5-pruned-emaonly_AWA0887_LR2en2_Accum1_EMA0p0.ckpt'

# Load files
vae_model = torch.load(vae_file_path, map_location="cpu")
full_model = torch.load(model_file_path, map_location="cpu")

for k, _ in full_model['state_dict'].items():
    if k.startswith('first_stage_model.'):
        targKey = 'model.' + k[18:]
        full_model['state_dict'][k] = copy.deepcopy(vae_model["state_dict"][targKey])

# Save model with new VAE
torch.save(full_model, new_model_name)