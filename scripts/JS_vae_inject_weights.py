# Copied almost entirely from https://gist.github.com/ProGamerGov/70061a08e3a2da6e9ed83e145ea24a70

import copy
import os
import torch

# Path to model and VAE files that you want to merge
vae_file_path = r'C:\Users\jeremy\Python_Projects\SDV1\vae_finetune\AllWikiArt_GN0p0887_Nom_256_10_8_16_0.0_0.0_0.0_0.002_0.0_False_False\last_model.pth'
model_file_path = r'C:\Users\jeremy\Python_Projects\SDV1\checkpoints\stable-diffusion-v1-5\v1-5-pruned-emaonly.ckpt'

# Name to use for new model file
new_model_name = r'C:\Users\jeremy\Python_Projects\SDV1\checkpoints\sdv15_finetuned_vae\test.ckpt'

# Load files
vae_model = torch.load(vae_file_path, map_location='cpu')
full_model = torch.load(model_file_path, map_location='cpu')

# Convert OrderedDict to Dict
_, fileExt = os.path.splitext(vae_file_path)
if fileExt == '.pth':
    vae_model = dict(vae_model)

for k, _ in full_model['state_dict'].items():
    if k.startswith('first_stage_model.'):
        if fileExt == '.pth':
            full_model['state_dict'][k] = copy.deepcopy(vae_model[k[18:]])
        else:
            full_model['state_dict'][k] = copy.deepcopy(vae_model['state_dict']['model.' + k[18:]])

# Save model with new VAE
torch.save(full_model, new_model_name)