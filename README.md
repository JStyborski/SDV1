# Stable Diffusion

This repository was forked from the CompVis Stable Diffusion repo, https://github.com/CompVis/stable-diffusion, commit 21f890f (the latest commit, as of 15/10/2024).

I added multiple scripts without removing the scripts or capabilities of the original codebase. 
All new scripts have the "JS_" prefix. 
New capabilities are listed below.
- Deterministic img2img (both forward and reverse) as referenced in https://arxiv.org/abs/2105.05233 (Appendix F).
- VAE finetuning, copied from https://github.com/Leminhbinh0209/FinetuneVAE-SD and then significantly modified.
- Adversarial attack on LDM (https://arxiv.org/abs/2305.12683), copied from https://github.com/psyker-team/mist and then significantly modified.
- Textual inversion (https://arxiv.org/abs/2208.01618), copied from https://blog.paperspace.com/dreambooth-stable-diffusion-tutorial-part-2-textual-inversion/ and then significantly modified.
  
## Requirements

You can run all scripts by installing and activating a conda environment as below. 
All libraries are identical to the original repo, though I added a few new libraries to support my custom scripts.

```
conda env create -f environment.yaml
conda activate sdv1
```

## Stable Diffusion v1

Please refer to the original repo for information on models. 

I have not altered the model configuration. I have only modified a few original files to fix bugs.

### Weights

Please refer to the original repo for information on weights.

I use checkpoints for SD v1.5 which can be downloaded from various repos on the web.

### Usage Examples

Adversarial Attack:
```
CUDA_VISIBLE_DEVICES=0 python scripts/JS_mist.py --src_img_dir ../Datasets/MyFaveCat --tgt_img_path ../Datasets/Poison_Targets/target.png
```

Textual Inversion:
```
CUDA_VISIBLE_DEVICES=0 python scripts/JS_text_inversion.py --src_img_dir ../Datasets/MyFaveCat --new_token MyFaveCatILoveHim --concept_type object
```

Text to Image:
```
CUDA_VISIBLE_DEVICES=0 python scripts/JS_txt2img.py --prompt A painting of MyFaveCatILoveHim.
```

Finetune VAE:
```
CUDA_VISIBLE_DEVICES=0 python scripts/JS_vae_finetune.py --src_img_dir ../Datasets/WikiArt_Imgs --log_prefix AllWikiArt
```

Image to Image:
```
CUDA_VISIBLE_DEVICES=0 python scripts/JS_img2img.py --src_img_dir ../Datasets/MyFaveCat --prompt A painting of my favorite cat.
```




