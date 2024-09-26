# Copied cell-by-cell from https://blog.paperspace.com/dreambooth-stable-diffusion-tutorial-part-2-textual-inversion/ and then modified extensively

import accelerate
import argparse
from distutils.util import strtobool
import math
import os
from PIL import Image
import random
import torch
import torch.utils.checkpoint
from torch.utils.data import Dataset
import torchvision.transforms as tvt
from tqdm.auto import tqdm

from omegaconf import OmegaConf
from img2img import load_model_from_config

class TextualInversionDataset(Dataset):
    def __init__(self, src_img_dir, img_repeats=100, img_size=512, center_crop=False, flip_pct=0.5, new_token="*", concept_type='object', use_filewords=True):
        self.src_img_dir = src_img_dir
        self.new_token = new_token

        # Open the set of prompt templates for the token
        prompt_file = f'./scripts/text_inv_prompts/{concept_type}_filewords_prompts.txt' if use_filewords else './scripts/text_inv_prompts/{concept_type}_prompts.txt'
        with open(prompt_file, 'r') as f:
            self.prompt_templates = f.read().splitlines()

        # Read the dataset of images
        self.img_list = sorted([f for f in os.listdir(self.src_img_dir) if f.endswith('.png') or f.endswith('.jpg')])
        self._length = len(self.img_list) * img_repeats

        # Define image transform
        self.transform = tvt.Compose([
            tvt.Lambda(lambda x: tvt.CenterCrop(min(x.size)) if center_crop else x),
            tvt.Resize((img_size, img_size), interpolation=tvt.functional._interpolation_modes_from_int(3)),  # 0 Nearest, 1 Lanczos, 2 Bilinear, 3 Bicubic
            tvt.RandomHorizontalFlip(p=flip_pct),
            tvt.ToTensor(),
            tvt.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Performs the 2 * [0, 1] - 1 operation expected by LDM
        ])

    def __len__(self):
        return self._length

    def create_text(self, filename_text):
        # Copied from SD WebUI
        text = random.choice(self.prompt_templates)
        tags = filename_text.replace('_', ',').replace('-', ',').replace(' ', ',').split(',')
        # if self.tag_drop_out != 0:
        #     tags = [t for t in tags if random.random() > self.tag_drop_out]
        # if self.shuffle_tags:
        #     random.shuffle(tags)
        text = text.replace('[filewords]', ' '.join(tags))
        text = text.replace('[name]', self.new_token)
        return text

    def __getitem__(self, i):
        example = {}
        filename, fileext = os.path.splitext(self.img_list[i % len(self.img_list)])
        example['text_values'] = self.create_text(filename)
        img = Image.open(os.path.join(self.src_img_dir, filename + fileext)).convert('RGB')
        example['pixel_values'] = self.transform(img)

        return example

def freeze_params(params):
    # Freeze all parameters in a given network
    for param in params:
        param.requires_grad = False

def training_function(args, train_dataset, model):

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.accum_iter)

    # Initialize the optimizer (only learn the input embeddings) and dataloader
    optimizer = torch.optim.AdamW(model.cond_stage_model.transformer.get_input_embeddings().parameters(), lr=args.lr)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Prepare model
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    # Move model to device and set in eval except for the text_encoder in train mode
    model.to(accelerator.device)
    model.eval()
    model.cond_stage_model.transformer.train()

    # Determine number of epochs required to achieve given steps
    # If batch_size * accum_iter * n_procs > dataset size, step/zero_grad triggers at end of dataset instead of carrying through to next epoch
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / (args.batch_size * args.accum_iter * args.n_procs))
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                # Get image encoding and text encoding, then process both in unet model
                # Might need to use model.module for multi GPU
                z_dist = model.encode_first_stage(batch['pixel_values'])
                if args.vae_sampling == 'deterministic':
                    # If True, latent always returns mean vector, else samples
                    z_dist.deterministic = True
                    z_dist.var = z_dist.std = torch.zeros_like(z_dist.mean).to(device=z_dist.parameters.device)
                z = model.get_first_stage_encoding(z_dist)
                c = model.get_learned_conditioning(batch['text_values'])
                loss = model(z, c)[0]
                accelerator.backward(loss)

                # Zero out the gradients for all token embeddings except the newly added
                # embeddings for the concept, as we only want to optimize the concept embeddings
                if accelerator.num_processes > 1:
                    grads = model.module.cond_stage_model.transformer.get_input_embeddings().weight.grad
                else:
                    grads = model.cond_stage_model.transformer.get_input_embeddings().weight.grad
                # Get the index for tokens that we want to zero the grads for
                index_grads_to_zero = torch.arange(len(model.cond_stage_model.tokenizer)) != args.new_token_id
                grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)

                # Step the optimizer to update the token embedding and then reset grads
                optimizer.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            # Update progress bar
            progress_bar.set_postfix(**{"loss": loss.detach().item()})

            # End training if current step over max setting
            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    # Save the newly trained embeddings
    if accelerator.is_main_process:
        learned_embeds = accelerator.unwrap_model(accelerator.unwrap_model(text_encoder)).get_input_embeddings().weight[args.new_token_id]
        learned_embeds_dict = {args.new_token: learned_embeds.detach().cpu()}
        os.makedirs(r'./text_inv_embeddings', exist_ok=True)
        torch.save(learned_embeds_dict, os.path.join(r'./text_inv_embeddings', args.new_token + '.pt'))

def arg_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sd_config', default='./configs/stable-diffusion/v1-inference-mist.yaml', type=str, help='Path to config which constructs model')
    parser.add_argument('--sd_ckpt', default='./checkpoints/stable-diffusion-v1-5/v1-5-pruned.ckpt', type=str, help='Path to checkpoint of model')
    parser.add_argument('--src_img_dir', default=None, type=str, help='Path of the directory of images to be processed.')
    parser.add_argument('--new_token', default='RRRRR', type=str, help='Name of token embedding to learn.')
    parser.add_argument('--init_token', default='*', type=str, help='String for to initial embedding for new token. Set as None for zeros init. Set as empty string for N(0,1) init.')
    parser.add_argument('--concept_type', default='object', choices=['object', 'style'], type=str, help='Is the new token an object or a style?')
    parser.add_argument('--use_filewords', default=True, type=lambda x: bool(strtobool(x)), help='Whether to use filename text as tags in prompt.')
    parser.add_argument('--img_repeats', default=1, type=int, help='Inflates the size of the source dataset to permit larger batch_size * accum_iter updates.')
    parser.add_argument('--img_size', default=512, type=int, help='Image size to input to LDM (after resize).')
    parser.add_argument('--center_crop', default=False, type=lambda x: bool(strtobool(x)), help='Boolean to center-crop images before resizing.')
    parser.add_argument('--flip_pct', default=0.5, type=float, help='Percentage chance for image horizontal flip.')
    parser.add_argument('--base_lr', default=5e-3, type=float, help='Base learning rate before scaling by num_processes, batch_size, and accum_iter.')
    parser.add_argument('--scale_lr', default=True, type=lambda x: bool(strtobool(x)), help='Boolean to rescale base_lr.')
    parser.add_argument('--batch_size', default=1, type=int, help='Total batch size across GPUs per iteration (before accumulation)')
    parser.add_argument('--accum_iter', default=1, type=int, help='Number of iterations to accumulate gradients before backpropagation.')
    parser.add_argument('--max_train_steps', default=10000, type=int, help='Maximum number of training steps before exit.')
    parser.add_argument('--vae_sampling', default='random', choices=['random', 'deterministic'], type=str, help='Encoding distribution sampling method - deterministic sets var/std to 0.')
    parser.add_argument('--rand_seed', default=42, type=int, help='RNG seed for reproducibility.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = arg_inputs()
    accelerate.utils.set_seed(args.rand_seed)
    # Other seeds?

    # Rescale parameters by the number of available processes (GPUs)
    args.n_procs = torch.cuda.device_count()
    args.batch_size = args.batch_size // args.n_procs
    args.lr = args.base_lr * args.batch_size * args.accum_iter * args.n_procs if args.scale_lr else args.base_lr

    # Instantiate train dataset
    train_dataset = TextualInversionDataset(src_img_dir=args.src_img_dir, img_repeats=args.img_repeats, img_size=args.img_size,
                                            center_crop=args.center_crop, flip_pct=args.flip_pct, new_token=args.new_token,
                                            concept_type=args.concept_type, use_filewords=args.use_filewords)

    # Load model
    config = OmegaConf.load(os.path.join(os.getcwd(), args.sd_config))
    config['model']['params']['unet_config']['params']['use_checkpoint'] = False  # To avoid a bug associated with calling CheckpointFunction on frozen UNet parameters
    ckpt_path = os.path.join(os.getcwd(), args.sd_ckpt)
    model = load_model_from_config(config, ckpt_path)

    # For naming convenience, declare model components (components are still linked to source model, so modifying them modifies model)
    tokenizer = model.cond_stage_model.tokenizer
    text_encoder = model.cond_stage_model.transformer

    # Add the new token in tokenizer
    num_added_tokens = tokenizer.add_tokens(args.new_token)
    assert num_added_tokens > 0, f'Tokenizer already contains {args.new_token}.'

    # Get the IDs corresponding to the input tokens and resize token embedder to reflect new token
    args.new_token_id = tokenizer.convert_tokens_to_ids(args.new_token)
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialize the new token embedding
    token_embeds = text_encoder.get_input_embeddings().weight.data
    if args.init_token is None:
        # Initialize new embedding with zeros
        token_embeds[args.new_token_id, :] = torch.zeros_like(token_embeds[0, :])
    elif args.init_token != '':
        # Initialize new embedding with the initializer embedding
        init_token_id = tokenizer.encode(args.init_token, add_special_tokens=False)
        assert len(init_token_id) == 1, f'The initializer_token {args.init_token} encodes as multiple tokens.'
        token_embeds[args.new_token_id] = token_embeds[init_token_id[0]]
    # else new embedding initializes with samples from N(0, 1) distribution

    # Freeze vae, unet, and all text_encoder parameters except for the token embeddings
    freeze_params(model.first_stage_model.parameters())  # VAE
    freeze_params(model.model.parameters())  # U-Net
    freeze_params(text_encoder.text_model.encoder.parameters())
    freeze_params(text_encoder.text_model.final_layer_norm.parameters())
    freeze_params(text_encoder.text_model.embeddings.position_embedding.parameters())

    accelerate.notebook_launcher(training_function, args=(args, train_dataset, model), num_processes=args.n_procs)
