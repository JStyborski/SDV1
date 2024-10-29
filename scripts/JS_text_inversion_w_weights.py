# Copied cell-by-cell from https://blog.paperspace.com/dreambooth-stable-diffusion-tutorial-part-2-textual-inversion/ and then modified extensively

import accelerate
import argparse
from distutils.util import strtobool
from JS_text_inversion import TextualInversionDataset, freeze_params, weighted_infonce_loss
from ldm.util import instantiate_from_config
import math
from omegaconf import OmegaConf
import os
from pytorch_lightning import seed_everything
import random
import torch
import torch.utils.checkpoint
from torch.utils.data import Dataset
from tqdm.auto import tqdm

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)

    # Replace text embedder with custom embedder defined here
    model.cond_stage_model.transformer.text_model.embeddings = CustomCLIPTextEmbeddings(model.cond_stage_model.transformer.text_model.config)

    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

class CustomCLIPTextEmbeddings(torch.nn.Module):
    # I copied this CLIPTextEmbeddings module from the CLIP scripts
    # Modified to use custom embedding vectors for tokens identified by custom mask

    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = torch.nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = torch.nn.Embedding(config.max_position_embeddings, embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

        # Added by JStyborski
        self.custom_mask = None
        self.custom_embed = None

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None) -> torch.Tensor:

        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)
            inputs_embeds[self.custom_mask] = self.custom_embed  # Added by JStyborski

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings

# def training_function(args, train_dataset, model):
def training_function(args, train_dataset, model, base_embeds, base_wts):

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.accum_iter)

    # Initialize the optimizer (only learn the input embeddings) and dataloader
    optimizer = torch.optim.AdamW([base_wts], lr=args.lr)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Prepare model
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    # Move model to device and set in eval except for the text_encoder in train mode
    model.to(accelerator.device)
    model.eval()

    # Determine number of epochs required to achieve given steps
    # If batch_size * accum_iter * n_procs > dataset size, step/zero_grad triggers at end of dataset instead of carrying through to next epoch
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / (args.batch_size * args.accum_iter * args.n_procs))
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description('Steps')
    global_step = 0

    for epoch in range(num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                # Get image encoding
                z_dist = model.encode_first_stage(batch['pixel_values'])
                if args.vae_sampling == 'deterministic':
                    # If True, latent always returns mean vector, else samples
                    z_dist.deterministic = True
                    z_dist.var = z_dist.std = torch.zeros_like(z_dist.mean).to(device=z_dist.parameters.device)
                z = model.get_first_stage_encoding(z_dist)

                # Calculate and store the weighted embedding vector, then get text encoding
                batch_encoding = model.cond_stage_model.tokenizer(batch['text_values'], truncation=True,
                                                                  max_length=model.cond_stage_model.tokenizer.model_max_length, return_length=True,
                                                                  return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
                token_ids = batch_encoding["input_ids"].to(model.device)
                if args.wts_softmax:
                    new_embed = torch.matmul(torch.nn.functional.softmax(base_wts, dim=1), base_embeds)
                else:
                    new_embed = torch.matmul(base_wts, base_embeds)
                model.cond_stage_model.transformer.text_model.embeddings.custom_mask = token_ids == args.new_token_id
                model.cond_stage_model.transformer.text_model.embeddings.custom_embed = new_embed
                c = model.get_learned_conditioning(batch['text_values'])

                # Calculate loss and add regularization to text embedding
                loss = model(z, c)[0]
                if args.emb_l2_wt > 0.:
                    loss += args.emb_l2_wt * torch.linalg.vector_norm(new_embed)
                if args.emb_cl_wt > 0.:
                    pos_id = model.cond_stage_model.tokenizer.convert_tokens_to_ids(random.choice(args.cl_pos_tokens))
                    neg_ids = random.sample(range(len(model.cond_stage_model.tokenizer) - 1), k=args.cl_batch_size-1)
                    all_ids = [pos_id] + neg_ids
                    all_embed = model.cond_stage_model.transformer.get_input_embeddings().weight[all_ids, :]
                    loss += args.emb_cl_wt * weighted_infonce_loss(new_embed, all_embed, wince_beta=args.cl_beta, wince_tau=args.cl_tau)
                accelerator.backward(loss, retain_graph=False)

                # Step the optimizer to update the token embedding and then reset grads
                optimizer.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            # Update progress bar
            progress_bar.set_postfix(**{'loss': loss.detach().item()})

            # End training if current step over max setting
            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    # Save the newly trained embeddings
    if accelerator.is_main_process:
        if args.wts_softmax:
            learned_embed = torch.matmul(torch.nn.functional.softmax(base_wts, dim=1), base_embeds)
        else:
            learned_embed = torch.matmul(base_wts, base_embeds)
        learned_embed_dict = {args.new_token: learned_embed.squeeze().detach().cpu()}
        os.makedirs(args.outdir, exist_ok=True)
        torch.save(learned_embed_dict, os.path.join(args.outdir, args.new_token + '.pt'))

def arg_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sd_config', default='./configs/stable-diffusion/v1-inference-mist.yaml', type=str, help='Path to config which constructs model.')
    parser.add_argument('--sd_ckpt', default='./checkpoints/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt', type=str, help='Path to checkpoint of model.')
    parser.add_argument('--src_img_dir', default=None, type=str, help='Path of the directory of images to be processed.')
    parser.add_argument('--prompt_dir', default='./scripts/text_inv_prompts', type=str, help='Directory containing prompt templates.')
    parser.add_argument('--outdir', default='./text_inv_embeddings', type=str, help='Directory to save trained tokens in.')
    parser.add_argument('--new_token', default=None, type=str, help='Name of token embedding to learn.')
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
    parser.add_argument('--vae_sampling', default='deterministic', choices=['deterministic', 'random'], type=str, help='Encoding distribution sampling method - deterministic sets var/std to 0.')
    parser.add_argument('--wts_base_tokens', default=None, nargs='+', type=str, help='Tokens to use as embedding basis for learning embedding weights - set as None to use all embeddings.')
    parser.add_argument('--wts_init', default='normal', choices=['normal', 'zeros'], type=str, help='Gaussian or zero weight initialization.')
    parser.add_argument('--wts_softmax', default=True, type=lambda x: bool(strtobool(x)), help='Whether to apply softmax to embedding weights.')
    parser.add_argument('--emb_l2_wt', default=0., type=float, help='Weight to apply on L2 loss for new token embedding - only runs if >0.')
    parser.add_argument('--emb_cl_wt', default=0., type=float, help='Weight to apply on contrastive loss for new token embedding - only runs if >0.')
    parser.add_argument('--cl_pos_tokens', default=None, nargs='+', type=str, help='Tokens to use as positive samples for new token.')
    parser.add_argument('--cl_batch_size', default=128, type=int, help='Batch size for contrastive learning.')
    parser.add_argument('--cl_beta', default=1., type=float, help='Weight to apply on negatives loss in contrastive learning - 0. means positive alignment only.')
    parser.add_argument('--cl_tau', default=0.2, type=float, help='Contrastive loss temperature.')
    parser.add_argument('--rand_seed', default=42, type=int, help='RNG seed for reproducibility.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = arg_inputs()
    accelerate.utils.set_seed(args.rand_seed)
    seed_everything(args.rand_seed)

    # Rescale parameters by the number of available processes (GPUs)
    args.n_procs = torch.cuda.device_count()
    args.batch_size = args.batch_size // args.n_procs
    args.lr = args.base_lr * args.batch_size * args.accum_iter * args.n_procs if args.scale_lr else args.base_lr

    # Instantiate train dataset
    train_dataset = TextualInversionDataset(src_img_dir=args.src_img_dir, prompt_dir=args.prompt_dir, img_repeats=args.img_repeats,
                                            img_size=args.img_size, center_crop=args.center_crop, flip_pct=args.flip_pct, new_token=args.new_token,
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
    assert num_added_tokens < 2, f'New token must encode as one token.'

    # Get the IDs corresponding to the input tokens and resize token embedder to reflect new token
    args.new_token_id = tokenizer.convert_tokens_to_ids(args.new_token)
    text_encoder.resize_token_embeddings(len(tokenizer))

    if args.wts_base_tokens is not None:
        base_ids = tokenizer.convert_tokens_to_ids(args.wts_base_tokens)
        base_embeds = text_encoder.get_input_embeddings().weight.data[base_ids].detach()
    else:
        base_embeds = text_encoder.get_input_embeddings().weight.data.detach()
    if args.wts_init == 'normal':
        base_wts = torch.nn.Parameter(0.1 * torch.randn(1, len(base_embeds)), requires_grad=False).to(base_embeds.device)
    elif args.wts_init == 'zeros':
        base_wts = torch.nn.Parameter(torch.zeros(1, len(base_embeds)), requires_grad=False).to(base_embeds.device)
    base_wts.requires_grad = True

    # Freeze vae, unet, and text_encoder parameters
    freeze_params(model.first_stage_model.parameters())  # VAE
    freeze_params(model.model.parameters())  # U-Net
    freeze_params(text_encoder.parameters())

    accelerate.notebook_launcher(training_function, args=(args, train_dataset, model, base_embeds, base_wts), num_processes=args.n_procs)
