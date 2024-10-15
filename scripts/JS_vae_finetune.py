# Copied this script from https://github.com/Leminhbinh0209/FinetuneVAE-SD and then made extensive modifications

import accelerate
from argparse import ArgumentParser
from distutils.util import strtobool
from JS_text_inversion import freeze_params
from ldm.util import instantiate_from_config
from ldm.modules.ema import LitEma
import os
from omegaconf import OmegaConf
from PIL import Image
from piq import LPIPS
from pytorch_lightning import seed_everything
import torch
from tqdm.auto import tqdm
from torchvision import transforms
from torch.utils.data import Dataset

torch.cuda.empty_cache()

def get_vae_weights(input_path):

    # Load checkpoint state dict
    pretrained_weights = torch.load(input_path)
    if 'state_dict' in pretrained_weights:
        pretrained_weights = pretrained_weights['state_dict']

    # Add 'first_stage_model' weights into the vae_weights dict
    vae_weight = {}
    for k in pretrained_weights.keys():
        if 'first_stage_model' in k:
            vae_weight[k.replace('first_stage_model.', '')] = pretrained_weights[k]

    return vae_weight

class FinetuneDataset(Dataset):
    def __init__(self, data_dir: str, inp_transform, trg_transform):
        self.data_dir = data_dir
        self.inp_transform = inp_transform
        self.trg_transform = trg_transform
        self.img_list = sorted([u for u in os.listdir(self.data_dir) if u.endswith(".png") or u.endswith(".jpg")])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.data_dir, self.img_list[idx])).convert('RGB')
        inp_img = self.inp_transform(img)
        trg_img = self.trg_transform(img)
        return inp_img, trg_img, self.img_list[idx]

class GaussianNoise(object):
    def __init__(self, sigma=0.5):
        self.sigma = sigma  # Standard deviation of the noise
    def __call__(self, x):
        x = x + torch.randn_like(x) * self.sigma  # Reparameterization trick
        x = torch.clamp(x, 0.0, 1.0)
        return x

def training_function(args, train_dataset, model):

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.accum_iter)

    # Initialize the optimizer and dataloader
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)

    # Prepare model
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    # Move model to device and set frozen portions to eval
    model.to(accelerator.device)
    model.train()
    if args.freeze_enc:
        model.encoder.eval()
    if args.freeze_dec:
        model.decoder.eval()

    # Initialize LPIPS loss and set the VGG16 network to eval
    lpips_loss_fn = LPIPS(mean=[0., 0., 0.], std=[1., 1., 1.], reduction='none')
    lpips_loss_fn.eval()

    if args.ema_decay > 0.:
        assert 0. < args.ema_decay < 1.
        model_ema = LitEma(model, decay=args.ema_decay, use_num_updates=False)  # Set use_num_updates to False because few EMA updates
        print(f'Keeping EMAs of {len(list(model_ema.buffers()))}.')

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.num_epochs), disable=not accelerator.is_local_main_process)
    progress_bar.set_description('Epochs')

    for e_step, epoch in enumerate(range(args.num_epochs)):
        for b_step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                # Get images from batch and change precision (if desired)
                inp_img, trg_img, _ = batch

                # Encode, sample, decode
                trg_posterior = model.encode(trg_img)
                posterior = model.encode(inp_img)
                z = posterior.sample()
                pred = model.decode(z)

                # Calculate latent prior KLDiv loss, LPIPS loss, latent alignment loss, and reconstruction loss
                kl_loss = posterior.kl().mean() if args.kl_loss_wt > 0. else torch.tensor([0.]).to(accelerator.device)
                lpips_loss = lpips_loss_fn(pred, trg_img).mean() if args.lpips_loss_wt > 0. else torch.tensor([0.]).to(accelerator.device)
                lat_align_loss = torch.nn.functional.mse_loss(posterior.mean, trg_posterior.mean) if args.lat_align_wt > 0. else torch.tensor([0.]).to(accelerator.device)
                rec_loss = torch.nn.functional.mse_loss(pred.contiguous(), trg_img.contiguous()) * pred.size(1)
                loss = args.kl_loss_wt * kl_loss + args.lpips_loss_wt * lpips_loss + args.lat_align_wt * lat_align_loss + rec_loss
                # rec_loss = torch.abs(trg_img.contiguous() - pred.contiguous())
                # if self.current_epoch < self.trainer.max_epochs // 3 * 2:
                #     rec_loss = rec_loss.mean() * rec_loss.size(1)  # L1 loss at epoch < 2/3 max_epochs
                #     loss = self.kl_loss_wt * kl_loss + self.lpips_loss_wt * lpips_loss + rec_loss
                # else:
                #     rec_loss = rec_loss.pow(2).mean() * rec_loss.size(1)  # L2 loss at epoch >= 2/3 max_epochs
                #     loss = self.kl_loss_wt * kl_loss + 0.1 * self.lpips_loss_wt * lpips_loss + rec_loss  # 0.1x LPIPS at epoch >= 2/3 max_epochs
                accelerator.backward(loss)

                # Step the optimizer to update the token embedding and then reset grads
                optimizer.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step and applies EMA update
            if accelerator.sync_gradients:
                if args.ema_decay > 0.:
                    model_ema(model)  # Update EMA shadow parameters with latest model weights
                    model_ema.copy_to(model)  # Copy EMA shadow parameters to model weights

            # Update progress bar with current batch
            progress_bar.set_postfix(**{'loss': loss.detach().item(), 'kl_loss': kl_loss.detach().item(), 'lpips_loss': lpips_loss.detach().item(),
                                        'lat_align_loss': lat_align_loss.detach().item()})

        accelerator.wait_for_everyone()
        progress_bar.update(1)

        # Save out checkpoint model
        if e_step % args.save_interval == 0 and accelerator.is_main_process:
            torch.save(model.state_dict(), f'{args.log_dir}/ep{e_step}_model.pth')

    # Save out final model
    if accelerator.is_main_process:
        torch.save(model.state_dict(), f'{args.log_dir}/last_model.pth')

def arg_inputs():
    parser = ArgumentParser()
    parser.add_argument('--vae_config', type=str, default='./configs/autoencoder/v1-vae.yaml', help='Path to config which constructs model')
    parser.add_argument('--sd_ckpt', type=str, default='./checkpoints/stable-diffusion-v1-5/v1-5-pruned.ckpt', help='Path to checkpoint of model')
    parser.add_argument('--freeze_enc', default=False, type=lambda x: bool(strtobool(x)), help='Boolean to freeze encoder during training.')
    parser.add_argument('--freeze_dec', default=False, type=lambda x: bool(strtobool(x)), help='Boolean to freeze decoder during training.')
    parser.add_argument('--src_img_dir', default=None, type=str, help='The directory that contains training images.')
    parser.add_argument('--output_dir', default='./vae_finetune', type=str, help='Directory to save outputs')
    parser.add_argument('--save_interval', default=10, type=int, help='Save model checkpoint every n epochs')
    parser.add_argument('--log_prefix', default='', type=str, help='Prefix title for log file')
    parser.add_argument('--img_size', default=256, type=int, help='Image input and output height/width, no cropping performed.')
    parser.add_argument('--num_epochs', default=10, type=int, help='Number of finetuning epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Total batch size across GPUs per iteration (before accumulation)')
    parser.add_argument('--accum_iter', default=1, type=int, help='Number of batches to accumulate gradients before backpropagation')
    parser.add_argument('--kl_loss_wt', default=0., type=float, help='Weight on KL divergence loss function. Recommend 0.')
    parser.add_argument('--lpips_loss_wt', default=0., type=float, help='Weight on LPIPS loss function. Recommend ~0.1.')
    parser.add_argument('--lat_align_wt', default=0., type=float, help='Weight on inp-trg encoding alignment weight. Recommend ~0.01.')
    parser.add_argument('--base_lr', default=0.0001, type=float, help='Base learning rate before scaling by num_processes, batch_size, and accum_iter.')
    parser.add_argument('--scale_lr', default=True, type=lambda x: bool(strtobool(x)), help='Boolean to rescale base_lr.')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum.')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='SGD weight decay.')
    parser.add_argument('--ema_decay', default=0.99, type=float, help='Weight for EMA decay, set to 0 for no EMA.')
    parser.add_argument('--rand_seed', default=42, type=int, help='RNG seed for reproducibility.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = arg_inputs()
    accelerate.utils.set_seed(args.rand_seed)
    seed_everything(args.rand_seed)

    # Set output filename and directory
    file_names = f'{args.log_prefix}_{args.img_size}_{args.num_epochs}_{args.batch_size}_{args.accum_iter}_{args.kl_loss_wt}_{args.lpips_loss_wt}' \
                 f'_{args.lat_align_wt}_{args.base_lr}_{args.ema_decay}_{args.freeze_enc}_{args.freeze_dec}'
    args.log_dir = f'{args.output_dir}/{file_names}'
    os.makedirs(args.log_dir, exist_ok=True)

    # Rescale parameters by the number of available processes (GPUs)
    args.n_procs = torch.cuda.device_count()
    args.batch_size = args.batch_size // args.n_procs
    args.lr = args.base_lr * args.batch_size * args.accum_iter * args.n_procs if args.scale_lr else args.base_lr

    # Define transforms for VAE input and reconstruction target
    inp_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        GaussianNoise(0.0887),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Performs the expected 2 * [0, 1] - 1 operation for LDM
    ])
    trg_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Performs the expected 2 * [0, 1] - 1 operation for LDM
    ])

    # Define dataset
    train_dataset = FinetuneDataset(args.src_img_dir, inp_transform, trg_transform)

    # Load config and get weights (not loaded yet)
    config = OmegaConf.load(args.vae_config)
    model = instantiate_from_config(config.model)
    vae_weights = get_vae_weights(args.sd_ckpt)
    model.load_state_dict(vae_weights, strict=True)

    # Freeze parameters as desired
    model.train()
    if args.freeze_enc:
        freeze_params(model.encoder.parameters())
    if args.freeze_dec:
        freeze_params(model.decoder.parameters())

    accelerate.notebook_launcher(training_function, args=(args, train_dataset, model), num_processes=args.n_procs)
