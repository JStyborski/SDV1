# Copied this script from https://github.com/Leminhbinh0209/FinetuneVAE-SD and then made extensive modifications

import os
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.optim as optim
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import Dataset
from pytorch_lightning import Trainer
from PIL import Image
from argparse import ArgumentParser
from piq import LPIPS
from ldm.util import   instantiate_from_config
from ldm.modules.ema import LitEma

torch.cuda.empty_cache()

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

class FinetuneVAE(pl.LightningModule):
    def __init__(self, vae_config=None, vae_weights=None, kl_loss_weight=0.1, lpips_loss_weight=0.1, optim='sgd', lr=1e-4, momentum=0.9,
                 weight_decay=5e-4, ema_decay=0.999, precision=32, log_dir=None):
        super().__init__()
        self.model = instantiate_from_config(vae_config)
        self.model.load_state_dict(vae_weights, strict=True)
        self.model.train()
        self.kl_loss_weight = kl_loss_weight
        self.lpips_loss_weight = lpips_loss_weight
        self.lpips_loss_fn = LPIPS(mean=[0., 0., 0.], std=[1., 1., 1.], reduction='none')  # LPIPS loss with VGG16 for images that have already been normalized
        self.lpips_loss_fn.eval()
        self.optim = optim
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_ema = ema_decay > 0
        if self.use_ema:
            self.ema_decay = ema_decay
            assert 0. < ema_decay < 1.
            self.model_ema = LitEma(self.model, decay=ema_decay, use_num_updates=False)  # Set use_num_updates to False because few EMA updates
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
        self.precision = precision
        self.log_dir = log_dir
        self.log_one_batch = False

    def configure_optimizers(self):
        if self.optim == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError
        return optimizer

    def training_step(self, batch, batch_idx):

        # Get images from batch and change precision (if desired)
        inp_img, trg_img, _ = batch
        if self.precision == 16:
            inp_img = inp_img.half()
            trg_img = trg_img.half()

        # Encode, sample, decode
        posterior = self.model.encode(inp_img)
        z = posterior.sample()
        pred = self.model.decode(z)

        # Calculate latent prior KLDiv loss, LPIPS loss, and reconstruction loss
        kl_loss = posterior.kl().mean()
        lpips_loss = self.lpips_loss_fn(pred, trg_img).mean()
        rec_loss = torch.abs(trg_img.contiguous() - pred.contiguous())
        if self.current_epoch < self.trainer.max_epochs // 3 * 2:
            rec_loss = rec_loss.mean() * rec_loss.size(1)  # L1 loss at epoch < 2/3 max_epochs
            loss = self.kl_loss_weight * kl_loss + self.lpips_loss_weight * lpips_loss + rec_loss
        else:
            rec_loss = rec_loss.pow(2).mean() * rec_loss.size(1)  # L2 loss at epoch >= 2/3 max_epochs
            loss = self.kl_loss_weight * kl_loss + 0.1 * self.lpips_loss_weight * lpips_loss + rec_loss  # 0.1x LPIPS at epoch >= 2/3 max_epochs

        self.log('kl_loss', kl_loss, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('lpips_loss', lpips_loss, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('rec_loss', rec_loss, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        return loss

    #def on_train_batch_end(self, outputs, batch, batch_idx):  # Can do EMA update per batch
    def on_train_epoch_end(self):  # Or update per epoch

        # EMA update weights
        if self.use_ema:
            self.model_ema(self.model)  # Update EMA shadow parameters with latest model weights
            self.model_ema.copy_to(self.model)  # Copy EMA shadow parameters to model weights

    def validation_step(self, batch, batch_idx):
        # Same as training_step, with slightly different logging
        inp_img, trg_img, _ = batch
        if self.precision == 16:
            inp_img = inp_img.half()
            trg_img = trg_img.half()
        posterior = self.model.encode(inp_img)
        z = posterior.mode()
        pred = self.model.decode(z)
        kl_loss = posterior.kl().mean()
        lpips_loss = self.lpips_loss_fn(pred, trg_img).mean()
        rec_loss = torch.abs(trg_img.contiguous() - pred.contiguous())
        if self.current_epoch < self.trainer.max_epochs // 3 * 2:
            rec_loss = rec_loss.mean() * rec_loss.size(1)  # L1 loss at epoch < 2/3 max_epochs
            loss = self.kl_loss_weight * kl_loss + self.lpips_loss_weight * lpips_loss + rec_loss
        else:
            rec_loss = rec_loss.pow(2).mean() * rec_loss.size(1)  # L2 loss at epoch >= 2/3 max_epochs
            loss = self.kl_loss_weight * kl_loss + 0.1 * self.lpips_loss_weight * lpips_loss + rec_loss  # 0.1x LPIPS at epoch >= 2/3 max_epochs
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_images(trg_img, pred)
        return {"kl_loss": kl_loss, "lpips_loss": lpips_loss, "rec_loss": rec_loss, 'val_loss': loss}

    def on_validation_end(self, validation_step_outputs):
        self.log_one_batch = False
        # Gather and log the validation_step outputs
        kl_loss = torch.stack([x['kl_loss'] for x in validation_step_outputs]).mean()
        lpips_loss = torch.stack([x['lpips_loss'] for x in validation_step_outputs]).mean()
        rec_loss = torch.stack([x['rec_loss'] for x in validation_step_outputs]).mean()
        val_loss = torch.stack([x['val_loss'] for x in validation_step_outputs]).mean()
        self.log('val_kl_loss', kl_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_lpips_loss', lpips_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_rec_loss', rec_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def log_images(self, input, output, names):
        if self.log_one_batch:
            return
        for img1, img2, name in zip(input, output, names):
            # Get target/pred images, transpose HxW, rescale to [0, 1], and calculate their difference
            img1 = img1.cpu().detach().numpy().transpose(1, 2, 0)
            img1 = (img1 + 1) / 2
            img2 = img2.cpu().detach().numpy().transpose(1, 2, 0)
            img2 = (img2 + 1) / 2
            diff = abs(img1 - img2)

            # Concatenate images to one array, convert to PIL image, and save
            img = np.concatenate([img1, img2, diff], axis=1)
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)
            os.makedirs(self.log_dir + "/" + str(self.current_epoch), exist_ok=True)
            img.save(os.path.join(self.log_dir, str(self.current_epoch), name))
        self.log_one_batch = True

def get_vae_weights(input_path):

    # Load checkpoint state dict
    pretrained_weights = torch.load(input_path)
    if 'state_dict' in pretrained_weights:
        pretrained_weights = pretrained_weights['state_dict']

    # Add 'first_stage_model" weights into the vae_weights dict
    vae_weight = {}
    for k in pretrained_weights.keys():
        if "first_stage_model" in k:
            vae_weight[k.replace("first_stage_model.", "")] = pretrained_weights[k]

    return vae_weight

def argument_inputs():

    parser = ArgumentParser()

    parser.add_argument('--train_root', type=str, default='./dataset/', help='The directory that contains training images')
    parser.add_argument('--val_root', type=str, default=None, help='The directory that contains validation images - set as None for no validation')
    parser.add_argument('--output_dir', type=str, default='../vae_finetune', help='Directory to save outputs')
    parser.add_argument('--save_interval', type=int, default=10, help='Save model checkpoint every n epochs')
    parser.add_argument('--log_prefix', type=str, default='', help='Prefix title for log file')
    parser.add_argument("--vae_config", type=str, default="../configs/autoencoder/v1-vae.yaml", help='Path to config which constructs model')
    parser.add_argument("--sd_ckpt", type=str, default='../checkpoints/stable-diffusion-v1-5/v1-5-pruned.ckpt', help='Path to checkpoint of model')
    parser.add_argument('--strategy', type=str, default='single_device', choices=['single_device', 'ddp'], help='Method for single/multi process training')
    parser.add_argument('--precision', type=int, default=16, choices=[16, 32], help='Floating point precision for torch tensors.')
    parser.add_argument('--image_size', type=int, default=256, help='Image input and output height/width, no cropping performed.')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32, help='Total batch size across GPUs per iteration (before accumulation)')
    parser.add_argument('--accum_iter', type=int, default=1, help='Number of batches to accumulate gradients before backpropagation')
    parser.add_argument('--kl_loss_weight', type=float, default=0.)
    parser.add_argument('--lpips_loss_weight', type=float, default=1.)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--ema_decay', type=float, default=0.99, help="Use use_ema")

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = argument_inputs()

    # JStyborski Edits
    args.train_root = r'C:\Users\jeremy\Python_Projects\Art_Styles\images\Rayonism_Natalia_Goncharova\Orig_Imgs'
    args.num_epochs = 200
    args.save_interval = 1
    args.batch_size = 1
    args.lr = 0.00002
    args.log_prefix = 'RNG_Orig_GN0p1774'

    # Set output filename and directory
    file_names = f"{args.log_prefix}_imgsize({args.image_size})_epochs({args.num_epochs})_bs({args.batch_size})_accum({args.accum_iter})" \
                 f"_kl({args.kl_loss_weight})_lpips({args.lpips_loss_weight})_lr({args.lr})_ema({args.ema_decay})"
    log_dir = f"{args.output_dir}/{file_names}"
    os.makedirs(log_dir, exist_ok=True)

    # Determine process settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    args.devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [0]
    if args.strategy == 'single_device':
        assert len(args.devices) == 1
        args.strategy = 'auto'  # Overwrite name to avoid device bug (https://github.com/Lightning-AI/pytorch-lightning/issues/18902)
    args.batch_size = args.batch_size // len(args.devices)  # Adjust total batch size to be per-device (which is what PL expects)

    # Load config and get weights (not loaded yet)
    config = OmegaConf.load(args.vae_config)
    vae_config = config.model
    vae_weights = get_vae_weights(args.sd_ckpt)

    # Define transforms for VAE input and reconstruction target
    inp_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        GaussianNoise(0.0887),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Performs the expected 2 * [0, 1] - 1 operation for LDM
    ])
    trg_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Performs the expected 2 * [0, 1] - 1 operation for LDM
    ])

    # Define datasets and dataloaders
    train_ds = FinetuneDataset(args.train_root, inp_transform, trg_transform)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    if args.val_root is not None:
        val_ds = FinetuneDataset(args.val_root, trg_transform, trg_transform)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
    else:
        val_dl = None

    # Instantiate PL module
    vae_plmod = FinetuneVAE(vae_config=vae_config, vae_weights=vae_weights, kl_loss_weight=args.kl_loss_weight,
                            lpips_loss_weight=args.lpips_loss_weight, lr=args.lr, ema_decay=args.ema_decay, precision=args.precision, log_dir=log_dir)
    vae_plmod = vae_plmod.to(device)

    # Instantiate PL Trainer
    checkpoint_callback = pl.callbacks.ModelCheckpoint(every_n_epochs=args.save_interval)  # Custom callback to avoid saving every epoch
    trainer = Trainer(min_epochs=1, max_epochs=args.num_epochs, precision=args.precision, strategy=args.strategy, accelerator=args.accelerator,
                      devices=args.devices, accumulate_grad_batches=args.accum_iter, num_sanity_val_steps=0 if args.val_root is None else 1,
                      callbacks=[checkpoint_callback], default_root_dir=log_dir)

    # Fit the model (i.e., run the optimization) and save
    trainer.fit(vae_plmod, train_dataloaders=train_dl, val_dataloaders=val_dl)
    torch.save(vae_plmod.model.state_dict(), f"{log_dir}/last_model.pth")
