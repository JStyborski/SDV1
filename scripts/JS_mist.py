# Copied the repo from https://github.com/psyker-team/mist and then simplified/cleaned their library significantly to the below script

from advertorch.attacks import LinfPGDAttack
import argparse
from JS_img2img import load_model_from_config, load_img
import JS_img2noise2img
from JS_text_inversion import freeze_params
from ldm.models.diffusion.ddim import DDIMSampler
import numpy as np
from omegaconf import OmegaConf
import os
from pytorch_lightning import seed_everything
import torch
import torchvision.transforms as tvt
import warnings

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class target_model(torch.nn.Module):
    """
    A virtual model which computes the semantic and textural loss in forward function.
    """

    def __init__(self, model, vae_sampling='random', ddim_eta=1., condition='', loss_mode=0, semantic_rate=10000):
        """
        :param model: A SDM model
        :param condition: The condition for computing the semantic loss (input prompt).
        """
        super().__init__()
        self.model = model
        self.vae_sampling = vae_sampling
        self.ddim_eta = ddim_eta
        self.condition = condition
        self.loss_mode = loss_mode
        self.semantic_rate = semantic_rate
        self.tgt_tens = None
        self.enc_loss_fn = torch.nn.MSELoss(reduction="sum")

        # If not using stochastic noise, need to create a DDIM sampler
        if self.ddim_eta < 1.:
            warnings.warn('Warning: DDIM eta set < 1, this will cause the noise prediction task to take a long time.')
            self.sampler = DDIMSampler(model)
            self.sampler.make_schedule(ddim_num_steps=1000, ddim_eta=self.ddim_eta, verbose=False)
            self.sampler.ddim_timesteps_prev = np.insert(self.sampler.ddim_timesteps[:-1], 0, 0)
            self.sampler.ddim_alphas_prev = model.alphas_cumprod[self.sampler.ddim_timesteps_prev].clone().detach().to(torch.float32).to(model.device)

    def forward(self, x):
        """
        Compute the loss based on different mode.
        The textural loss shows the distance between the input image and target image in latent space.
        The semantic loss describes the semantic content of the image.
        :return: The loss used for updating gradient in the adversarial attack.
        """

        # Encode input via VAE encoder
        src_posterior = self.model.encode_first_stage(x)
        if self.vae_sampling == 'deterministic':
            src_posterior.deterministic = True
            src_posterior.var = torch.zeros_like(src_posterior.mean).to(device=src_posterior.parameters.device)
            src_posterior.std = torch.zeros_like(src_posterior.mean).to(device=src_posterior.parameters.device)
        z_src = self.model.get_first_stage_encoding(src_posterior)

        # Calculate diffusion noise prediction loss
        if self.loss_mode == 0 or self.loss_mode == 2:
            c = self.model.get_learned_conditioning(self.condition)
            if self.ddim_eta == 1.:
                # Stochastic noise and loss prediction
                semantic_loss = self.model(z_src, c)[0]
            else:
                # Calculate noised latent by DDIM inversion, then back-calculate the corresponding noise
                n_steps = torch.randint(1, 11, (1,))  # img2noise2img accepts n_steps in [0, 1000] - 0 skips the noising process entirely (i.e., 0 noise) which is misaligned with the alphas, therefore I disallow 0
                noise_latent, _ = JS_img2noise2img.image_to_noise(z_src, self.model, self.sampler, n_steps, cond=c, uncond=None, cfg_scale=1., record_latents=False)
                t = (n_steps - 1).to(z_src.device).long()  # Corresponds to ddim_timesteps_prev[n_steps - 1] for 1000 ddim_steps
                noise = (noise_latent - self.sampler.ddim_alphas_prev[t].sqrt() * z_src) / (1 - self.sampler.ddim_alphas_prev[t]).sqrt()
                semantic_loss = self.model.p_losses(z_src, c, t, noise.detach())[0]

        # Calculate loss as MSE distance between src/trg encodings
        if self.loss_mode == 1 or self.loss_mode == 2:
            tgt_posterior = self.model.encode_first_stage(self.tgt_tens)
            if self.vae_sampling == 'deterministic':
                tgt_posterior.deterministic = True
                tgt_posterior.var = torch.zeros_like(tgt_posterior.mean).to(device=tgt_posterior.parameters.device)
                tgt_posterior.std = torch.zeros_like(tgt_posterior.mean).to(device=tgt_posterior.parameters.device)
            z_tgt = self.model.get_first_stage_encoding(tgt_posterior)
            textural_loss = self.enc_loss_fn(z_src, z_tgt)

        if self.loss_mode == 0:
            return semantic_loss
        elif self.loss_mode == 1:
            return -textural_loss  # Invert because we maximize the negative loss (drive MSE dist towards 0)
        elif self.loss_mode == 2:
            return self.semantic_rate * semantic_loss - textural_loss

class identity_loss(torch.nn.Identity):
    # Overwrite forward function of identity loss to ignore second input
    def forward(self, x, y):
        return x

def arg_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sd_config', default='./configs/stable-diffusion/v1-inference-mist.yaml', type=str, help='Path to config which constructs model')
    parser.add_argument('--sd_ckpt', default='./checkpoints/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt', type=str, help='Path to checkpoint of model')
    parser.add_argument('--src_img_dir', default=None, type=str, help='Path of the directory of images to be processed.')
    parser.add_argument('--img_size', default=512, type=int, help='Image size to input to LDM (after resize).')
    parser.add_argument('--concept_type', default='object', choices=['object', 'style'], type=str, help='Is the concept an object or a style?')
    parser.add_argument('--tgt_img_path', default=None, type=str, help='If applicable (loss_mode = 1 or 2), the target image to guide textural loss.')
    parser.add_argument('--output_dir', default='Misted_Images', type=str, help='path of output dir')
    parser.add_argument('--vae_sampling', default='random', choices=['random', 'deterministic'], type=str, help='Encoding distribution sampling method - deterministic sets var/std to 0.')
    parser.add_argument('--ddim_eta', default=1., type=float, help='DDIM eta in image-to-noise (1.0 is random, 0.0 is DDIM inversion).')
    parser.add_argument('--loss_mode', default=2, type=int, help='Attack mode - 0: Semantic, 1: Textural, 2: Joint')
    parser.add_argument('--semantic_rate', default=10000, type=int, help='Semantic loss factor under joint loss.')
    parser.add_argument('--pgd_steps', default=100, type=int, help='Number of attack steps.')
    parser.add_argument('--alpha', default=1, type=int, help='Attack step size.')
    parser.add_argument('--epsilon', default=16, type=int, help='Maximum perturbation from original input.')
    parser.add_argument('--rand_seed', default=42, type=int, help='RNG seed for reproducibility.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = arg_inputs()
    seed_everything(args.rand_seed)

    # Process args inputs
    args.alpha = 2 * args.alpha / 255.
    args.epsilon = 2 * args.epsilon / 255.
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model configuration and checkpoint
    config = OmegaConf.load(os.path.join(os.getcwd(), args.sd_config))
    config['model']['params']['unet_config']['params']['use_checkpoint'] = False  # To avoid a bug associated with calling CheckpointFunction on frozen UNet parameters
    ckpt_path = os.path.join(os.getcwd(), args.sd_ckpt)
    model = load_model_from_config(config, ckpt_path).to(device)

    # Initialize model (LDM) for attack - contains forward process and loss calculation
    input_prompt = 'a photo' if args.concept_type == 'object' else 'a painting'
    net = target_model(model, vae_sampling=args.vae_sampling, ddim_eta=args.ddim_eta, condition=input_prompt, loss_mode=args.loss_mode, semantic_rate=args.semantic_rate)
    freeze_params(net.model.parameters())
    net.eval()

    # Load target tensor into model, if used
    if args.loss_mode == 1 or args.loss_mode == 2:
        assert args.tgt_img_path is not None
        net.tgt_tens = load_img(args.tgt_img_path, args.img_size).to(device)

    # Loop through images in input directory
    for img_id in os.listdir(args.src_img_dir):

        # Avoid other files/folders in directory
        fileName, fileExt = os.path.splitext(img_id)
        if fileExt not in ['.jpg', '.png']:
            continue

        # Load input image
        src_tens = load_img(os.path.join(args.src_img_dir, img_id), args.img_size).to(device)

        # Untargeted PGD attack (maximize loss)
        attack = LinfPGDAttack(predict=net, loss_fn=identity_loss(), eps=args.epsilon, nb_iter=args.pgd_steps, eps_iter=args.alpha, clip_min=-1.0)  # Initializes the attack class and variables
        attack_output = attack.perturb(src_tens, torch.tensor([0]).to(device))  # Performs the iterative PGD algorithm

        # Save output adversarial image
        output = torch.clamp((attack_output[0] + 1.0) / 2.0, min=0.0, max=1.0).detach()
        output = tvt.functional.to_pil_image(output)
        output_path = os.path.join(args.output_dir, img_id)
        print('Output image saved in path {}'.format(output_path))
        output.save(output_path)
