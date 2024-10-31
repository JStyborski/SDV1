"""make variations of input image"""

import argparse, os, sys, glob
from distutils.util import strtobool
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
import JS_img2noise2img


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
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

def load_img(path, img_size=None):
    image = Image.open(path).convert('RGB')
    if img_size is None:
        w, h = image.size
        print(f'Loaded input image of size ({w}, {h}) from {path}.')
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        image = image.resize((w, h), resample=Image.LANCZOS)
    else:
        print(f'Loaded input image of size ({img_size}, {img_size}) from {path}.')
        image = image.resize((img_size, img_size), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def arg_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', default='a painting of a cat', type=str, nargs='?', help="the prompt to render")
    parser.add_argument('--src_img_dir', type=str, nargs='?', help='The directory that contains images to apply img2img.')
    parser.add_argument('--outdir', default='outputs/img2img-samples', type=str, nargs='?', help='dir to write results to')
    parser.add_argument('--skip_grid', default=True, type=lambda x: bool(strtobool(x)), help='do not save a grid, only individual samples. Helpful when evaluating lots of samples')
    parser.add_argument('--skip_save', default=False, type=lambda x: bool(strtobool(x)), help='Do not save indiviual samples. For speed measurements.')
    parser.add_argument('--fixed_code', default=False, type=lambda x: bool(strtobool(x)), help='If enabled, uses the same starting code across all samples.')
    parser.add_argument('--vae_sampling', default='random', choices=['random', 'deterministic'], type=str, help='Encoding distribution sampling method - deterministic sets var/std to 0.')
    parser.add_argument('--use_diffusion', default=True, type=lambda x: bool(strtobool(x)), help='Whether to use diffusion or skip it (VAE only).')
    parser.add_argument('--sampler', default='ddim', choices=['ddim', 'plms', 'dpm'], type=str, help='Solver type.')
    parser.add_argument('--ddim_steps', default=50, type=int, help='Number of ddim sampling steps (across T=1000 DDPM steps).')
    parser.add_argument('--use_orig_tsteps', default=True, type=lambda x: bool(strtobool(x)), help='Whether to use original timestep scheme (wrong) or mine (correct)')
    parser.add_argument('--ddim_eta_fwd', default=1.0, type=float, help='DDIM eta in image-to-noise (1.0 is random, 0.0 is DDIM inversion).')
    parser.add_argument('--ddim_eta_rev', default=0.0, type=float, help='DDIM eta in noise-to-image (1.0 is DDPM, 0.0 is DDIM).')
    parser.add_argument('--n_iter', default=1, type=int, help='Number of sampling runs per image.')
    parser.add_argument('--img_size', default=512, type=int, help='Image size to input to LDM (after resize).')
    parser.add_argument('--C', default=4, type=int, help='Latent channels.')
    parser.add_argument('--f', default=8, type=int, help='Downsampling factor, often 8 or 16.')
    parser.add_argument('--n_samples', default=2, type=int, help='How many samples to produce for each given prompt. A.k.a batch size')
    parser.add_argument('--n_rows', default=0, type=int, help='Rows in the grid (default: n_samples)')
    parser.add_argument('--scale_fwd', default=5.0, type=float, help='CFG scale in forward process: 1.0 corresponds to conditional output, >1.0 emphasizes conditioning.')
    parser.add_argument('--scale_rev', default=5.0, type=float, help='CFG scale in reverse process: 1.0 corresponds to conditional output, >1.0 emphasizes conditioning.')
    parser.add_argument('--strength', default=0.75, type=float, help='Strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image')
    parser.add_argument('--from-file', type=str, help='If specified, load prompts from this file')
    parser.add_argument('--config', default='./configs/stable-diffusion/v1-inference.yaml', type=str, help='path to config which constructs model')
    parser.add_argument('--ckpt', default='./checkpoints/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt', type=str, help='path to checkpoint of model')
    parser.add_argument('--seed', default=42, type=int, help='Seed (for reproducible sampling).')
    parser.add_argument('--precision', default='autocast', choices=['full', 'autocast'], type=str, help='evaluate at this precision')
    parser.add_argument('--save_latents', default=False, type=lambda x: bool(strtobool(x)), help='Whether to save out latents.')
    parser.add_argument('--save_streamlines', default=False, type=lambda x: bool(strtobool(x)), help='Whether to save out every step of the diffusion latents.')
    args = parser.parse_args()
    return args

def main():

    opt = arg_inputs()
    seed_everything(opt.seed)

    # Load model and push to device
    config = OmegaConf.load(f'{opt.config}')
    model = load_model_from_config(config, f'{opt.ckpt}')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    if opt.sampler == 'dpm':
        sampler = DPMSolverSampler(model)
    elif opt.sampler == 'plms':
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]
    else:
        print(f'reading prompts from {opt.from_file}')
        with open(opt.from_file, 'r') as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    # Create output folders
    sample_path = os.path.join(opt.outdir, 'Reconstr_Imgs')
    os.makedirs(sample_path, exist_ok=True)
    if opt.save_latents:
        mean_path = os.path.join(opt.outdir, 'Lat_Mean')
        os.makedirs(mean_path, exist_ok=True)
        std_path = os.path.join(opt.outdir, 'Lat_Std')
        os.makedirs(std_path, exist_ok=True)
        init_path = os.path.join(opt.outdir, 'Lat_Init')
        os.makedirs(init_path, exist_ok=True)
    if opt.save_streamlines:
        i2n_path = os.path.join(opt.outdir, 'Lat_I2N')
        os.makedirs(i2n_path, exist_ok=True)
        n2i_path = os.path.join(opt.outdir, 'Lat_N2I')
        os.makedirs(n2i_path, exist_ok=True)

    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    # Loop over all files in a directory
    for imgFile in os.listdir(opt.src_img_dir):

        opt.init_img = os.path.join(opt.src_img_dir, imgFile)
        if not os.path.isfile(opt.init_img):
            continue

        assert os.path.isfile(opt.init_img)
        init_image = load_img(opt.init_img, opt.img_size).to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)

        # Encode input via VAE encoder
        enc_posterior = model.encode_first_stage(init_image)
        if opt.vae_sampling == 'deterministic':
            # If True, latent always returns posterior mean vector, else samples
            enc_posterior.deterministic = True
            enc_posterior.var = torch.zeros_like(enc_posterior.mean).to(device=enc_posterior.parameters.device)
            enc_posterior.std = torch.zeros_like(enc_posterior.mean).to(device=enc_posterior.parameters.device)
        fileName, _ = os.path.splitext(os.path.basename(opt.init_img))
        init_latent = model.get_first_stage_encoding(enc_posterior)  # move to latent space
        if opt.save_latents:
            torch.save(enc_posterior.mean.detach().cpu(), os.path.join(mean_path, fileName + '_post_mean.pt'))
            torch.save(enc_posterior.std.detach().cpu(), os.path.join(std_path, fileName + '_post_std.pt'))
            torch.save(init_latent.detach().cpu(), os.path.join(init_path, fileName + '_init_latent.pt'))

        sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta_rev, verbose=False)

        # Nominal timestepping:
        # ddim_alphas corresponds to timesteps [1, 101, 201, ... 801, 901], ddim_alphas_prev corresponds to [0, 1, 101, ..., 701, 801]
        # On sampler.stochastic_encode, it applies noise corresponding to ddim_alphas[n_steps]
        # e.g., opt.strength = 0.5, opt.ddim_steps = 10 -> n_steps = 5, ddim_alphas[5] corresponds to 501 timestep
        # On sampler.decode, it loops through reverse steps starting from index n_steps - 1
        # e.g., opt.strength = 0.5, opt.ddim_steps = 10 -> n_steps = 5, decodes 401 -> 301 -> 201 -> 101 -> 1 -> 0
        # 1) Index error on fwd noising if opt.strength=1.0, 2) Due to shift, can never noise fully (t=1000), 3) Mismatch in fwd noise (501) and rev denoise (401)
        # Updated timestepping:
        # ddim_alphas corresponds to timesteps [99, 199, 299, ... 899, 999], ddim_alphas_prev corresponds to [0, 99, 199, ..., 799, 899]
        # n_steps input to sampler.stochastic_encode is modified to n_steps-1
        # e.g., opt.strength = 0.5, opt.ddim_steps = 10 -> n_steps = 5, ddim_alphas[4] corresponds to 499 timestep
        # n_steps input to sampler.decode is unchanged, so it still starts from index n_steps - 1
        # e.g., opt.strength = 0.5, opt.ddim_steps = 10 -> n_steps = 5, decodes 499 -> 399 -> 299 -> 199 -> 99 -> 0
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(model.device)
        if opt.use_orig_tsteps:
            sampler.ddim_timesteps_prev = np.insert(sampler.ddim_timesteps[:-1], 0, 0)
            sampler.ddim_alphas_prev = to_torch(model.alphas_cumprod[sampler.ddim_timesteps_prev])
        else:
            nTimesteps = len(model.alphas_cumprod)
            assert nTimesteps % opt.ddim_steps == 0
            stepSize = nTimesteps // opt.ddim_steps
            sampler.ddim_timesteps = np.array(range(stepSize - 1, nTimesteps, stepSize))
            sampler.ddim_timesteps_prev = np.insert(sampler.ddim_timesteps[:-1], 0, 0)
            sampler.ddim_alphas = to_torch(model.alphas_cumprod[sampler.ddim_timesteps])
            sampler.ddim_alphas_prev = to_torch(model.alphas_cumprod[sampler.ddim_timesteps_prev])
            sampler.ddim_sqrt_one_minus_alphas = to_torch(torch.sqrt(1 - sampler.ddim_alphas))
            sampler.ddim_sigmas = to_torch(opt.ddim_eta_rev * torch.sqrt((1 - sampler.ddim_alphas_prev) / (1 - sampler.ddim_alphas) * (1 - sampler.ddim_alphas / sampler.ddim_alphas_prev)))

        assert 0. <= opt.strength <= 1., 'Can only work with strength in [0.0, 1.0]'
        assert opt.strength * opt.ddim_steps % 1 == 0, 'To ensure that denoising strength aligns with timestep indexing'
        n_steps = int(opt.strength * opt.ddim_steps)
        print(f'Target t is {n_steps} steps')

        precision_scope = autocast if opt.precision == 'autocast' else nullcontext
        with torch.no_grad():
            with precision_scope('cuda'):
                with model.ema_scope():
                    all_samples = list()
                    for n in trange(opt.n_iter, desc='Sampling'):
                        for prompts in tqdm(data, desc='data'):

                            # Get text prompt encodings
                            uc = None
                            if opt.scale_fwd != 1.0 or opt.scale_rev != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [''])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)

                            # Diffusion process
                            if opt.use_diffusion and opt.ddim_eta_fwd == 1. and not(opt.save_streamlines):
                                # Stochastic encode, decode stochasticity determined by opt.ddim_eta_rev
                                z_enc = sampler.stochastic_encode(init_latent, torch.tensor([n_steps - 1] * batch_size).to(device))
                                samples = sampler.decode(z_enc, c, n_steps, unconditional_guidance_scale=opt.scale_rev, unconditional_conditioning=uc)
                                if opt.save_latents:
                                    torch.save(z_enc.cpu(), os.path.join(i2n_path, fileName + '_i2n_final.pt'))
                                    torch.save(samples.cpu(), os.path.join(n2i_path, fileName + '_n2i_final.pt'))
                            elif opt.use_diffusion:
                                # Encode/decode stochasticity determined by opt.ddim_eta_fwd and opt.ddim_eta_rev
                                sampler.ddim_sigmas = to_torch(opt.ddim_eta_fwd * torch.sqrt((1 - sampler.ddim_alphas_prev) / (1 - sampler.ddim_alphas) * (1 - sampler.ddim_alphas / sampler.ddim_alphas_prev)))
                                noise_latent, i2nList = JS_img2noise2img.image_to_noise(init_latent, model, sampler, n_steps, cond=c, uncond=uc, cfg_scale=opt.scale_fwd, record_latents=opt.save_streamlines)
                                sampler.ddim_sigmas = to_torch(opt.ddim_eta_rev * torch.sqrt((1 - sampler.ddim_alphas_prev) / (1 - sampler.ddim_alphas) * (1 - sampler.ddim_alphas / sampler.ddim_alphas_prev)))
                                samples, n2iList = JS_img2noise2img.noise_to_image(noise_latent, model, sampler, n_steps, cond=c, uncond=uc, cfg_scale=opt.scale_rev, record_latents=opt.save_streamlines)
                                if opt.save_streamlines:
                                    torch.save(torch.stack(i2nList), os.path.join(i2n_path, fileName + '_i2n_sl.pt'))
                                    torch.save(torch.stack(n2iList), os.path.join(n2i_path, fileName + '_n2i_sl.pt'))
                                elif opt.save_latents:
                                    torch.save(noise_latent.cpu(), os.path.join(i2n_path, fileName + '_i2n_final.pt'))
                                    torch.save(samples.cpu(), os.path.join(n2i_path, fileName + '_n2i_final.pt'))
                            else:
                                # Skip diffusion process entirely
                                # The program already does similar since n_steps would be set to 0, but in the DDIM sampler, the 0th alpha value (~0.04)
                                # still results in significant noise added to the image, which is then not denoised at all
                                samples = init_latent

                            # Decode samples from latent space
                            x_samples = model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                            # Save image
                            if not opt.skip_save:
                                for x_sample in x_samples:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    #Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:05}.png"))
                                    Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, imgFile))
                                    base_count += 1
                            all_samples.append(x_samples)

                    if not opt.skip_grid:
                        # additionally, save as grid
                        grid = torch.stack(all_samples, 0)
                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                        grid = make_grid(grid, nrow=n_rows)

                        # to image
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        Image.fromarray(grid.astype(np.uint8)).save(os.path.join(sample_path, f'grid-{grid_count:04}.png'))
                        grid_count += 1

    print(f'Your samples are ready and waiting for you here: \n{sample_path} \n'
          f' \nEnjoy.')

if __name__ == '__main__':
    main()
