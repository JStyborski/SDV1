"""make variations of input image"""

import argparse, os, sys, glob
import PIL
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
import time
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
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


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompt", type=str, nargs="?", default="a painting of a virus monster playing guitar", help="the prompt to render")
    parser.add_argument("--init-img", type=str, nargs="?", help="path to the input image")
    parser.add_argument("--outdir", type=str, nargs="?", help="dir to write results to", default="outputs/img2img-samples")
    parser.add_argument("--skip_grid", action='store_true', help="do not save a grid, only individual samples. Helpful when evaluating lots of samples")
    parser.add_argument("--skip_save", action='store_true', help="do not save indiviual samples. For speed measurements.")
    parser.add_argument("--ddim_steps", type=int, default=50, help="number of ddim sampling steps")
    parser.add_argument("--plms", action='store_true', help="use plms sampling")
    parser.add_argument("--fixed_code", action='store_true', help="if enabled, uses the same starting code across all samples ")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="ddim eta (eta=0.0 corresponds to deterministic sampling")
    parser.add_argument("--n_iter", type=int, default=1, help="sample this often")
    parser.add_argument("--C", type=int, default=4, help="latent channels")
    parser.add_argument("--f", type=int, default=8, help="downsampling factor, most often 8 or 16")
    parser.add_argument("--n_samples", type=int, default=2, help="how many samples to produce for each given prompt. A.k.a batch size")
    parser.add_argument("--n_rows", type=int, default=0, help="rows in the grid (default: n_samples)")
    parser.add_argument("--scale", type=float, default=5.0, help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))")
    parser.add_argument("--strength", type=float, default=0.75, help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image")
    parser.add_argument("--from-file", type=str, help="if specified, load prompts from this file")
    parser.add_argument("--config", type=str, default='./configs/stable-diffusion/v1-inference.yaml', help="path to config which constructs model")
    parser.add_argument("--ckpt", type=str, default='./checkpoints/stable-diffusion-v1-5/v1-5-pruned.ckpt', help="path to checkpoint of model")
    parser.add_argument("--seed", type=int, default=42, help="the seed (for reproducible sampling)")
    parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast")

    opt = parser.parse_args()

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        raise NotImplementedError("PLMS sampler not (yet) supported")
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
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    #sample_path = os.path.join(outpath, "samples")
    #os.makedirs(sample_path, exist_ok=True)

    ###################
    # JStyborski Edit #
    ###################
    #sample_path = os.path.join(outpath, 'Diff_Reconstr_Imgs')
    sample_path = os.path.join(outpath, outFolderName)
    os.makedirs(sample_path, exist_ok=True)
    if opt.save_latents:
        mean_path = os.path.join(outpath, 'Lat_Mean' + ckptSuffix)
        os.makedirs(mean_path, exist_ok=True)
        std_path = os.path.join(outpath, 'Lat_Std' + ckptSuffix)
        os.makedirs(std_path, exist_ok=True)
        init_path = os.path.join(outpath, 'Lat_Init' + ckptSuffix)
        os.makedirs(init_path, exist_ok=True)
        if opt.deterministicDiff is not None:
            i2n_path = os.path.join(outpath, 'Lat_I2N' + ckptSuffix)
            os.makedirs(i2n_path, exist_ok=True)
            n2i_path = os.path.join(outpath, 'Lat_N2I' + ckptSuffix)
            os.makedirs(n2i_path, exist_ok=True)
    ###################

    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    # JStyborski Edit (adding the loop over files)
    for imgFile in os.listdir(opt.img_dir):
        opt.init_img = os.path.join(opt.img_dir, imgFile)
        if not os.path.isfile(opt.init_img):
            continue

        assert os.path.isfile(opt.init_img)
        init_image = load_img(opt.init_img).to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        #init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

        ###################
        # JStyborski Edit #
        ###################
        enc_posterior = model.encode_first_stage(init_image)
        if opt.deterministicVAE:
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
        ###################

        sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

        ###################
        # JStyborski Edit #
        ###################
        # Nominal timestepping:
        # ddim_alphas corresponds to timesteps [1, 101, 201, ... 801, 901], ddim_alphas_prev corresponds to [0, 1, 101, ..., 701, 801]
        # On sampler.stochastic_encode, it applies noise corresponding to ddim_alphas[t_enc]
        # e.g., opt.strength = 0.5, opt.ddim_steps = 10 -> t_enc = 5, ddim_alphas[5] corresponds to 501 timestep
        # On sampler.decode, it loops through reverse steps starting from index t_enc - 1
        # e.g., opt.strength = 0.5, opt.ddim_steps = 10 -> t_enc = 5, decodes 401 -> 301 -> 201 -> 101 -> 1 -> 0
        # 1) Index error on fwd noising if opt.strength=1.0, 2) Due to shift, can never noise fully (t=1000), 3) Mismatch in fwd noise (501) and rev denoise (401)
        # Updated timestepping:
        # ddim_alphas corresponds to timesteps [99, 199, 299, ... 899, 999], ddim_alphas_prev corresponds to [0, 99, 199, ..., 799, 899]
        # t_enc input to sampler.stochastic_encode is modified to t_enc-1
        # e.g., opt.strength = 0.5, opt.ddim_steps = 10 -> t_enc = 5, ddim_alphas[4] corresponds to 499 timestep
        # t_enc input to sampler.decode is unchanged, so it still starts from index t_enc - 1
        # e.g., opt.strength = 0.5, opt.ddim_steps = 10 -> t_enc = 5, decodes 499 -> 399 -> 299 -> 199 -> 99 -> 0
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(model.device)
        if opt.useOrigTimesteps:
            sampler.ddim_timesteps_prev = np.insert(sampler.ddim_timesteps[:-1], 0, 0)
            sampler.ddim_alphas_prev = to_torch(model.alphas_cumprod[sampler.ddim_timesteps_prev])  # For some reason, this is stored as numpy array
        else:
            nTimesteps = len(model.alphas_cumprod)
            assert nTimesteps % opt.ddim_steps == 0
            stepSize = nTimesteps // opt.ddim_steps
            sampler.ddim_timesteps = np.array(range(stepSize - 1, nTimesteps, stepSize))
            sampler.ddim_timesteps_prev = np.insert(sampler.ddim_timesteps[:-1], 0, 0)
            sampler.ddim_alphas = to_torch(model.alphas_cumprod[sampler.ddim_timesteps])
            sampler.ddim_alphas_prev = to_torch(model.alphas_cumprod[sampler.ddim_timesteps_prev])
            sampler.ddim_sqrt_one_minus_alphas = to_torch(torch.sqrt(1 - sampler.ddim_alphas))
            sampler.ddim_sigmas = to_torch(opt.ddim_eta * torch.sqrt((1 - sampler.ddim_alphas_prev) / (1 - sampler.ddim_alphas)
                                                                     * (1 - sampler.ddim_alphas / sampler.ddim_alphas_prev)))
        ###################

        assert 0. <= opt.strength <= 1., 'Can only work with strength in [0.0, 1.0]'
        # JStyborski Edit
        assert opt.strength * opt.ddim_steps % 1 == 0, 'To ensure that denoising strength aligns with timestep indexing'
        t_enc = int(opt.strength * opt.ddim_steps)
        print(f"Target t_enc is {t_enc} steps")

        precision_scope = autocast if opt.precision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    tic = time.time()
                    all_samples = list()
                    for n in trange(opt.n_iter, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if opt.scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)

                            # encode (scaled latent)
                            #z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * batch_size).to(device))
                            # decode it
                            #samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale, unconditional_conditioning=uc)

                            ###################
                            # JStyborski Edit #
                            ###################
                            if opt.deterministicDiff is None:
                                # Skip diffusion process entirely
                                # The program already does similar since t_enc would be set to 0, but in the DDIM sampler, the 0th alpha value (~0.04)
                                # still results in significant noise added to the image, which is then not denoised at all
                                samples = init_latent
                            elif opt.deterministicDiff:
                                # Deterministic encode/decode
                                noise_latent, i2nList = JS_img2noise2img.image_to_noise(init_latent, model, sampler, t_enc, cond=c, uncond=uc, record_latents=opt.save_streamlines)
                                samples, n2iList = JS_img2noise2img.noise_to_image(noise_latent, model, sampler, t_enc, cond=c, uncond=uc, record_latents=opt.save_streamlines)
                                if opt.save_latents and opt.save_streamlines:
                                    torch.save(torch.stack(i2nList), os.path.join(i2n_path, fileName + '_i2n_sl.pt'))
                                    torch.save(torch.stack(n2iList), os.path.join(n2i_path, fileName + '_n2i_sl.pt'))
                                elif opt.save_latents:
                                    torch.save(noise_latent.cpu(), os.path.join(i2n_path, fileName + '_i2n_final.pt'))
                                    torch.save(samples.cpu(), os.path.join(n2i_path, fileName + '_n2i_final.pt'))
                            else:
                                # Stochastic encode/decode
                                z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc - 1] * batch_size).to(device))
                                samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale, unconditional_conditioning=uc)
                                if opt.save_latents:
                                    torch.save(z_enc.cpu(), os.path.join(i2n_path, fileName + '_i2n_final.pt'))
                                    torch.save(samples.cpu(), os.path.join(n2i_path, fileName + '_n2i_final.pt'))
                            ###################

                            x_samples = model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                            if not opt.skip_save:
                                for x_sample in x_samples:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    #Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:05}.png"))
                                    # JStyborski Edit
                                    #Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, fileName + '_diff_recon.png'))
                                    Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, fileName + outFileSuffix))
                                    base_count += 1
                            all_samples.append(x_samples)

                    if not opt.skip_grid:
                        # additionally, save as grid
                        grid = torch.stack(all_samples, 0)
                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                        grid = make_grid(grid, nrow=n_rows)

                        # to image
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                        grid_count += 1

                    toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")

if __name__ == "__main__":
    main()
