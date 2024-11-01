import argparse, os, sys, glob
from distutils.util import strtobool
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
import torchvision.transforms as tvt
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor


# load safety model
safety_model_id = 'CompVis/stable-diffusion-safety-checker'
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype('uint8')
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open('assets/rick.jpeg').convert('RGB').resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors='pt')
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', default='a painting of a virus monster playing guitar', type=str, nargs='?', help='the prompt to render')
    parser.add_argument('--embdir', default='./text_inv_embeddings', type=str, help='Path to folder containing custom text embeddings.')
    parser.add_argument('--outdir', default='./outputs/txt2img-samples', type=str, nargs='?', help='Directory to write files to.')
    parser.add_argument('--skip_grid', default=True, type=lambda x: bool(strtobool(x)), help='Do not save a grid, only individual samples - grids are useful when evaluating many samples.')
    parser.add_argument('--skip_save', default=False, type=lambda x: bool(strtobool(x)), help='Do not save individual samples - useful for speed measurements.')
    parser.add_argument('--ddim_steps', default=50, type=int, help='Number of DDIM sampling steps.')
    parser.add_argument('--sampler', default='ddim', choices=['ddim', 'plms', 'dpm'], type=str, help='Solver type.')
    parser.add_argument('--laion400m', default=False, type=lambda x: bool(strtobool(x)), help='Use the LAION400M model.')
    parser.add_argument('--fixed_code', default=False, type=lambda x: bool(strtobool(x)), help='Use the same starting code across samples.')
    parser.add_argument('--ddim_eta', default=0.0, type=float, help='DDIM eta (eta=0.0 corresponds to deterministic sampling).')
    parser.add_argument('--n_iter', default=2, type=int, help='Number of generation batches to run.')
    parser.add_argument('--H', default=512, type=int, help='Generated image height (pixels).')
    parser.add_argument('--W', default=512, type=int, help='Generated image width (pixels).')
    parser.add_argument('--C', default=4, type=int, help='Number of latent channels.')
    parser.add_argument('--f', default=8, type=int, help='Downsampling factor.')
    parser.add_argument('--n_samples', default=3, type=int, help='Number of samples per generation (i.e., batch size).')
    parser.add_argument('--n_rows', default=0, type=int, help='Number of rows in the output grid (defaults to n_samples)')
    parser.add_argument('--scale', default=7.5, type=float, help='Unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))')
    parser.add_argument('--from-file', type=str, help='If specified, load prompts from this file.')
    parser.add_argument('--config', default='./configs/stable-diffusion/v1-inference.yaml', type=str, help='Path to config which constructs model.')
    parser.add_argument('--ckpt', default='./checkpoints/stable-diffusion-v1-5/v1-5-pruned-emaonly.ckpt', type=str, help='Path to model checkpoint.')
    parser.add_argument('--seed', default=42, type=int,  help='RNG seed for reproducibility.')
    parser.add_argument('--precision', default='autocast', choices=['full', 'autocast'], type=str, help='Precision to evaluate.')
    parser.add_argument('--use_wm', default=False, type=lambda x: bool(strtobool(x)), help='Apply watermarking to output images.')
    opt = parser.parse_args()

    if opt.laion400m:
        print('Falling back to LAION 400M model...')
        opt.config = './configs/latent-diffusion/txt2img-1p4B-eval.yaml'
        opt.ckpt = './models/ldm/text2img-large/model.ckpt'
        opt.outdir = './outputs/txt2img-samples-laion400m'

    seed_everything(opt.seed)

    config = OmegaConf.load(f'{opt.config}')
    model = load_model_from_config(config, f'{opt.ckpt}')

    #################
    # JStyborski Edit

    # Load pretrained text embeddings
    text_inv_dict = {}
    for f in os.listdir(opt.embdir):
        pt_file = torch.load(os.path.join(os.getcwd(), opt.embdir, f))
        text_inv_dict.update(pt_file)

    # Add tokens into tokenizer and resize the embedding dictionary
    model.cond_stage_model.tokenizer.add_tokens(list(text_inv_dict.keys()))
    model.cond_stage_model.transformer.resize_token_embeddings(len(model.cond_stage_model.tokenizer))

    # Overwrite the new embedding dictionary entries with the trained embeddings
    token_embeds = model.cond_stage_model.transformer.get_input_embeddings().weight.data
    emb_tens = torch.stack(list(text_inv_dict.values()), dim=0)
    token_embeds[-len(text_inv_dict):] = emb_tens
    #################

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

    wm = 'StableDiffusionV1'
    if opt.use_wm:
        print('Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...')
        wm_encoder = WatermarkEncoder()
        wm_encoder.set_watermark('bytes', wm.encode('utf-8'))
    else:
        wm_encoder = None

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

    sample_path = os.path.join(outpath, 'samples')
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=='autocast' else nullcontext
    with torch.no_grad():
        with precision_scope('cuda'):
            with model.ema_scope():
                all_samples = list()
                for n in trange(opt.n_iter, desc='Sampling'):
                    for prompts in tqdm(data, desc='data'):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [''])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                         conditioning=c,
                                                         batch_size=opt.n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        # x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                        # x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                        # x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                        if not opt.skip_save:
                            for x_sample in x_samples_ddim:
                                img = tvt.functional.to_pil_image(x_sample, mode='RGB')
                                img = put_watermark(img, wm_encoder)
                                img.save(os.path.join(sample_path, f'{base_count:05}.png'))
                                base_count += 1

                        if not opt.skip_grid:
                            all_samples.append(x_samples_ddim)

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    img = Image.fromarray(grid.astype(np.uint8))
                    img = put_watermark(img, wm_encoder)
                    img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1

    print(f'Your samples are ready and waiting for you here: \n{outpath} \n'
          f' \nEnjoy.')


if __name__ == '__main__':
    main()
