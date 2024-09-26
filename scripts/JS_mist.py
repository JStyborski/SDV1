# Copied the repo from https://github.com/psyker-team/mist and then simplified/cleaned their library significantly to the below script

import argparse
from distutils.util import strtobool
from omegaconf import OmegaConf
import os
import ssl

import torch
import torchvision.transforms as tvt
from pytorch_lightning import seed_everything

from img2img import load_model_from_config, load_img

from advertorch.attacks import LinfPGDAttack

ssl._create_default_https_context = ssl._create_unverified_context
os.environ['TORCH_HOME'] = os.getcwd()
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hub/')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class target_model(torch.nn.Module):
    """
    A virtual model which computes the semantic and textural loss in forward function.
    """

    def __init__(self, model, condition='', loss_mode=0, semantic_rate=10000):
        """
        :param model: A SDM model.
        :param condition: The condition for computing the semantic loss (input prompt).
        """
        super().__init__()
        self.model = model
        self.condition = condition
        self.loss_mode = loss_mode
        self.semantic_rate = semantic_rate
        self.tgt_tens = None
        self.enc_loss_fn = torch.nn.MSELoss(reduction="sum")

    def forward(self, x):
        """
        Compute the loss based on different mode.
        The textural loss shows the distance between the input image and target image in latent space.
        The semantic loss describes the semantic content of the image.
        :return: The loss used for updating gradient in the adversarial attack.
        """

        z_src = self.model.get_first_stage_encoding(self.model.encode_first_stage(x)).to(device)

        if self.loss_mode == 0 or self.loss_mode == 2:
            c = self.model.get_learned_conditioning(self.condition)
            semantic_loss = self.model(z_src, c)[0]
        if self.loss_mode == 1 or self.loss_mode == 2:
            z_tgt = self.model.get_first_stage_encoding(self.model.encode_first_stage(self.tgt_tens)).to(device)
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

def parse_args():
    parser = argparse.ArgumentParser(description='Configs for Mist V1.2')
    parser.add_argument('--sd_config', default='./configs/stable-diffusion/v1-inference-mist.yaml', type=str, help='Path to config which constructs model')
    parser.add_argument('--sd_ckpt', default='./checkpoints/stable-diffusion-v1-5/v1-5-pruned.ckpt', type=str, help='Path to checkpoint of model')
    parser.add_argument('--src_dir_path', default=None, type=str, help='Path of the directory of images to be processed.')
    parser.add_argument('--src_is_obj', default=False, type=lambda x: bool(strtobool(x)), help='Whether the images to poison is an object or style.')
    parser.add_argument('--tgt_img_path', default=r'D:\Art_Styles\Rayonism_Natalia_Goncharova\Target_Imgs\MIST_Target_512.png')
    parser.add_argument('--output_name', default='Misted_Image', type=str, help='path of saved image')
    parser.add_argument('--output_dir', default='Misted_Images', type=str, help='path of output dir')
    parser.add_argument('--input_size', default=512, type=int, help='Image input size to LDM.')
    parser.add_argument('--mode', default=2, type=int, help='Attack mode - 0: Semantic, 1: Textural, 2: Joint')
    parser.add_argument('--rate', default=10000, type=int, help='Semantic loss factor under joint loss.')
    parser.add_argument('--pgd_steps', default=100, type=int, help='Number of attack steps.')
    parser.add_argument('--alpha', default=1, type=int, help='Attack step size.')
    parser.add_argument('--epsilon', default=16, type=int, help='Maximum perturbation from original input.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    seed_everything(23)

    # Process args inputs
    args.alpha = 2 * args.alpha / 255.
    args.epsilon = 2 * args.epsilon / 255.
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model configuration and checkpoint
    config = OmegaConf.load(os.path.join(os.getcwd(), args.sd_config))
    ckpt_path = os.path.join(os.getcwd(), args.sd_ckpt)
    model = load_model_from_config(config, ckpt_path).to(device)

    # Initialize model (LDM) for attack - contains forward process and loss calculation
    input_prompt = 'a photo' if args.src_is_obj else 'a painting'
    net = target_model(model, condition=input_prompt, loss_mode=args.mode, semantic_rate=args.rate)
    net.eval()

    # Load target tensor into model, if used
    if args.mode == 1 or args.mode == 2:
        assert args.tgt_img_path is not None
        net.tgt_tens = load_img(args.tgt_img_path).to(device)

    # Loop through images in input directory
    for img_id in os.listdir(args.src_dir_path):

        # Avoid other files/folders in directory
        fileName, fileExt = os.path.splitext(img_id)
        if fileExt not in ['.jpg', '.png']:
            continue

        # Load input image
        src_tens = load_img(os.path.join(args.src_dir_path, img_id)).to(device)

        # Untargeted PGD attack (maximize loss)
        attack = LinfPGDAttack(predict=net, loss_fn=identity_loss(), eps=args.epsilon, nb_iter=args.pgd_steps, eps_iter=args.alpha, clip_min=-1.0)  # Initializes the attack class and variables
        attack_output = attack.perturb(src_tens, torch.tensor([0]).to(device))  # Performs the iterative PGD algorithm

        # Save output adversarial image
        output = torch.clamp((attack_output[0] + 1.0) / 2.0, min=0.0, max=1.0).detach()
        output = tvt.functional.to_pil_image(output)
        output_path = os.path.join(args.output_dir, img_id)
        print("Output image saved in path {}".format(output_path))
        output.save(output_path)
