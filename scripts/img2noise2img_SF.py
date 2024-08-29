import torch

def gen_noise_from_image(self, image_batch, steps, b_idx=-1):
    skip = self.num_timesteps // steps
    seq_0_to_T = range(0, self.num_timesteps, skip)
    seq = list(seq_0_to_T)
    seq2 = seq[1:] + [self.num_timesteps]
    b_sz = len(image_batch)
    xt = image_batch
    with torch.no_grad():
        for i, j in zip(seq, seq2):
            ab_tt = self.alphas_cumprod[i]  # alpha_bar_t
            ab_t1 = self.alphas_cumprod[j]  # alpha_bar_{t+1}
            a_t1 = ab_t1 / ab_tt            # alpha_{t+1}
            t = (torch.ones(b_sz, device=self.device) * i)
            et = self.model(xt, t) # epsilon_t
            if b_idx == 0:
                msg = f"gen_noise_from_image()seq=[{seq[0]}~{seq[-1]}], len={len(seq)}"
                v, m = torch.var_mean(et)
                #log_info(f"{msg}; ts={t[0]:3.0f}, ab[{i:3d}]:{ab_tt:.6f}, ab[{j:3d}]:{ab_t1:.6f}, a[{j:3d}]:{a_t1:.6f}. epsilon var:{v:.4f}, mean:{m:7.4f}")
                print(f"{msg}; ts={t[0]:3.0f}, ab[{i:3d}]:{ab_tt:.6f}, ab[{j:3d}]:{ab_t1:.6f}, a[{j:3d}]:{a_t1:.6f}. epsilon var:{v:.4f}, mean:{m:7.4f}")
            noise_coef = (1 - ab_t1).sqrt() - (a_t1 - ab_t1).sqrt()
            xt_next = a_t1.sqrt() * xt + noise_coef * et
            xt = xt_next
        # for
    # with
    return et

def gen_image_from_noise(self, noise_batch, steps, b_idx=-1):
    skip = self.num_timesteps // steps
    seq_0_to_T = range(0, self.num_timesteps, skip)
    seq = list(reversed(seq_0_to_T))
    msg = f"gen_image_from_noise()seq=[{seq[0]}~{seq[-1]}], len={len(seq)}"
    b_sz = len(noise_batch)
    x_t = noise_batch
    seq2 = seq[1:] + [-1]
    with torch.no_grad():
        for i, j in zip(seq, seq2):
            at = self.alphas_cumprod[i+1] # alpha_bar_t
            aq = self.alphas_cumprod[j+1] # alpha_bar_{t-1}
            mt = at / aq
            t = (torch.ones(b_sz, device=self.device) * i)
            et = self.model(x_t, t) # epsilon_t
            if b_idx == 0:
                v, m = torch.var_mean(et)
                #log_info(f"{msg}; ts={i:03d}, ab:{at:.6f}, aq:{aq:.6f}. epsilon var:{v:.4f}, mean:{m:7.4f}")
                print(f"{msg}; ts={i:03d}, ab:{at:.6f}, aq:{aq:.6f}. epsilon var:{v:.4f}, mean:{m:7.4f}")
            xt_next = (x_t - (1 - at).sqrt() * et) / mt.sqrt() + (1 - aq).sqrt() * et
            x_t = xt_next
        # for
    # with
    return x_t