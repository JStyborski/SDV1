import torch

def image_to_noise(image_batch, model, sampler, t_enc, cond, uncond, cfg_scale, record_latents=False):
    # Initialize x_t and loop through prediction steps
    condInput = cond if uncond is None or cfg_scale == 1. else torch.cat([cond, uncond], dim=0)
    x_t = image_batch
    latentList = [x_t.detach().cpu()] if record_latents else None
    with torch.no_grad():
        for tIdx in range(t_enc):
            t = sampler.ddim_timesteps_prev[tIdx]
            if uncond is None or cfg_scale == 1.:
                eps_t = model.apply_model(x_t, torch.full([len(x_t)], t, device=x_t.device), condInput)
            else:
                eps_t_cond, eps_t_uncond = model.apply_model(torch.cat([x_t] * 2), torch.full([2 * len(x_t)], t, device=x_t.device), condInput).chunk(2)
                eps_t = eps_t_uncond + cfg_scale * (eps_t_cond - eps_t_uncond)
            eps = torch.randn_like(eps_t)
            ab_t = sampler.ddim_alphas_prev[tIdx]        # alpha_bar_{t}
            ab_dt = sampler.ddim_alphas[tIdx]            # alpha_bar_{t+dt}
            a_dt = ab_dt / ab_t                          # Product of alphas between alpha_{t} and alpha_{t+dt}
            sig_t = sampler.ddim_sigmas[tIdx]            # sigma_{t} (has already been scaled by ddim_eta)
            x_dt = a_dt.sqrt() * x_t + ((1 - ab_dt - sig_t ** 2).sqrt() - (a_dt - ab_dt).sqrt()) * eps_t + sig_t * eps  # Eqn 12 of DDIM (classifier-guidance paper showed the eqn can be used for forward process too)
            x_t = x_dt
            if record_latents:
                latentList.append(x_t.cpu())
    return x_t, latentList

def noise_to_image(noise_batch, model, sampler, t_enc, cond, uncond, cfg_scale, record_latents=False):
    # Initialize x_t and loop through prediction steps
    condInput = cond if uncond is None or cfg_scale == 1. else torch.cat([cond, uncond], dim=0)
    x_t = noise_batch
    latentList = [x_t.detach().cpu()] if record_latents else None
    with torch.no_grad():
        for tIdx in reversed(range(t_enc)):
            t = sampler.ddim_timesteps[tIdx]
            if uncond is None or cfg_scale == 1.:
                eps_t = model.apply_model(x_t, torch.full([len(x_t)], t, device=x_t.device), condInput)
            else:
                eps_t_cond, eps_t_uncond = model.apply_model(torch.cat([x_t] * 2), torch.full([2 * len(x_t)], t, device=x_t.device), condInput).chunk(2)
                eps_t = eps_t_uncond + cfg_scale * (eps_t_cond - eps_t_uncond)
            eps = torch.randn_like(eps_t)
            ab_t = sampler.ddim_alphas[tIdx]        # alpha_bar_{t}
            ab_dt = sampler.ddim_alphas_prev[tIdx]  # alpha_bar_{t-dt}
            a_dt = ab_dt / ab_t                     # Product of alphas between alpha_{t} and alpha_{t-dt}
            sig_t = sampler.ddim_sigmas[tIdx]       # sigma_{t} (has already been scaled by ddim_eta)
            x_dt = a_dt.sqrt() * x_t + ((1 - ab_dt - sig_t ** 2).sqrt() - (a_dt - ab_dt).sqrt()) * eps_t + sig_t * eps  # Eqn 12 of DDIM
            x_t = x_dt
            if record_latents:
                latentList.append(x_t.cpu())
    return x_t, latentList