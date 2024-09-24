import torch

def image_to_noise(image_batch, model, sampler, t_enc, cond, uncond, record_latents=False):
    # Initialize x_t and loop through prediction steps
    condInput = cond if uncond is None else torch.cat([cond, uncond], dim=0)
    nRepeat = 1 if uncond is None else 2
    x_t = image_batch
    latentList = [x_t.detach().cpu()] if record_latents else None
    with torch.no_grad():
        for tIdx in range(t_enc):
            t = sampler.ddim_timesteps_prev[tIdx]
            eps_t = model.apply_model(x_t.repeat(nRepeat, 1, 1, 1), torch.ones(len(image_batch)).cuda() * t, cond=condInput)
            ab_t = sampler.ddim_alphas_prev[tIdx]        # alpha_bar_t
            ab_dt = sampler.ddim_alphas[tIdx]            # alpha_bar_{t+dt}
            a_dt = ab_dt / ab_t                          # Product of alphas between alpha_{t} and alpha_{t+dt}
            x_dt = a_dt.sqrt() * x_t + ((1 - ab_dt).sqrt() - (a_dt - ab_dt).sqrt()) * eps_t  # Eqn 13 of DDIM (classifier-guidance paper showed the eqn can be used for forward process too)
            x_t = x_dt
            if record_latents:
                latentList.append(x_t.cpu())
    return x_t, latentList

def noise_to_image(noise_batch, model, sampler, t_enc, cond, uncond, record_latents=False):
    # Initialize x_t and loop through prediction steps
    condInput = cond if uncond is None else torch.cat([cond, uncond], dim=0)
    nRepeat = 1 if uncond is None else 2
    x_t = noise_batch
    latentList = [x_t.detach().cpu()] if record_latents else None
    with torch.no_grad():
        for tIdx in reversed(range(t_enc)):
            t = sampler.ddim_timesteps[tIdx]
            eps_t = model.apply_model(x_t.repeat(nRepeat, 1, 1, 1), torch.ones(len(noise_batch)).cuda() * t, cond=condInput)
            ab_t = sampler.ddim_alphas[tIdx]        # alpha_bar_t
            ab_dt = sampler.ddim_alphas_prev[tIdx]  # alpha_bar_{t-dt}
            a_dt = ab_dt / ab_t                     # Product of alphas between alpha_{t} and alpha_{t-dt}
            x_dt = a_dt.sqrt() * x_t + ((1 - ab_dt).sqrt() - (a_dt - ab_dt).sqrt()) * eps_t  # Eqn 13 of DDIM
            x_t = x_dt
            if record_latents:
                latentList.append(x_t.cpu())
    return x_t, latentList