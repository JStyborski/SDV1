import torch

def gen_image_from_noise(noise_batch, model=None, cond=None, uncond=None, alphas_cumprod=None, forward_latent_sample_list=None):
    num_timesteps = len(alphas_cumprod)
    steps = 100
    skip = num_timesteps // steps
    seq_0_to_T = range(0, num_timesteps, skip)
    seq = list(reversed(seq_0_to_T))
    msg = f"gen_image_from_noise()seq=[{seq[0]}~{seq[-1]}], len={len(seq)}"
    b_sz = len(noise_batch)
    x_t = noise_batch
    seq2 = seq[1:] + [-1]

    latent_distance_list = []
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float32):
            for index, (i, j) in enumerate(zip(seq, seq2)):
                at = alphas_cumprod[i + 1]  # alpha_bar_t
                aq = alphas_cumprod[j + 1]  # alpha_bar_{t-1}
                mt = at / aq
                t = (torch.ones(b_sz).cuda() * i)
                x_out = model.apply_model(x_t.repeat(2, 1, 1, 1), t, cond=torch.cat([cond, uncond], dim=0))  # epsilon_t
                # et = x_out[1] + 9*(x_out[0] - x_out[1])
                et = x_out[0].unsqueeze(0)

                # if b_idx == 0:
                #     v, m = torch.var_mean(et)
                #     log_info(f"{msg}; ts={i:03d}, ab:{at:.6f}, aq:{aq:.6f}. epsilon var:{v:.4f}, mean:{m:7.4f}")
                xt_next = (x_t - (1 - at).sqrt() * et) / mt.sqrt() + (1 - aq).sqrt() * et
                x_t = xt_next

                latent_distance_list.append(torch.nn.functional.mse_loss(forward_latent_sample_list[-(index + 1)].cuda(), x_t).item())
    import matplotlib.pyplot as plt
    plt.figure(1)
    x_data = list(torch.arange(len(latent_distance_list)))
    y_data = latent_distance_list
    plt.plot(x_data, y_data, )
    plt.title('loss record')

    plt.savefig('latent_distance_list.png')

    return x_t


def gen_noise_from_image(image_batch, model=None, cond=None, uncond=None, alphas_cumprod=None):
    num_timesteps = len(alphas_cumprod)
    steps = 100
    skip = num_timesteps // steps
    seq_0_to_T = range(0, num_timesteps, skip)

    seq = list(seq_0_to_T)
    print(num_timesteps, seq)
    seq2 = seq[1:] + [num_timesteps - 1]
    b_sz = len(image_batch)
    xt = image_batch

    forward_latent_sample_list = []
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float32):
            for index, (i, j) in enumerate(zip(seq, seq2)):
                ab_tt = alphas_cumprod[i]  # alpha_bar_t
                ab_t1 = alphas_cumprod[j]  # alpha_bar_{t+1}
                a_t1 = ab_t1 / ab_tt  # alpha_{t+1}
                t = (torch.ones(b_sz).cuda() * i)
                # et = model(xt, t) # epsilon_t
                x_out = model.apply_model(xt.repeat(2, 1, 1, 1), t, cond=torch.cat([cond, uncond], dim=0))
                # et = (x_out[1] + 9*(x_out[0] - x_out[1])).unsqueeze(0)
                et = x_out[0].unsqueeze(0)

                # if b_idx == 0:
                #     msg = f"gen_noise_from_image()seq=[{seq[0]}~{seq[-1]}], len={len(seq)}"
                #     v, m = torch.var_mean(et)
                #     log_info(f"{msg}; ts={t[0]:3.0f}, ab[{i:3d}]:{ab_tt:.6f}, ab[{j:3d}]:{ab_t1:.6f}, "
                #              f"a[{j:3d}]:{a_t1:.6f}. epsilon var:{v:.4f}, mean:{m:7.4f}")
                noise_coef = (1 - ab_t1).sqrt() - (a_t1 - ab_t1).sqrt()
                xt_next = a_t1.sqrt() * xt + noise_coef * et
                xt = xt_next

                forward_latent_sample_list.append(xt.detach().cpu())
                # if index > 10:
                #     break
    return xt, forward_latent_sample_list
    # return et


def denoise_by_2noise2image(dl, model=None, cond=None, uncond=None, callback=None, disable=None, eta=0.0):
    timesteps = torch.arange(1, 1000, 1).cuda()
    alphas_cumprod = model.alphas_cumprod
    alphas = alphas_cumprod[timesteps]

    alphas_prev = alphas_cumprod[torch.nn.functional.pad(timesteps[:-1], pad=(1, 0))].to(torch.float64)
    alphas_next = alphas_cumprod[torch.nn.functional.pad(timesteps[1:], pad=(1, 0))].to(torch.float64)
    sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
    sigmas = eta * np.sqrt((1 - alphas_prev.cpu().numpy()) / (1 - alphas.cpu()) * (1 - alphas.cpu() / alphas_prev.cpu().numpy()))

    cond = model.cond_stage_model([''])
    uncond = model.cond_stage_model([''])

    model.first_stage_model.float()

    with torch.no_grad():
        for index, entry in enumerate(dl):
            x = entry.latent_sample.to('cuda')

            gen_e_T = gen_noise_from_image(image_batch=x, model=model, cond=cond, uncond=uncond, alphas_cumprod=alphas_cumprod)
            # print('x size', x.size(), 'et size', e_T.size())
            # x = gen_image_from_noise(noise_batch=e_T, model=model, cond=cond, uncond=uncond, alphas_cumprod=alphas_cumprod)

            # e_T = torch.randn(x.size()).cuda()
            # x = gen_image_from_noise(noise_batch=e_T, model=model, cond=cond, uncond=uncond, alphas_cumprod=alphas_cumprod)
            # gen_e_T = gen_noise_from_image(image_batch=x, model=model, cond=cond, uncond=uncond, alphas_cumprod=alphas_cumprod)
            gen_x = gen_image_from_noise(noise_batch=gen_e_T, model=model, cond=cond, uncond=uncond, alphas_cumprod=alphas_cumprod)

            # print('distance between latent vector', torch.square(e_T - gen_e_T).mean().item())

            x_samples_ddim, forward_latent_sample_list = model.decode_first_stage(x)
            gen_x_samples_ddim = model.decode_first_stage(gen_x, forward_latent_sample_list)

            image_to_save = torch.cat([entry.torchdata, x_samples_ddim, gen_x_samples_ddim], dim=-1)
            dir_to_save = 'img2noise2img_examples_mist_JuliaSPowell_deterministic'
            os.makedirs(dir_to_save, exist_ok=True)
            assert cv2.imwrite(os.path.join(dir_to_save, str(index) + '_forward_and_reverse_inference_output_during_IT.jpg'),
                               ((torch.clamp(image_to_save, min=-1, max=1) / 2 + 0.5) * 255)[0].detach().cpu().numpy().transpose(1, 2, 0)[..., ::-1])
            # exit()
            continue

            # intermediate_output = adversarial_ddim(shared.sd_model, x, timesteps, cond=cond, uncond=uncond)
            # x_samples_ddim = processing.decode_latent_batch(shared.sd_model, intermediate_output, target_device=devices.cpu, check_for_nans=True)

            intermediate_output = x
            x_samples_ddim = model.decode_first_stage(intermediate_output)
            original_x_samples_ddim = model.decode_first_stage(entry.latent_sample.to('cuda'))

            print('distance between latent vector', torch.square(x - entry.latent_sample.cuda()).mean().item())

            image_to_save = torch.cat([x_samples_ddim, original_x_samples_ddim, entry.torchdata], dim=-1)
            cv2.imwrite('forward_and_reverse_inference_output_during_IT.jpg',
                        ((torch.clamp(image_to_save, min=-1, max=1) / 2 + 0.5) * 255)[0].detach().cpu().numpy().transpose(1, 2, 0)[..., ::-1])

            exit()
            # for x_sample in x_samples_ddim:
            #     print('x_sample after inference ', x_sample.size(), x_sample.min(), x_sample.max(), ((torch.clamp(x_sample, min = -1, max = 1)/2 + 0.5)*255).max(), ((torch.clamp(x_sample, min = -1, max = 1)/2 + 0.5)*255).min())
            #     cv2.imwrite('forward_and_reverse_inference_output_during_IT.jpg', ((torch.clamp(x_sample, min = -1, max = 1)/2 + 0.5)*255).detach().cpu().numpy().transpose(1,2,0)[...,::-1])

    return x