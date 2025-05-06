import torch as th
import numpy as np
from diffusion import gaussian_diffusion as gd
from diffusion import SpacedDiffusion, space_timesteps
from diffusion.gaussian_diffusion import _extract_into_tensor

# gaussian_diffusion should modified:
# 1) p_mean_variance: pack "raw_output" into output dict
# 2) p_sample: a.enable external reparameterization noise injection; 
#              b.pack "extra"(macs) & "raw_output" into output dict for adapter loss computation
# 3) ddim_sample: the same as p_sample

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class SpSpacedDiffusion(SpacedDiffusion):
    '''
    def __init__(self, use_timesteps, **kwargs):
        super.__init__(use_timesteps, **kwargs)
    '''
    '''
    # p_mean_variance():           
    call forward function, compute mean & var
    # p_sample():                  
    call p_mean_variance, generate sample of one step
    # p_sample_loop_progressive():
    call p_sample, the step generator of overall denoise process, yield the same output format as p_sample
    # p_sample_loop():            
    call p_sample_loop_progressive, output final sample & intermediate activations

    #
    alpha_bar: alpha_bar_t
    alpha_bar_prev: alpha_bar_t-1
    '''
    # DDPM
    
    
    # DDIM
    def ddim_sample_loop_sp(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        # Accumulate ratio of macs
        final = None
        macs = 0
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
            tmp_macs = sample["extra"]
            macs += sample["extra"].mean()    #(1)
        
        avg_macs = macs / self.num_timesteps
        return final["sample"], avg_macs
    
    def ddim_train_adapter_loop_progressive_ddp(
        self,
        warp_model,                      #must wrapped for DDP
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        amp_device_type="cuda",
        amp_dtype = th.float16,
        iter_in_mode="share_sp",          #iteration input mode
        mask_steps=[]                     #steps that are always dense
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None: 
            img = noise
        else:
            img = th.randn(*shape, device=device)
        # Share initial noised image
        sp_img = img
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        with th.autocast(device_type=amp_device_type, dtype=amp_dtype):
            for i in indices:
                t = th.tensor([i] * shape[0], device=device)
                # Share reparameterization noise
                reparam_noise = th.randn_like(img)
                # Forward both model & sp_model
                if(i in mask_steps):
                    model_kwargs["mode"] = ""
                else:
                    model_kwargs["mode"] = "sp"
                sp_out = self.ddim_sample(warp_model, sp_img, t, clip_denoised=clip_denoised,
                                          denoised_fn=denoised_fn, cond_fn=cond_fn,
                                          model_kwargs=model_kwargs, eta=eta,
                                          noise=reparam_noise)
                with th.no_grad():
                    model_kwargs["mode"] = ""
                    out = self.ddim_sample(warp_model, img, t, clip_denoised=clip_denoised,
                                           denoised_fn=denoised_fn, cond_fn=cond_fn,
                                           model_kwargs=model_kwargs, eta=eta,
                                           noise=reparam_noise)
                # Sample out handling
                sample, sp_sample = out["sample"], sp_out["sample"]     
                raw_output, sp_raw_output = out["raw_output"], sp_out["raw_output"]        
                macs_sp = sp_out["extra"]             #(batch_size)
                # Quality Loss 
                #loss_qa = mean_flat((sample - sp_sample) ** 2)
                loss_qa = mean_flat((raw_output - sp_raw_output) ** 2)
                # Step out
                loss_out = {"loss_qa": loss_qa, "macs_sp": macs_sp}
                sample_out = {"sample": sample, "sp_sample": sp_sample}
                # NN input re-assign
                if(iter_in_mode == "share_sp"):
                    img, sp_img = sp_sample.detach(), sp_sample.detach()
                elif(iter_in_mode == "share_dns"):
                    img, sp_img = sample.detach(), sample.detach()
                elif(iter_in_mode == "unique"):
                    img, sp_img = sample.detach(), sp_sample.detach()
                else:
                    raise NotImplementedError(iter_in_mode)
                yield loss_out, sample_out

def create_diffusion(
    timestep_respacing,
    noise_schedule="linear", 
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpSpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type
        # rescale_timesteps=rescale_timesteps,
    )
    