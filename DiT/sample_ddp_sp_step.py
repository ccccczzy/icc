# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from download import find_model
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
from diffusers.models import AutoencoderKL
from diffusion import create_diffusion
from models.models_sp_step_decomp import DiT_models
from func.func_decomp import DiT_set_decomp_weight
from adapters.sched_sp_step import get_sched
from utils.utils_train import requires_grad, num_param


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def main(args):
    # Setup torch
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Load pre-trained DiT model:
    if args.ckpt is None:
        assert args.model_name == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000
    latent_size = args.image_size // 8
    patch_size = int(args.model_name[-1])
    num_tokens = (latent_size // patch_size)**2         #256->256, 512->1024
    model = DiT_models[args.model_name](input_size=latent_size, num_classes=args.num_classes, use_inc=args.use_inc)
    
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    num_blocks = len(model.blocks)
    in_channels = model.in_channels
    
    # Load & attach adapters
    if(args.is_load_decomp):
        ckpt_adapter = torch.load(args.results_dir)
        
    # Load & attach scheduler    
    sched = get_sched(args.num_steps, args.sched_type, args.sched_kwargs)
    model.sched_attach(sched)
    model = model.to(device)
    if(rank == 0):
        num_param_model = num_param(model)
    
    # Load lowrank weights
    if(args.is_load_decomp):
        if(rank == 0):
            print(f"Beginning to load lowrank weight.")
        DiT_set_decomp_weight(model, ckpt_adapter["decomp_weight"], device)      
    else:
        if(rank == 0):
            print(f"Do not set decomposed weight.")
    requires_grad(model, False)
    model.eval()
    
    # Num of param
    if(rank == 0):
        num_param_decomp = num_param(model) - num_param_model
        ratio_decomp = num_param_decomp / num_param_model
        print(f"Parameter number of model:\t\t{num_param_model:.2f}\tM")
        print(f"Parameter number of decomposed weight:\t{num_param_decomp:.2f}\tM")
        print(f"Ratio of decomposed weight parameter:\t{ratio_decomp:.4f}")
    
    # Create diffusion
    diffusion = create_diffusion(str(args.num_steps))
    # Solver selection
    if(args.solver == "ddpm"):
        sample_loop = diffusion.p_sample_loop
    elif(args.solver == "ddim"):
        sample_loop = diffusion.ddim_sample_loop
    else:
        raise NotImplementedError(args.solver)
    
    # Load pre-trained VAE
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model_name.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    # Modify: add solver details to folder_name
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
                  f"{args.solver}{args.num_steps}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.samples_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for _ in pbar:
        # Sync clear cache
        model.sched.clr()
        
        # Sample inputs:
        z = torch.randn(n, in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            sample_fn = model.forward_with_cfg_sp
        else:
            model_kwargs = dict(y=y)
            sample_fn = model.forward_sp

        # Sample images:
        samples = sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            
        if args.is_save:
            samples = vae.decode(samples / 0.18215).sample
            samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = i * dist.get_world_size() + rank + total
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
            total += global_batch_size
        
        
    # Global merge of ratio of MACs
    macs_ref = model.macs_ref_item()
    if(getattr(model, "macs_sp_item", None) != None):
        macs_sp = model.macs_sp_item()
    else:
        macs_sp = macs_ref
    ratio_macs = macs_sp / macs_ref
    if rank == 0:
        print(f"Ratio of MACs: {ratio_macs: .4f}")
        
    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0 and args.is_save:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path
    parser.add_argument("--is-load-decomp", type=bool, default=True) 
    parser.add_argument("--results-dir", type=str, default="./results_decomp/ASVD128/checkpoints/0000000.pt")
    parser.add_argument("--samples-dir", type=str, default="./samples_sp")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    # Model var
    parser.add_argument("--model_name", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    # Sched var
    parser.add_argument("--sched-type", type=str, default="PERIOD")  
    parser.add_argument("--sched-kwargs", type=dict, default=dict(period=2))      #For "PERIOD"                  
    # Adapter-related var for model   
    parser.add_argument("--use-inc", type=bool, default=True)
    # Sampling var
    parser.add_argument("--solver", type=str, default="ddim")
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    # Other    
    parser.add_argument("--tf32", default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--per-proc-batch-size", type=int, default=8)   #16 for 256, 4 for 512 @ 4090D
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--is-save", type=bool, default=True)
    args = parser.parse_args()
    main(args)
