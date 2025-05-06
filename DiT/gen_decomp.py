import argparse
import os
from glob import glob
from download import find_model

import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler
from diffusers.models import AutoencoderKL
from diffusion import create_diffusion
from models.models_stat_svd import DiT_models
from func.func_decomp import DiT_wtv_src_stat_loop, DiT_wtv_calc, DiT_wsvd_merge_lowrank, DiT_get_decomp_weight
from func.func_decomp import DiT_get_wtv, DiT_set_wtv, DiT_print_wtv
from utils.utils_train import create_logger, num_param, center_crop_arr


def main(args):
    # Setup torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    
    # Load pre-trained DiT model
    latent_size = args.image_size // 8    #256->32, 512->64 
    model = DiT_models[args.model_name](input_size=latent_size, num_classes=args.num_classes).to(device)
    ckpt_path = f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    num_blocks = len(model.blocks)
    num_param_model = num_param(model)
    num_tokens = (latent_size // int(args.model_name[-1]))**2
    
    # Generate Weight Vector for Weighted Singular Value Decomposition
    if(args.use_wtv == True):
        model_name_string = args.model_name.replace("/", "-")
        wtv_name = f"{model_name_string}-size{args.image_size}-sample{args.num_samples}-epoch{args.num_epochs}"
        wtv_path_save = args.wtv_dir + "/" + wtv_name + f"-step{args.num_steps}" + f"-{args.wtv_src}" + f"-left{args.ena_lwtv}-ratio{args.wtv_max_ratio_left}-right{args.ena_rwtv}-ratio{args.wtv_max_ratio_right}" + ".pt"
        wtv_path_load = wtv_path_save
        # Load weight vector
        if(args.is_load_wtv == True):
            print("Beginning to load weight vector.")
            wtv = torch.load(wtv_path_load)
            DiT_set_wtv(model, wtv, device)
        else:
            print("Beginning to generate weight vector.")
            # Setup diffusion
            diffusion = create_diffusion(str(args.num_steps))
            # Load pre-trained VAE
            #vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path = "./pretrained_vae").to(device)
            vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
            # Setup dataloader
            transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
            dataset = ImageFolder(args.data_path, transform=transform)
            if(args.num_samples is None):
                dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
            else:
                sampler = RandomSampler(dataset, num_samples=args.num_samples, replacement=False)
                dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler, 
                                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
            # Calcaulate Fisher Information Matrix
            DiT_wtv_src_stat_loop(model, args.wtv_src, vae, dataloader, diffusion, num_epochs=args.num_epochs, spec_range=args.spec_range, device="cuda")
            
            # Calculate Weight for SVD
            DiT_wtv_calc(model, args.wtv_src, args.wtv_max_ratio_left, args.wtv_max_ratio_right)
            
            # Save weight vector
            if(args.is_save_wtv == True):
                print("Beginning to save weight vector.")
                wtv = DiT_get_wtv(model, device)
                torch.save(wtv, wtv_path_save)
        DiT_print_wtv(model)
                      
    # Search rank
    if(args.use_search == True):
        pass
    else:
        print("Beginning to set rank mannually.")
        keys = ["q", "k", "v", "proj", "fc1", "fc2"]
        rank_lists = {key: [] for key in keys}
        rank_lists["q"] = [args.rank for _ in range(num_blocks)]
        rank_lists["k"] = [args.rank for _ in range(num_blocks)]
        rank_lists["v"] = [args.rank for _ in range(num_blocks)]
        rank_lists["proj"] = [args.rank for _ in range(num_blocks)]
        rank_lists["fc1"] = [args.rank for _ in range(num_blocks)]
        rank_lists["fc2"] = [args.rank for _ in range(num_blocks)]
 
    # Weighted Singular Value Decomposition
    energy_lists, abs_error_lists, rel_error_lists = DiT_wsvd_merge_lowrank(model, rank_lists, args.ena_lwtv, args.ena_rwtv)
    
    # Export lowrank weight
    decomp_weight = DiT_get_decomp_weight(model, "cpu")
    
    # Save
    if(args.is_save):
        # Setup experimental folder
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model_name.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(f"{args}")
        # Save
        ckpt = {"decomp_weight": decomp_weight}
        torch.save(ckpt, f"{checkpoint_dir}/{0:07d}.pt")
    
    # Log number of parameters
    num_param_decomp = num_param(model) - num_param_model
    ratio_param_decomp = num_param_decomp / num_param_model
    macs_ref = model.macs_ref_item()
    macs_decomp = model.macs_decomp_item()
    ratio_macs_decomp = macs_decomp / macs_ref
    keys = ["q", "k", "v", "proj", "fc1", "fc2"]
    if(args.is_save):
        print_func = logger.info
    else:
        print_func = print
        
    print_func(f"Parameter number of model:\t\t{num_param_model:.2f}\tM")
    print_func(f"Parameter number of decomposed weight:\t{num_param_decomp:.2f}\tM")
    print_func(f"Ratio of decomposed weight parameter:\t{ratio_param_decomp:.4f}")
    print_func(f"Ratio of decomposed computation:\t{ratio_macs_decomp}")
    for key in keys:
        print_func(f"PCA accumulative energy of {key}:")
        print_func([f"{elem:.2f}" for elem in energy_lists[key]])
    for key in keys:
        print_func(f"Absolute reconstruction error of {key}:")
        print_func([f"{elem:.4f}" for elem in abs_error_lists[key]])
    for key in keys:
        print_func(f"Relative reconstruction error of {key}:")
        print_func([f"{elem:.4f}" for elem in rel_error_lists[key]])
    
    return model
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--is-save", type=bool, default=True)
    # Path
    parser.add_argument("--results-dir", type=str, default="./results_decomp")
    parser.add_argument("--wtv-dir", type=str, default="./wtv")
    # Model var
    parser.add_argument("--model_name", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    # Dataset var
    parser.add_argument("--data-path", type=str, default="~/autodl-tmp/imagenet/train")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=1)
    # Diffusion var
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--spec-range", type=list, default=None)
    # SVD Weight Vector calculation
    parser.add_argument("--use-wtv", type=bool, default=False)
    parser.add_argument("--wtv-src", type=str, default="mag")       # "mag", "delta_mag"
    parser.add_argument("--wtv-max-ratio-left", type=float, default=1000)
    parser.add_argument("--wtv-max-ratio-right", type=float, default=1000)
    parser.add_argument("--ena-lwtv", type=bool, default=True)
    parser.add_argument("--ena-rwtv", type=bool, default=True)
    parser.add_argument("--is-load-wtv", type=bool, default=False)  # only used when use_wtv is True
    parser.add_argument("--is-save-wtv", type=bool, default=False)  # only used when use_wtv is True and is_load_wtv is False
    # SVD rank search  
    parser.add_argument("--use-search", type=bool, default=False)               # Always False
    # Mannally rank setting, mannually set rank, only used when use_search is False
    parser.add_argument("--rank", type=int, default=128)   
    # Other
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    model = main(args)