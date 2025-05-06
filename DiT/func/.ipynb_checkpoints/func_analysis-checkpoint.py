import torch
from utils.utils_jacob import deriv_gelu
import matplotlib.pyplot as plt

path = "./analysis/"

'''
def DiT_analysis_mlp_act(model, thr_zero=0.2):
    for idx_block, block in enumerate(model.blocks):
        # Plot distribution of GELU input
        dist_act_x = block.mlp.dist_act_i
        boundaries = dist_act_x.boundaries.cpu().numpy()
        hist = dist_act_x.hist.cpu().numpy()
        plt.plot(boundaries, hist)
        # Number of zero
        deriv = deriv_gelu(torch.tensor(boundaries))
        mask_zero = deriv.abs() < thr_zero
        num_zero = (mask_zero * hist).sum().item()
        ratio_zero = num_zero / hist.sum().item()
        value_ratio_zero = (mask_zero * hist * deriv).sum().item() / (hist * deriv).sum().item()
        print(f"Block{idx_block}: Ratio: {ratio_zero:.3f}")
    # Plot curve of gelu
    #x = boundaries
    #y = deriv_gelu(torch.tensor(x)).numpy()
    #plt.plot(x, y*7e7)
    plt.savefig("./analysis/analysis_mlp_act.png")
'''

    
def DiT_analysis_dist_linear_i(model):
    with open("./analysis/dist.txt", "a", encoding="utf-8") as file:
        print_func = file.write
        for idx_block, block in enumerate(model.blocks):
            plt.clf()
            dist = block.attn.dist_qkv_i
            #dist.plt(path + f"./dist_qkv_i{idx_block:02d}.png") 
            mean = dist.mean()
            var = dist.var()
            max_val = dist.true_max_val
            min_val = dist.true_min_val
            print_func(f"\n qkv Block:{idx_block:02d} mean:{mean:.3f} var:{var:.3f} max:{max_val:.3f} min:{min_val:.3f}")

        for idx_block, block in enumerate(model.blocks):
            plt.clf()
            dist = block.attn.dist_proj_i
            #dist.plt(path + f"./dist_proj_i{idx_block:02d}.png")
            mean = dist.mean()
            var = dist.var()
            max_val = dist.true_max_val
            min_val = dist.true_min_val
            print_func(f"\n proj Block:{idx_block:02d} mean:{mean:.3f} var:{var:.3f} max:{max_val:.3f} min:{min_val:.3f}")

        for idx_block, block in enumerate(model.blocks):
            plt.clf()
            dist = block.mlp.dist_fc1_i
            #dist.plt(path + f"./dist_fc1_i{idx_block:02d}.png")
            mean = dist.mean()
            var = dist.var()
            max_val = dist.true_max_val
            min_val = dist.true_min_val
            print_func(f"\n fc1 Block:{idx_block:02d} mean:{mean:.3f} var:{var:.3f} max:{max_val:.3f} min:{min_val:.3f}")

        for idx_block, block in enumerate(model.blocks):
            plt.clf()
            dist = block.mlp.dist_fc2_i
            #dist.plt(path + f"./dist_fc1_i{idx_block:02d}.png")
            mean = dist.mean()
            var = dist.var()
            max_val = dist.true_max_val
            min_val = dist.true_min_val
            print_func(f"\n fc2 Block:{idx_block:02d} mean:{mean:.3f} var:{var:.3f} max:{max_val:.3f} min:{min_val:.3f}")
        
        
        
    
    
        
    