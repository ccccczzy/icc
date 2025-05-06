import torch
import torch.nn as nn
from utils.utils_decomp import fim_calc, wsvd, wsvd_merge_lowrank
from utils.utils_decomp import sparse_metric, shift2ratio

merge_qkv_wb = False

# Fisher Information Matrix        
def DiT_fim_calc(model, idx_iter):
    # Calculate Fisher Information Matrix
    for block in model.blocks:
        attn, mlp = block.attn, block.mlp
        # Attn
        qkv_fim = fim_calc(getattr(attn, "qkv_fim", 0), attn.qkv.weight, idx_iter)     #(3*hidden_size, hidden_size)
        proj_fim = fim_calc(getattr(attn, "proj_fim", 0), attn.proj.weight, idx_iter)  #(hidden_size, hidden_size)
        setattr(attn, "qkv_fim", qkv_fim)
        setattr(attn, "proj_fim", proj_fim)
        # Mlp
        fc1_fim = fim_calc(getattr(attn, "fc1_fim", 0), mlp.fc1.weight, idx_iter)      #(mlp_ratio*hidden_size, hidden_size)
        fc2_fim = fim_calc(getattr(attn, "fc2_fim", 0), mlp.fc2.weight, idx_iter)      #(hidden_size, mlp_ratio*hidden_size)
        setattr(mlp, "fc1_fim", fc1_fim)
        setattr(mlp, "fc2_fim", fc2_fim)
        
        
def DiT_wtv_src_stat_loop_fim(model, vae, dataloader, diffusion, num_epochs=1, accu_every_iter=2, device="cuda"):
    # Compute the FIM
    model.train()
    # Setup optimizer to clear gradient
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    # Loop
    idx_iter = 0
    for idx_epoch in range(num_epochs):
        print(f"epoch: {idx_epoch}")
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            with torch.enable_grad():
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                model_kwargs = dict(y=y)
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean() / accu_every_iter
                # Calculate and accumulate gradient
                loss.backward()
                if((idx_iter + 1) % accu_every_iter == 0 or idx_iter == (len(dataloader) - 1)):
                    print(f"Iteration{idx_iter:08d} FIM calculation is finished.")
                    # Accumulate FIM
                    DiT_fim_calc(model, idx_iter)
                    # Clear gradient
                    optimizer.zero_grad(set_to_none=True)
                idx_iter = idx_iter + 1
            
def DiT_wtv_src_stat_loop_mag(model, vae, dataloader, diffusion, num_epochs=1, spec_range=None, device="cuda"):
    # Stat the mean magnitude
    label_dropout_prob = 0.1
    model.eval()
    for idx_epoch in range(num_epochs):
        print(f"epoch: {idx_epoch}")
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            if(spec_range == None):
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            else:
                t = torch.randint(spec_range[0], spec_range[1], (x.shape[0],), device=device)
            # Encode x to latent space
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            # Label dropout
            ids_label_dropout = torch.rand(y.shape[0], device=y.device) < label_dropout_prob            
            y = torch.where(ids_label_dropout, model.y_embedder.num_classes, y)
            model_kwargs = dict(y=y, stat_flag="mag")
            noise = torch.randn_like(x)
            x_t = diffusion.q_sample(x, t, noise=noise) 
            x_next = diffusion.ddim_sample(model, x_t, t, clip_denoised=False, model_kwargs=model_kwargs)["sample"]

def DiT_wtv_src_stat_loop_delta_mag(model, vae, dataloader, diffusion, num_epochs=1, device="cuda"):
    label_dropout_prob = 0.1
    model.eval()
    for idx_epoch in range(num_epochs):
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            # Denoise process definition: t2 --> t1
            t2 = torch.randint(1, diffusion.num_timesteps, (x.shape[0],), device=device)
            # Encode x to latent space
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            # Label dropout
            ids_label_dropout = torch.rand(y.shape[0], device=y.device) < label_dropout_prob 
            y = torch.where(ids_label_dropout, model.y_embedder.num_classes, y)
            model_kwargs = dict(y=y)
            # Add noise
            noise = torch.randn_like(x)
            x_t2 = diffusion.q_sample(x, t2, noise=noise)
            # Sample x_t1, cache values
            model_kwargs["stat_flag"] = "cache"
            x_t1 = diffusion.ddim_sample(model, x_t2, t2, clip_denoised=False, model_kwargs=model_kwargs)["sample"]
            # Denoising x_t1, accumulate delta mag
            model_kwargs["stat_flag"] = "delta_mag"
            t1 = t2 - 1
            x_t0 = diffusion.ddim_sample(model, x_t1, t1, clip_denoised=False, model_kwargs=model_kwargs)["sample"]
            
def DiT_wtv_src_stat_loop(model, wtv_src, vae, dataloader, diffusion, num_epochs=1, spec_range=None, device="cuda"):
    if(wtv_src == "fim"):
        DiT_wtv_src_stat_loop_fim(model, vae, dataloader, diffusion, num_epochs, 2, device)
    elif(wtv_src == "mag"):
        DiT_wtv_src_stat_loop_mag(model, vae, dataloader, diffusion, num_epochs, spec_range, device)
    elif(wtv_src == "delta_mag"):
        DiT_wtv_src_stat_loop_delta_mag(model, vae, dataloader, diffusion, num_epochs, device)
    else:
        raise NotImplementError(wtv_src)
                  
# Weight Vector
def DiT_wtv_calc_fim(model):
    # Calculate Weight Vector from FIM
    # Both left (output channel) weight vector & right (input channel) weight vector can be obtained 
    for block in model.blocks:
        attn, mlp = block.attn, block.mlp
        # Attn
        q_fim, k_fim, v_fim = attn.qkv_fim.chunk(3, dim=0)
        setattr(attn, "q_lwtv", q_fim.sum(dim=-1).sqrt())                #(hidden_size)
        setattr(attn, "k_lwtv", k_fim.sum(dim=-1).sqrt())                #(hidden_size)
        setattr(attn, "v_lwtv", v_fim.sum(dim=-1).sqrt())                #(hidden_size)
        setattr(attn, "proj_lwtv", attn.proj_fim.sum(dim=-1).sqrt())     #(hidden_size)
        setattr(attn, "q_rwtv", q_fim.sum(dim=0).sqrt())                 #(hidden_size)
        setattr(attn, "k_rwtv", k_fim.sum(dim=0).sqrt())                 #(hidden_size)
        setattr(attn, "v_rwtv", v_fim.sum(dim=0).sqrt())                 #(hidden_size)
        setattr(attn, "proj_rwtv", attn.proj_fim.sum(dim=0).sqrt())      #(hidden_size)
        # Mlp
        setattr(mlp, "fc1_lwtv", mlp.fc1_fim.sum(dim=-1).sqrt())         #(mlp_ratio*hidden_size)
        setattr(mlp, "fc2_lwtv", mlp.fc2_fim.sum(dim=-1).sqrt())         #(hidden_size)
        setattr(mlp, "fc1_rwtv", mlp.fc1_fim.sum(dim=0).sqrt())          #(hidden_size)
        setattr(mlp, "fc2_rwtv", mlp.fc2_fim.sum(dim=0).sqrt())          #(mlp_ratio*hidden_size)      
         
def DiT_wtv_calc_mag(model, max_ratio_left, max_ratio_right):
    # Calculate Weight Vector from average magnitude
    alpha = 0.5
    for block in model.blocks:
        attn, mlp = block.attn, block.mlp
        # Right (input channel)
        qkv_rwtv = shift2ratio((attn.qkv_x_mag / attn.qkv_x_cnt) ** alpha, max_ratio_right)
        proj_rwtv = shift2ratio((attn.proj_x_mag / attn.proj_x_cnt) ** alpha, max_ratio_right)
        fc1_rwtv = shift2ratio((mlp.fc1_x_mag / mlp.fc1_x_cnt) ** alpha, max_ratio_right)
        fc2_rwtv = shift2ratio((mlp.fc2_x_mag / mlp.fc2_x_cnt) ** alpha, max_ratio_right)
        # Left (output channel)
        qkv_lwtv = (attn.qkv_y_mag / attn.qkv_y_cnt) ** alpha
        q_lwtv, k_lwtv, v_lwtv = torch.chunk(qkv_lwtv, chunks=3)
        q_lwtv = shift2ratio(q_lwtv, max_ratio_left)
        k_lwtv = shift2ratio(k_lwtv, max_ratio_left)
        v_lwtv = shift2ratio(v_lwtv, max_ratio_left)
        proj_lwtv = shift2ratio((attn.proj_y_mag / attn.proj_y_cnt) ** alpha, max_ratio_left)
        fc1_lwtv = shift2ratio((mlp.fc1_y_mag / mlp.fc1_y_cnt) ** alpha, max_ratio_left)
        fc2_lwtv = shift2ratio((mlp.fc2_y_mag / mlp.fc2_y_cnt) ** alpha, max_ratio_left)
        # Attn
        setattr(attn, "q_rwtv", qkv_rwtv)         #(hidden_size)
        setattr(attn, "k_rwtv", qkv_rwtv)         #(hidden_size)
        setattr(attn, "v_rwtv", qkv_rwtv)         #(hidden_size)
        setattr(attn, "proj_rwtv", proj_rwtv)     #(hidden_size)
        setattr(attn, "q_lwtv", q_lwtv)           #(hidden_size)
        setattr(attn, "k_lwtv", k_lwtv)           #(hidden_size)
        setattr(attn, "v_lwtv", v_lwtv)           #(hidden_size)
        setattr(attn, "proj_lwtv", proj_lwtv)     #(hidden_size)
        # Mlp
        setattr(mlp, "fc1_rwtv", fc1_rwtv)        #(hidden_size)
        setattr(mlp, "fc2_rwtv", fc2_rwtv)        #(mlp_ratio*hidden_size)
        setattr(mlp, "fc1_lwtv", fc1_lwtv)        #(mlp_ratio*hidden_size)
        setattr(mlp, "fc2_lwtv", fc2_lwtv)        #(hidden_size)
        
def DiT_wtv_calc(model, wtv_src, max_ratio_left, max_ratio_right):
    if(wtv_src == "fim"):
        DiT_wtv_calc_fim(model)
    elif(wtv_src in ["mag", "delta_mag"]):
        DiT_wtv_calc_mag(model, max_ratio_left, max_ratio_right)
    else:
        raise NotImplementError(wtv_src)

def DiT_get_wtv(model, device):
    keys_attn = ["q_lwtv", "k_lwtv", "v_lwtv", "proj_lwtv", 
                 "q_rwtv", "k_rwtv", "v_rwtv", "proj_rwtv"]
    keys_mlp = ["fc1_lwtv", "fc2_lwtv", "fc1_rwtv", "fc2_rwtv"]
    wtv = {key: [] for key in (keys_attn + keys_mlp)}
    for block in model.blocks:
        attn, mlp = block.attn, block.mlp
        # Attn
        for key in keys_attn:
            tmp = getattr(attn, key, None)
            tmp = tmp.to(device) if (tmp != None) else None
            wtv[key].append(tmp)
        # Mlp
        for key in keys_mlp:
            tmp = getattr(mlp, key, None)
            tmp = tmp.to(device) if (tmp != None) else None
            wtv[key].append(tmp)
    return wtv

def DiT_set_wtv(model, wtv, device):
    keys_attn = ["q_lwtv", "k_lwtv", "v_lwtv", "proj_lwtv", 
                 "q_rwtv", "k_rwtv", "v_rwtv", "proj_rwtv"]
    keys_mlp = ["fc1_lwtv", "fc2_lwtv", "fc1_rwtv", "fc2_rwtv"]
    for idx_block, block in enumerate(model.blocks) :
        attn, mlp = block.attn, block.mlp
        # Attn
        for key in keys_attn:
            tmp = wtv[key][idx_block]
            tmp = tmp.to(device) if (tmp != None) else None
            setattr(attn, key, tmp)
        # Mlp
        for key in keys_mlp:
            tmp = wtv[key][idx_block]
            tmp = tmp.to(device) if (tmp != None) else None
            setattr(mlp, key, tmp)
            
def DiT_print_wtv(model):
    keys_attn = ["q", "k", "v", "proj"]
    keys_mlp = ["fc1", "fc2"]
    keys = keys_attn + keys_mlp
    eps = 1e-6
    for key in keys:
        print(key)
        for idx_block, block in enumerate(model.blocks):
            layer = block.attn if (key in keys_attn) else block.mlp
            tmp_rwtv = getattr(layer, key + "_rwtv", None)
            max_val = tmp_rwtv.max().item()
            min_val = tmp_rwtv.min().item()
            ratio = (max_val + eps) / (min_val + eps)
            sparsity_rate = sparse_metric(tmp_rwtv)
            print(f"Right: Block{idx_block:2d}: ratio: {ratio: .2f}\t\t max:{max_val:.2f}\t min:{min_val:.2f}\t sparse:{sparsity_rate:.3f}")
            
            tmp_lwtv = getattr(layer, key + "_lwtv", None)
            max_val = tmp_lwtv.max().item()
            min_val = tmp_lwtv.min().item()
            ratio = (max_val + eps) / (min_val + eps)
            sparsity_rate = sparse_metric(tmp_lwtv)
            print(f"Left : Block{idx_block:2d}: ratio: {ratio: .2f}\t\t max:{max_val:.2f}\t min:{min_val:.2f}\t sparse:{sparsity_rate:.3f}")
            
            

# Weighted SVD
# Export & import decomposed weight     
def DiT_get_decomp_weight(model, device="cpu"):
    if(merge_qkv_wb == True):
        keys_attn = ["q_wa", "k_wa", "v_wa", "qkv_wb", "proj_wa", "proj_wb"]
    else:
        keys_attn = ["q_wa", "q_wb", "k_wa", "k_wb", "v_wa", "v_wb", "proj_wa", "proj_wb"]
    keys_mlp = ["fc1_wa", "fc1_wb", "fc2_wa", "fc2_wb"]
    decomp_weight = {key: [] for key in (keys_attn + keys_mlp)}
    for block in model.blocks:
        attn, mlp = block.attn, block.mlp
        # MHA
        for key in keys_attn:
            decomp_weight[key].append(getattr(attn, key).data.to(device))
        # FFN
        for key in keys_mlp:
            decomp_weight[key].append(getattr(mlp, key).data.to(device))
    return decomp_weight
        
def DiT_set_decomp_weight(model, decomp_weight, device="cpu"):
    if(merge_qkv_wb == True):
        keys_attn = ["q_wa", "k_wa", "v_wa", "qkv_wb", "proj_wa", "proj_wb"]
    else:
        keys_attn = ["q_wa", "q_wb", "k_wa", "k_wb", "v_wa", "v_wb", "proj_wa", "proj_wb"]
    keys_mlp = ["fc1_wa", "fc1_wb", "fc2_wa", "fc2_wb"]
    for idx_block, block in enumerate(model.blocks) :
        attn, mlp = block.attn, block.mlp
        # MHA
        for key in keys_attn:
            setattr(attn, key, nn.Parameter(decomp_weight[key][idx_block].to(device)))
        # FFN
        for key in keys_mlp:
            setattr(mlp, key, nn.Parameter(decomp_weight[key][idx_block].to(device)))
            
def DiT_wsvd_merge_lowrank(model, rank_lists, ena_lwtv=True, ena_rwtv=True):
    keys = ["q", "k", "v", "proj", "fc1", "fc2"]
    energy_lists = {key: [] for key in keys}
    abs_error_lists = {key: [] for key in keys}
    rel_error_lists = {key: [] for key in keys}
    for idx_block, block in enumerate(model.blocks):
        attn, mlp = block.attn, block.mlp
        rank_q = rank_lists["q"][idx_block]
        rank_k = rank_lists["k"][idx_block]
        rank_v = rank_lists["v"][idx_block]
        rank_proj = rank_lists["proj"][idx_block]
        rank_fc1 = rank_lists["fc1"][idx_block]
        rank_fc2 = rank_lists["fc2"][idx_block]
        # Attn
        q_lwtv = getattr(attn, "q_lwtv", None) if ena_lwtv else None
        k_lwtv = getattr(attn, "k_lwtv", None) if ena_lwtv else None
        v_lwtv = getattr(attn, "v_lwtv", None) if ena_lwtv else None
        proj_lwtv = getattr(attn, "proj_lwtv", None) if ena_lwtv else None
        q_rwtv = getattr(attn, "q_rwtv", None) if ena_rwtv else None
        k_rwtv = getattr(attn, "k_rwtv", None) if ena_rwtv else None
        v_rwtv = getattr(attn, "v_rwtv", None) if ena_rwtv else None
        proj_rwtv = getattr(attn, "proj_rwtv", None) if ena_rwtv else None
        q_w, k_w, v_w = attn.qkv.weight.data.chunk(3, dim=0) 
        q_wa, q_wb, q_energy, q_abs_error, q_rel_error = wsvd_merge_lowrank(q_w, rank_q, q_lwtv, q_rwtv)     
        k_wa, k_wb, k_energy, k_abs_error, k_rel_error = wsvd_merge_lowrank(k_w, rank_q, k_lwtv, k_rwtv)
        v_wa, v_wb, v_energy, v_abs_error, v_rel_error = wsvd_merge_lowrank(v_w, rank_v, v_lwtv, v_rwtv)
        proj_wa, proj_wb, proj_energy, proj_abs_error, proj_rel_error = wsvd_merge_lowrank(attn.proj.weight.data, rank_proj, proj_lwtv, proj_rwtv)
        setattr(attn, "q_wa", nn.Parameter(q_wa))                        #(hidden_size, rank_q)
        setattr(attn, "k_wa", nn.Parameter(k_wa))                        #(hidden_size, rank_k)
        setattr(attn, "v_wa", nn.Parameter(v_wa))                        #(hidden_size, rank_v)
        if(merge_qkv_wb == True):
            qkv_wb = torch.cat([q_wb, k_wb, v_wb], dim=0).contiguous()
            setattr(attn, "qkv_wb", nn.Parameter(qkv_wb))                #(rank_q+rank_k+rankv, hidden_size)
        else:
            setattr(attn, "q_wb", nn.Parameter(q_wb))                    #(rank_q, hidden_size)
            setattr(attn, "k_wb", nn.Parameter(k_wb))                    #(rank_k, hidden_size)
            setattr(attn, "v_wb", nn.Parameter(v_wb))                    #(rank_v, hidden_size)
        setattr(attn, "proj_wa", nn.Parameter(proj_wa))                  #(hidden_size, rank_proj)
        setattr(attn, "proj_wb", nn.Parameter(proj_wb))                  #(rank_proj, hidden_size)
        
        # Mlp
        fc1_lwtv = getattr(mlp, "fc1_lwtv", None) if ena_lwtv else None
        fc2_lwtv = getattr(mlp, "fc2_lwtv", None) if ena_lwtv else None
        fc1_rwtv = getattr(mlp, "fc1_rwtv", None) if ena_rwtv else None
        fc2_rwtv = getattr(mlp, "fc2_rwtv", None) if ena_rwtv else None
        fc1_wa, fc1_wb, fc1_energy, fc1_abs_error, fc1_rel_error = wsvd_merge_lowrank(mlp.fc1.weight.data, rank_fc1, fc1_lwtv, fc1_rwtv)
        fc2_wa, fc2_wb, fc2_energy, fc2_abs_error, fc2_rel_error = wsvd_merge_lowrank(mlp.fc2.weight.data, rank_fc2, fc2_lwtv, fc2_rwtv)
        setattr(mlp, "fc1_wa", nn.Parameter(fc1_wa))                     #(hidden_size, rank_fc1)
        setattr(mlp, "fc1_wb", nn.Parameter(fc1_wb))                     #(rank_fc1, hidden_size)
        setattr(mlp, "fc2_wa", nn.Parameter(fc2_wa))                     #(hidden_size, rank_fc2)
        setattr(mlp, "fc2_wb", nn.Parameter(fc2_wb))                     #(rank_fc2, hidden_size)
        # Other information
        for key in keys:
            energy_lists[key].append(locals()[key + "_energy"])
            abs_error_lists[key].append(locals()[key + "_abs_error"])
            rel_error_lists[key].append(locals()[key + "_rel_error"])
        print(f"Block{idx_block:02d} merged wsvd is finished.")
    return energy_lists, abs_error_lists, rel_error_lists
        

'''
def DiT_wsvd(model, keeps_only=True):
    for idx_block, block in enumerate(model.blocks):
        attn, mlp = block.attn, block.mlp
        # Attn
        q_w, k_w, v_w = attn.qkv.weight.data.chunk(3, dim=0)
        q_u, q_s, q_vh = wsvd(q_w, getattr(attn, "q_wtv", None))     
        k_u, k_s, k_vh = wsvd(k_w, getattr(attn, "k_wtv", None))
        v_u, v_s, v_vh = wsvd(v_w, getattr(attn, "v_wtv", None))
        qkv_vh = torch.cat([q_vh, k_vh, v_vh], dim=0)
        proj_u, proj_s, proj_vh = wsvd(attn.proj.weight.data, getattr(attn, "proj_wtv", None))
        # Mlp
        fc1_u, fc1_s, fc1_vh = wsvd(mlp.fc1.weight.data, getattr(mlp, "fc1_wtv", None))
        fc2_u, fc2_s, fc2_vh = wsvd(mlp.fc2.weight.data, getattr(mlp, "fc2_wtv", None))
        # Set singular value
        setattr(attn, "q_s", q_s)
        setattr(attn, "k_s", k_s)
        setattr(attn, "v_s", v_s)
        setattr(attn, "proj_s", proj_s)
        setattr(mlp, "fc1_s", fc1_s)
        setattr(mlp, "fc2_s", fc2_s)
        if(keeps_only == False):
            # Set other
            setattr(attn, "q_u", q_u)
            setattr(attn, "k_u", k_u)
            setattr(attn, "v_u", v_u)
            setattr(attn, "qkv_vh", qkv_vh)
            setattr(attn, "proj_u", proj_u)
            setattr(attn, "proj_vh", proj_vh)
            setattr(mlp, "fc1_u", fc1_u)
            setattr(mlp, "fc1_vh", fc1_vh)
            setattr(mlp, "fc2_u", fc1_u)
            setattr(mlp, "fc2_vh", fc1_vh)
        print(f"Block{idx_block:02d} wsvd is finished.")
        
def DiT_get_singular(model, format="dict"):
    # return:
    # "dict":   q_s/k_s/v_s/proj_s/fc1_s/fc2_s: tensor, (num_blocks, hidden_size)
    # "tensor": tensor, (num_blocks, 6, hidden_size)
    keys_attn = ["q_s", "k_s", "v_s", "proj_s"]
    keys_mlp = ["fc1_s", "fc2_s"]
    s = {key : None for key in (keys_attn + keys_mlp)}
    for block in model.blocks:
        attn, mlp = block.attn, block.mlp
        # MHA
        for key in keys_attn:
            if(s[key] == None):
                s[key] = getattr(attn, key).unsqueeze(0)
            else:
                s[key] = torch.cat([s[key], getattr(attn, key).unsqueeze(0)], dim=0)
        # FFN
        for key in keys_mlp:
            if(s[key] == None):
                s[key] = getattr(mlp, key).unsqueeze(0)
            else:
                s[key] = torch.cat([s[key], getattr(mlp, key).unsqueeze(0)], dim=0)
    # dict2tensor
    if(format == "tensor"):
        q_s, k_s, v_s, proj_s = s["q_s"], s["k_s"], s["v_s"], s["proj_s"]
        fc1_s, fc2_s = s["fc1_s"], s["fc2_s"]
        q_s = q_s.unsqueeze(1)
        k_s = k_s.unsqueeze(1)
        v_s = v_s.unsqueeze(1)
        proj_s = proj_s.unsqueeze(1)
        fc1_s = fc1_s.unsqueeze(1)
        fc2_s = fc2_s.unsqueeze(1)
        s_list = [q_s, k_s, v_s, proj_s, fc1_s, fc2_s]
        s = torch.cat(s_list, dim=1)
    return s

        
# For rank search
def DiT_rank_decide(model, min_pca_accu_energy):
    for block in model.blocks:
        pass

def DiT_rank_search(model, max_num_param_lowrank, rank_init=1152, grain=2, report_every_search=100):
    # rank_init % grain == 0
    # Size
    hidden_size = model.blocks[0].mlp.fc1.in_features
    mlp_ratio = model.blocks[0].mlp.fc1.out_features // hidden_size
    num_blocks = len(model.blocks)
    # Calculate score = relative_singular_value / scale, remove the rank with smallest score first 
    # Load singular value
    sigma = DiT_get_singular(model, "tensor")                             #(num_blocks, 6, hidden_size)
    # Calculate scale = (M+N)/C
    scale_attn, scale_mlp = 2, (1+mlp_ratio)
    scale = [scale_attn] * 4 + [scale_mlp] * 2
    scale = torch.tensor(scale, device = sigma.device)                    #(6)
    # Initialize rank
    sigma = sigma[:, :, :rank_init]                                       #(num_blocks, 6, rank_init)
    rank = torch.ones([num_blocks, 6], device=sigma.device) * rank_init   #(num_blocks, 6)
    rank = rank.long()
    # Initialize score
    score = sigma / sigma.sum(dim=-1, keepdim=True)                       #(num_blocks, 6, rank_init)
    score = score / scale.unsqueeze(0).unsqueeze(-1)                      #(num_blocks, 6, rank_init)
    score = score.reshape(num_blocks, 6, rank_init // grain, grain)       #(num_blocks, 6, rank_init // grain, grain)
    score = score.mean(dim=-1, keepdim=True).expand(-1, -1, -1, grain)    #(num_blocks, 6, rank_init // grain, grain)
    score = score.reshape(num_blocks, 6, rank_init)                       #(num_blocks, 6, rank_init), every {grain} elements are the same
    #print(score)
    # Calculate number of parameter
    num_param_lowrank = scale.unsqueeze(0) * rank                         #(num_blocks, 6)
    num_param_lowrank = num_param_lowrank.sum().item() * hidden_size / 1e6  #unit: M
    # Search loop
    cnt_search = 0
    while(num_param_lowrank > max_num_param_lowrank):
        idx_rank_min_score = rank - 1                                                               #(num_blocks, 6)
        min_score_per_layer = torch.gather(score, dim=-1, index=idx_rank_min_score.unsqueeze(-1))   #(num_blocks, 6, 1)
        min_score_per_layer = min_score_per_layer.squeeze(-1)                                       #(num_blocks, 6)
        idx_block, idx_layer = torch.where(min_score_per_layer == min_score_per_layer.min())  
        idx_block, idx_layer = idx_block[0], idx_layer[0]
        # Update rank & score
        #print(idx_block, idx_layer)
        rank_upd = rank[idx_block, idx_layer].item() - grain
        score_upd = score[idx_block, idx_layer, :rank_upd] / score[idx_block, idx_layer, :rank_upd].sum()
        rank[idx_block, idx_layer] = rank_upd
        score[idx_block, idx_layer, :rank_upd] = score_upd        
        # Update number of parameter
        num_param_lowrank = scale.unsqueeze(0) * rank
        num_param_lowrank = num_param_lowrank.sum().item() * hidden_size / 1e6
        # Report
        if((cnt_search + 1) % report_every_search == 0):
            print(f"Search{cnt_search:08d} is finished. Number of lowrank weight: {num_param_lowrank:.4f}M")
        # Update counter
        cnt_search += 1
    # To list
    rank_q = rank[:, 0].tolist()
    rank_k = rank[:, 1].tolist()
    rank_v = rank[:, 2].tolist()
    rank_proj = rank[:, 3].tolist()
    rank_fc1 = rank[:, 4].tolist()
    rank_fc2 = rank[:, 5].tolist()
    print(rank_q)
    print(rank_k)
    print(rank_v)
    print(rank_proj)
    print(rank_fc1)
    print(rank_fc2)
    return rank_q, rank_k, rank_v, rank_proj, rank_fc1, rank_fc2
'''