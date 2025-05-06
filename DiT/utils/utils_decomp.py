import torch
import math
import torch.nn as nn

'''
def lowrank_scale_bias(x, cache_x, use_scale=False, use_bias=False):
    #x, cache_x:          tensor, (batch_size, num_tokens, hidden_size)
    #use_scale, use_bias: bool
    #return: scale/bias:  tensor, (batch_size, num_tokens, 1) or float, default by 1/0
    if(use_scale == False and use_bias == False):
        scale, bias = 1, 0
    elif(use_scale == True and use_bias == True):
        # x = ay + bz, where y is cache_x, z is a unit const vector
        size_vec = x.shape[-1]
        const = 1 / math.sqrt(size_vec)
        #(batch_size, num_tokens, 1)
        prod_xy = (x * cache_x).sum(dim=-1, keepdim=True)
        prod_zz = 1
        prod_yz = cache_x.sum(dim=-1, keepdim=True) * const
        prod_xz = x.sum(dim=-1, keepdim=True) * const
        prod_yy = (cache_x ** 2).sum(dim=-1, keepdim=True)
        denominator = prod_yy * prod_zz - prod_yz ** 2
        scale = (prod_xy * prod_zz - prod_yz * prod_xz) / denominator
        bias = (prod_xz * prod_yy - prod_yz * prod_xy) / denominator * const
    elif(use_scale == True):
        scale = (x*cache_x).sum(dim=-1, keepdim=True) / (cache_x**2).sum(dim=-1, keepdim=True)
        bias = 0
    return scale, bias
'''

def fim_calc(fim, weight, idx_iter):
    # Calculate Fisher Information Matrix
    # idx_iter starts from 0
    fim = (fim*idx_iter + (weight.grad)**2) / (idx_iter+1)
    return fim

def decomp_check(x, a, b):
    # Check the absolute & relative error of reconstructed matrix
    reconst_x = a @ b
    abs_error = (x - reconst_x).flatten().abs().mean().item()
    rel_error = abs_error / (x.flatten().abs().mean().item())
    return abs_error, rel_error

def wsvd(x, lwtv=None, rwtv=None):
    # Weighted Singular Value Decomposition
    # x:          tensor, (M, N)
    # lwtv:       tensor, (M)
    # rwtv:       tensor, (N)
    # return: u:  tensor, (M, M)
    #         s: 
    #         vh: tensor, (N, N)
    if(lwtv != None):
        lwtv += 1e-6
    if(rwtv != None):
        rwtv += 1e-6
    x = x.to(torch.float32)
    x = (torch.diag(lwtv) @ x) if (lwtv != None) else x
    x = (x @ torch.diag(rwtv)) if (rwtv != None) else x
    u, s, vh = torch.linalg.svd(x)
    u = (torch.diag(lwtv).inverse() @ u) if (lwtv != None) else u
    vh = (vh @ torch.diag(rwtv).inverse()) if (rwtv != None) else vh
    return u, s, vh

def wsvd_merge_lowrank(x, rank, lwtv=None, rwtv=None, merge_mode="u", idx_group=0):
    # Weighted Singular Value Decomposition & merge
    # Approximate x with a @ b
    u, s, vh = wsvd(x, lwtv, rwtv)
    s_sel = s[idx_group*rank : (idx_group+1)*rank]
    u_sel = u[:, idx_group*rank : (idx_group+1)*rank]
    vh_sel = vh[idx_group*rank : (idx_group+1)*rank, :]
    if(merge_mode == "u"):
        a = u_sel @ torch.diag(s_sel)
        b = vh_sel
    elif(merge_mode == "v"):
        a = u_sel
        b = torch.diag(s_sel) @ vh_sel
    elif(merge_mode == "uv"):
        a = u_sel @ torch.diag(s_sel.sqrt())
        b = torch.diag(s_sel.sqrt()) @ vh_sel
    a = a.contiguous()
    b = b.contiguous()
    pca_energy = s_sel.sum().item() / s.sum().item()
    abs_error, rel_error = decomp_check(x, a, b)
    return a, b, pca_energy, abs_error, rel_error

def sparse_metric(x):
    #reference: https://spaces.ac.cn/archives/9595
    # range: [1/sqrt(n), 1]
    n = x.numel()
    l1 = x.norm(p=1)
    l2 = x.norm(p=2)
    sp = (1 / math.sqrt(n)) * l1 / l2
    return sp

def shift2ratio(x, max_ratio=None):
    if(max_ratio == None):
        return x
    else:
        return x.clamp(x.max().item() / max_ratio)


'''
def wsvd_merge_lowrank_thr_energy(x, thr, lwtv=None, rwtv=None, merge_mode="u"):
    u, s, vh = wsvd(x, lwtv, rwtv)
    s_cumsum = s.cumsum() 
    energy = s_cumsum / s.sum().item()
    indices = torch.where(energy >= thr)[0]
    rank = indices[0].item() + 1
    
    s_sel = s[:rank]
    u = u[:, :rank]
    vh = vh[:ranl, :]
    us = u @ torch.diag(s_sel)
    pca_energy = s_sel.sum().item() / s.sum().item()
    abs_error, rel_error = decomp_check(x, us, vh)
    return us, vh, pca_energy, abs_error, rel_error
'''



def linear_decomp(x, wa, wb, bias=None):
    # w = wa * wb
    # y = wx + bias
    x = F.linear(x, wb)
    x = F.linear(x, wa, bias)
    return x

class decomp_Linear(nn.Linear):
    def __init__(self):
        pass
    
    
