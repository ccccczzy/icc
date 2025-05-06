import torch
import torch.nn as nn
import math


# Gate:  do or donot compute 
#################################################################################
#                          Gate of MHA/FFN Layer                                #
#################################################################################
class gate_NULL(nn.Module):
    # Do nothing
    def __init__(self):
        super().__init__()
    
    def forward(self, x, cache, i_name, o_name):
        gate = torch.ones(x.shape[0], device=x.device) 
        return gate

class gate_RAND(nn.Module):
    # Randomly decide compute or not
    def __init__(self, dense_prob=0.5):
        super().__init__()
        self.dense_prob = dense_prob
        
    def forward(self, x, cache, i_name, o_name):
        batch_size = x.shape[0]
        dense_ids = torch.rand(x.shape[0], device=x.device) < self.dense_prob
        gate = torch.zeros(x.shape[0], device=x.device)
        gate = torch.where(dense_ids, 1, gate)
        return gate

class gate_ACCU(nn.Module):
    # Decide compute or not based on accumulated input L1/L2 error
    def __init__(self, p=1, thr=0.2):
        super().__init__()
        self.thr = thr
        self.p = p
        
    def forward(self, x, cache, i_name, o_name):
        # Cache read
        cache_x = cache[i_name]
        # Flatten
        delta = (x - cache_x).flatten(start_dim=1)          #(batch_size, XXX)
        x = x.flatten(start_dim=1)                          #(batch_size, XXX)
        # Compute error                                     #(batch_size)
        error = torch.norm(delta, dim=1, p=self.p) / torch.norm(x, dim=1, p=self.p)   #Relative L1/L2
        # Accumulate error
        accu_error = cache.get("accu_error", None)
        accu_error = error if (accu_error == None) else (error + accu_error)
        # Thr
        logits = accu_error - self.thr
        gate = (logits > 0) * 1. 
        # Update accumulated error
        cache["accu_error"] = (1-gate) * accu_error + gate * torch.zeros_like(accu_error)
        return gate

class gate_PERIOD(nn.Module):
    # Decide compute or not based on a fixed period
    def __init__(self, period=2):
        super().__init__()
        self.period = period   #int
        
    def forward(self, x, cache, i_name, o_name):
        # Accumulate cycle
        accu_num_cyc = cache.get("accu_num_cyc", 0)
        accu_num_cyc = 0 if (accu_num_cyc is None) else accu_num_cyc
        accu_num_cyc = accu_num_cyc + 1
        # Gate & update accumulated number of cycles
        if(accu_num_cyc % self.period == 0):
            gate = torch.ones(x.shape[0], device=x.device) 
            cache["accu_num_cyc"] = 0
        else:
            gate = torch.zeros(x.shape[0], device=x.device)
            cache["accu_num_cyc"] = accu_num_cyc
        return gate
    
class gate_PERIOD_cfg(nn.Module):
    def __init__(self, period=2):
        super().__init__()
        self.period = period
    
    def forward(self, x, cache, i_name, o_name):
        pass
        
#################################################################################
#                               Gate of DiT                                     #
#################################################################################
class gate_DiTBlock(nn.Module):
    # The collection of gates for a single DiT block; cannot be used alone
    def __init__(self, type_attn, kwargs_attn, type_mlp, kwargs_mlp):
        super().__init__()
        name_gate_attn = "gate_" + type_attn
        name_gate_mlp = "gate_" + type_mlp
        self.gate_attn = globals()[name_gate_attn](**kwargs_attn)
        self.gate_mlp = globals()[name_gate_mlp](**kwargs_mlp)

class gate_DiT(nn.Module):
    # The collection of gates for a DiT model; cannot be used alone
    def __init__(self, num_blocks, hidden_size, type_attn, kwargs_attn, type_mlp, kwargs_mlp):
        super().__init__()
        self.gates_blocks = nn.ModuleList()
        for idx_block in range(num_blocks):
            self.gates_blocks.append(gate_DiTBlock(type_attn, kwargs_attn, type_mlp, kwargs_mlp))

#################################################################################
#                               Gate Config                                     #
#################################################################################
# Only gate for DiT-XL/2 is implemented
def gate_DiT_XL_2(**kwargs):
    return gate_DiT(num_blocks=28, hidden_size=1152, **kwargs)

gate_DiT_models = {'DiT-XL/2': gate_DiT_XL_2}