import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from models.models import DiTBlock
from models.models import DiT
from models.models import modulate

from utils.utils_stat import stat_dist


def stat_upd(layer, x, stat_flag, name, norm="min_max"):
    if(stat_flag == "cache"):
        # Update cache for delta
        setattr(layer, name+"_cache", x)
    else:
        # Update counter
        cnt = getattr(layer, name+"_cnt", 0)
        setattr(layer, name+"_cnt", cnt+1)
        
        # Delta or not
        if(stat_flag == "delta_mag"):
            delta = x - getattr(layer, name+"_cache")
            tmp_mag = delta.abs().mean(dim=1)
        elif(stat_flag == "mag"):
            tmp_mag = x.abs().mean(dim=1)                                    #(batch_size, C), averafe across tokens 
            
        # Norm inside each batch to eliminate difference between different steps
        if(norm == None):
            tmp_mag = tmp_mag.mean(dim=0)
        elif(norm == "min_max"):
            eps = 1e-6
            max_val = tmp_mag.max(dim=-1, keepdim=True)[0]                   #(batch_size, 1)
            min_val = tmp_mag.min(dim=-1, keepdim=True)[0]                   #(batch_size, 1)
            tmp_mag = (tmp_mag - min_val + eps) / (max_val - min_val + eps)  #(batch_size, C)
            tmp_mag = tmp_mag.mean(dim=0)                                    #(C)
        else:
            raise NotImplementError(norm)
            
        # Update mag
        mag = getattr(layer, name+"_mag", 0)
        mag = mag + tmp_mag
        setattr(layer, name+"_mag", mag)
            

#################################################################################
#                               Sparse Layers                                   #
#################################################################################

class SpMlp(Mlp):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False
    ):
        super().__init__(in_features, hidden_features, out_features, act_layer, norm_layer, bias, drop, use_conv)
        self.dist_act_x = stat_dist(-10, 10, 0.1)
    
    def macs_ref_item(self, num_tokens):
        # return:    int
        # 'item' means it returns a value insteal of a 1-element tensor 
        macs_fc1 = num_tokens * self.fc1.in_features * self.fc1.out_features
        macs_fc2 = num_tokens * self.fc2.in_features * self.fc2.out_features
        macs = macs_fc1 + macs_fc2
        return macs

    def macs_decomp_item(self, num_tokens):
        rank_fc1 = self.fc1_wa.shape[1]
        rank_fc2 = self.fc2_wa.shape[1]
        macs_fc1 = num_tokens * (self.fc1.in_features + self.fc1.out_features) * rank_fc1
        macs_fc2 = num_tokens * (self.fc2.in_features + self.fc2.out_features) * rank_fc2
        macs = macs_fc1 + macs_fc2
        return macs
    
    def forward_stat(self, x, stat_flag):
        stat_upd(self, x, stat_flag, "fc1_x")
        x = self.fc1(x)
        stat_upd(self, x, stat_flag, "fc1_y")
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        stat_upd(self, x, stat_flag, "fc2_x")
        x = self.fc2(x)
        stat_upd(self, x, stat_flag, "fc2_y")
        x = self.drop2(x)
        return x
        
class SpAttention(Attention):
    #fused_attn: Final[bool]
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm
    ) -> None:
        super().__init__(dim, num_heads, qkv_bias, qk_norm, attn_drop, proj_drop, norm_layer)
    
    def macs_ref_item(self, num_tokens):
        macs_qkv = num_tokens * self.qkv.in_features * self.qkv.out_features
        macs_proj = num_tokens * self.proj.in_features * self.proj.out_features
        macs_mm = 2 * num_tokens * num_tokens * self.proj.in_features
        macs = macs_qkv + macs_proj + macs_mm
        return macs

    def macs_decomp_item(self, num_tokens):
        rank_q = self.q_wa.shape[1]
        rank_k = self.k_wa.shape[1]
        rank_v = self.v_wa.shape[1]
        rank_proj = self.proj_wa.shape[1]
        macs_qkv = num_tokens * (self.qkv.in_features + self.qkv.out_features // 3) * (rank_q + rank_k + rank_v)
        macs_proj = num_tokens * (self.proj.in_features + self.proj.out_features) * rank_proj
        macs_mm = 2 * num_tokens * num_tokens * self.proj.in_features
        macs = macs_qkv + macs_proj + macs_mm
        return macs
    
    def forward_stat(self, x, stat_flag):
        B, N, C = x.shape
        stat_upd(self, x, stat_flag, "qkv_x")
        qkv = self.qkv(x)
        stat_upd(self, qkv, stat_flag, "qkv_y")
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        stat_upd(self, x, stat_flag, "proj_x")
        x = self.proj(x)
        stat_upd(self, x, stat_flag, "proj_y")
        x = self.proj_drop(x)
        return x

#################################################################################
#                                 Sparse DiT                                    #
#################################################################################

class SpDiTBlock(DiTBlock):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, 
                 lowrank_use_attn=True, lowrank_use_mlp=True,
                 **block_kwargs):
        super().__init__(hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs)
        #Sparse Attn
        self.attn = SpAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs) 
        #Sparse Mlp 
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = SpMlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.hidden_size = hidden_size
    
    def macs_ref_item(self, num_tokens):
        macs_attn = self.attn.macs_ref_item(num_tokens)
        macs_mlp = self.mlp.macs_ref_item(num_tokens)
        macs_adaLN_modulation = self.adaLN_modulation[-1].in_features * self.adaLN_modulation[-1].out_features
        macs = macs_attn + macs_mlp + macs_adaLN_modulation
        return macs
    
    def macs_decomp_item(self, num_tokens):
        macs_attn = self.attn.macs_decomp_item(num_tokens)
        macs_mlp = self.mlp.macs_decomp_item(num_tokens)
        macs_adaLN_modulation = self.adaLN_modulation[-1].in_features * self.adaLN_modulation[-1].out_features
        macs = macs_attn + macs_mlp + macs_adaLN_modulation
        return macs
    
    def forward_stat(self, x, c, stat_flag):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn.forward_stat(modulate(self.norm1(x), shift_msa, scale_msa), stat_flag)
        x = x + gate_mlp.unsqueeze(1) * self.mlp.forward_stat(modulate(self.norm2(x), shift_mlp, scale_mlp), stat_flag)
        return x

class SpDiT(DiT):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__(input_size, patch_size, in_channels, hidden_size, depth, num_heads, mlp_ratio, class_dropout_prob, num_classes, learn_sigma)
        #Sparse DiT Block
        self.blocks = nn.ModuleList([
            SpDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.initialize_weights()
        self.num_tokens = (input_size // patch_size)**2
    
    def macs_prepost_item(self):
        num_tokens = self.num_tokens
        # x_embedder
        macs_x_embedder = num_tokens * self.x_embedder.proj.in_channels * self.x_embedder.proj.out_channels * self.x_embedder.proj.kernel_size[0] * self.x_embedder.proj.kernel_size[1]
        # t_embedder
        macs_t_embedder = 0
        macs_t_embedder += self.t_embedder.mlp[0].in_features * self.t_embedder.mlp[0].out_features
        macs_t_embedder += self.t_embedder.mlp[2].in_features * self.t_embedder.mlp[2].out_features
        # final_layer
        macs_final_layer = 0
        macs_final_layer += self.final_layer.adaLN_modulation[-1].in_features * self.final_layer.adaLN_modulation[-1].out_features
        macs_final_layer += num_tokens * self.final_layer.linear.in_features * self.final_layer.linear.out_features
        # Summation
        macs = macs_x_embedder + macs_t_embedder + macs_final_layer
        return macs
    
    def macs_ref_item(self):
        # Blocks
        macs_blocks = 0
        for block in self.blocks:
            macs_blocks += block.macs_ref_item(self.num_tokens)
        # Others
        macs_prepost = self.macs_prepost_item()
        # Summation
        macs = macs_blocks + macs_prepost
        return macs/1e9     # Unit: GMACs
    
    def macs_decomp_item(self):
        # Blocks
        macs_blocks = 0
        for block in self.blocks:
            macs_blocks += block.macs_decomp_item(self.num_tokens)
        # Others
        macs_prepost = self.macs_prepost_item()
        # Summation
        macs = macs_blocks + macs_prepost
        return macs/1e9    # Unit: GMACs
    
    def forward_stat(self, x, t, y, stat_flag):
        # Pre-process
        x = self.x_embedder(x) + self.pos_embed     #(batch_size, num_tokens, hidden_size) 
        t = self.t_embedder(t)                      #(batch_size, hidden_size)
        y = self.y_embedder(y, self.training)       #(batch_size, hidden_size)
        c = t + y                                   #(batch_size, hidden_size)
        # DiT Blocks
        for block in self.blocks:
            x = block.forward_stat(x, c, stat_flag) #(batch_size, num_tokens, hidden_size)
        # Post-process
        x = self.final_layer(x, c)                  #(batch_size, num_tokens, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                      #(batch_size, out_channels, H, W)
        return x
    
    def forward(self, x, t, y, stat_flag=None):
        if(stat_flag == None):
            return self.super.forward(x, t, y)
        else:
            return self.forward_stat(x, t, y, stat_flag)
            
####################################################################################
#                                   SpDiT Configs                                  #
####################################################################################
def DiT_XL_2(**kwargs):
    return SpDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return SpDiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return SpDiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return SpDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return SpDiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return SpDiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return SpDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return SpDiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return SpDiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return SpDiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return SpDiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return SpDiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

def DiT_SS_2(**kwargs):
    #single block for test
    return SpDiT(depth=4, hidden_size=1152, patch_size=2, num_heads=16, **kwargs) 

DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}