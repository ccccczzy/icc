import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from models.models import DiTBlock
from models.models import DiT
from models.models import modulate
from models.cache import print_cache_life, cache

# Global & Static gate
# Global: skip or not skip all layers
# Static: pre-defined sparse scheduler --> remain a static graph
def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
#################################################################################
#                     Shared Sparse Modules & Functions                         #
#################################################################################

def linear_decomp(x, wa, wb, bias=None):
    # w = wa * wb
    # y = wx + bias
    x = F.linear(x, wb)
    x = F.linear(x, wa, bias)
    return x

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
        # Intermediate Cache
        self.cache = cache()

    def macs_ref_item(self, num_tokens):
        # return:    int
        # 'item' means it returns a value insteal of a 1-element tensor 
        macs_fc1 = num_tokens * self.fc1.in_features * self.fc1.out_features
        macs_fc2 = num_tokens * self.fc2.in_features * self.fc2.out_features
        macs = macs_fc1 + macs_fc2
        return macs

    def macs_inc_item(self, num_tokens):
        rank_fc1 = self.fc1_wa.shape[1]
        rank_fc2 = self.fc2_wa.shape[1]
        macs_fc1 = num_tokens * (self.fc1.in_features + self.fc1.out_features) * rank_fc1
        macs_fc2 = num_tokens * (self.fc2.in_features + self.fc2.out_features) * rank_fc2
        macs = macs_fc1 + macs_fc2
        return macs
    
    def forward_upd(self, x):
        self.cache.content["fc1_x"] = x
        x = self.fc1(x)
        self.cache.content["fc1_y"] = x
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        self.cache.content["fc2_x"] = x
        x = self.fc2(x)
        self.cache.content["fc2_y"] = x
        x = self.drop2(x)
        return x
    
    def forward_reuse(self, x):
        #ratio = (mean_flat((x - self.cache.content["fc1_x"])**2) / mean_flat(self.cache.content["fc1_x"]**2)).mean()
        #sim1 = F.cosine_similarity(x, self.cache.content["fc1_x"])
        #sim2 = F.cosine_similarity(x - self.cache.content["fc1_x"], self.cache.content["fc1_x"])
        #print(ratio)
        x = self.cache.content["fc2_y"]
        x = self.drop2(x)
        return x
    
    def forward_inc(self, x):
        # Forward with low-rank approximated weights for delta_x
        # Low-rank fc1
        cache_x, cache_y = self.cache.content["fc1_x"], self.cache.content["fc1_y"]                                 
        x = x - cache_x                                                                #(batch_size, num_tokens, hidden_size)
        x = linear_decomp(x, self.fc1_wa, self.fc1_wb)                                 #(batch_size, num_tokens, mlp_ratio*hidden_size)
        x = x + cache_y                                                                #(batch_size, num_tokens, mlp_ratio*hidden_size)
        # Other
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        # Low-rank fc2
        cache_x, cache_y = self.cache.content["fc2_x"], self.cache.content["fc2_y"]                                   
        x = x - cache_x                                                                #(batch_size, num_tokens, mlp_ratio*hidden_size)
        x = linear_decomp(x, self.fc2_wa, self.fc2_wb)                                 #(batch_size, num_tokens, hidden_size)
        x = x + cache_y                                                                #(batch_size, num_tokens, hidden_size)
        # Other
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
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__(dim, num_heads, qkv_bias, qk_norm, attn_drop, proj_drop, norm_layer)
        # Intermediate Cache
        self.cache = cache()
        
    def macs_ref_item(self, num_tokens):
        macs_qkv = num_tokens * self.qkv.in_features * self.qkv.out_features
        macs_proj = num_tokens * self.proj.in_features * self.proj.out_features
        macs_mm = 2 * num_tokens * num_tokens * self.proj.in_features
        macs = macs_qkv + macs_proj + macs_mm
        return macs

    def macs_inc_item(self, num_tokens):
        rank_q = self.q_wa.shape[1]
        rank_k = self.k_wa.shape[1]
        rank_v = self.v_wa.shape[1]
        rank_proj = self.proj_wa.shape[1]
        macs_qkv = num_tokens * (self.qkv.in_features + self.qkv.out_features // 3) * (rank_q + rank_k + rank_v)
        macs_proj = num_tokens * (self.proj.in_features + self.proj.out_features) * rank_proj
        macs_mm = 2 * num_tokens * num_tokens * self.proj.in_features
        macs = macs_qkv + macs_proj + macs_mm
        return macs
    
    def forward_upd(self, x):
        B, N, C = x.shape
        self.cache.content["qkv_x"] = x
        qkv = self.qkv(x)
        self.cache.content["qkv_y"] = qkv
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
        self.cache.content["proj_x"] = x
        x = self.proj(x)
        self.cache.content["proj_y"] = x
        x = self.proj_drop(x)
        return x
    
    def forward_reuse(self, x):
        x = self.cache.content["proj_y"]
        x = self.proj_drop(x)
        return x
    
    def forward_inc(self, x):
        B, N, C = x.shape
        # Low-rank QKV projection
        cache_x, cache_y = self.cache.content["qkv_x"], self.cache.content["qkv_y"]
        x = x - cache_x
        q = linear_decomp(x, self.q_wa, self.q_wb)                                         #(batch_size, num_tokens, hidden_size)
        k = linear_decomp(x, self.k_wa, self.k_wb)                                         #(batch_size, num_tokens, hidden_size)
        v = linear_decomp(x, self.v_wa, self.v_wb)                                         #(batch_size, num_tokens, hidden_size)
        qkv = torch.cat([q, k, v], dim=-1).contiguous()                                    #(batch_size, num_tokens, 3*hidden_size)
        qkv = qkv + cache_y                                                                #(batch_size, num_tokens, 3*hidden_size)
        '''
        rank_q = self.q_wa.shape[1]
        rank_k = self.k_wa.shape[1]
        rank_v = self.v_wa.shape[1]
        cache_x, cache_y = self.cache.content["qkv_x"], self.cache.content["qkv_y"]
        x = x - cache_x
        q, k, v = F.linear(x, self.qkv_wb).split([rank_q, rank_k, rank_v], dim=-1)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        q, k, v = F.linear(q, self.q_wa), F.linear(k, self.k_wa), F.linear(v, self.v_wa)   #(batch_size, num_tokens, hidden_size)
        qkv = torch.cat([q, k, v], dim=-1).contiguous()                                    #(batch_size, num_tokens, 3*hidden_size)
        qkv = qkv + cache_y                                                                #(batch_size, num_tokens, 3*hidden_size)
        '''
        # Other
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
        # Low-rank output projection
        cache_x, cache_y = self.cache.content["proj_x"], self.cache.content["proj_y"]
        x = x - cache_x
        x = linear_decomp(x, self.proj_wa, self.proj_wb)
        x = x + cache_y
        # Other
        x = self.proj_drop(x)
        return x
    

#################################################################################
#                                 Sparse DiT                                    #
#################################################################################

class SpDiTBlock(DiTBlock):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__(hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs)
        #Sparse Attn
        self.attn = SpAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs) 
        #Sparse Mlp 
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = SpMlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0,)
        self.hidden_size = hidden_size

    def cache_clr(self):
        self.attn.cache.clr()
        self.mlp.cache.clr() 
        
    def cache_detach(self):
        self.attn.cache.detach()
        self.mlp.cache.detach()
        
    def macs_ref_item(self, num_tokens):
        macs_attn = self.attn.macs_ref_item(num_tokens)
        macs_mlp = self.mlp.macs_ref_item(num_tokens)
        macs_adaLN_modulation = self.adaLN_modulation[-1].in_features * self.adaLN_modulation[-1].out_features
        macs = macs_attn + macs_mlp + macs_adaLN_modulation
        return macs
    
    def macs_reuse_item(self):
        macs = self.adaLN_modulation[-1].in_features * self.adaLN_modulation[-1].out_features
        return macs
    
    def macs_inc_item(self, num_tokens):
        macs_attn = self.attn.macs_inc_item(num_tokens)
        macs_mlp = self.mlp.macs_inc_item(num_tokens)
        macs_adaLN_modulation = self.adaLN_modulation[-1].in_features * self.adaLN_modulation[-1].out_features
        macs = macs_attn + macs_mlp + macs_adaLN_modulation
        return macs
    
    def forward_upd(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn.forward_upd(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp.forward_upd(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
    
    def forward_reuse(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn.forward_reuse(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp.forward_reuse(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
    
    def forward_inc(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn.forward_inc(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp.forward_inc(modulate(self.norm2(x), shift_mlp, scale_mlp))
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
        # Additional args
        use_inc = False
    ):
        super().__init__(input_size, patch_size, in_channels, hidden_size, depth, num_heads, mlp_ratio, class_dropout_prob, num_classes, learn_sigma)
        #Sparse DiT Block
        self.blocks = nn.ModuleList([
            SpDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.initialize_weights()
        self.num_tokens = (input_size // patch_size)**2
        self.use_inc = use_inc

    def sched_attach(self, sched):
        self.sched = sched   
 
    def cache_clr(self):
        for block in self.blocks:
            block.cache_clr()
        self.sched.clr()
            
    def cache_detach(self):
        for block in self.blocks:
            block.cache_detach()
            
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
        macs_blocks = len(self.blocks) * self.blocks[0].macs_ref_item(self.num_tokens)
        # Others
        macs_prepost = self.macs_prepost_item()
        # Summation
        macs = macs_blocks + macs_prepost
        return macs/1e9     # Unit: GMACs
    
    def macs_reuse_item(self):
        # Blocks
        macs_blocks = len(self.blocks) * self.blocks[0].macs_reuse_item()
        # Others
        macs_prepost = self.macs_prepost_item()
        # Summation
        macs = macs_blocks + macs_prepost
        return macs/1e9    # Unit: GMACs
    
    def macs_inc_item(self):
        # Blocks
        macs_blocks = 0
        for block in self.blocks:
            macs_blocks += block.macs_inc_item(self.num_tokens)
        # Others
        macs_prepost = self.macs_prepost_item()
        # Summation
        macs = macs_blocks + macs_prepost
        return macs/1e9    # Unit: GMACs
    
    def macs_sp_item(self):
        macs = 0
        num_steps = self.sched.num_steps
        for idx in range(num_steps):
            gate = self.sched.gate_list[idx]
            if(gate == 1):
                macs += self.macs_ref_item()
            elif(self.use_inc):
                macs += self.macs_inc_item()
            else:
                macs += self.macs_reuse_item()
        macs = macs / num_steps
        return macs
                
    def forward_sp(self, x, t, y):
        # Sparse sched
        gate = self.sched.step()
        if(gate == 1):
            block_forward_name = "forward_upd"
        elif(self.use_inc):
            block_forward_name = "forward_inc"
        else:
            block_forward_name = "forward_reuse"
        # Pre-process
        x = self.x_embedder(x) + self.pos_embed           # (batch_size, num_tokens, hidden_size) 
        t = self.t_embedder(t)                            # (batch_size, hidden_size)
        y = self.y_embedder(y, self.training)             # (batch_size, hidden_size)
        c = t + y                                         # (batch_size, hidden_size)
        # DiT Blocks
        for block in self.blocks:
            x = getattr(block, block_forward_name)(x, c)  # (batch_size, num_tokens, hidden_size)
        # Post-process
        x = self.final_layer(x, c)                        # (batch_size, num_tokens, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                            # (batch_size, out_channels, H, W) 
        return x
    
    def forward_with_cfg_sp(self, x, t, y, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward_sp(combined, t, y)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

####################################################################################
#                                 SpDiT Wrapper                                    #
####################################################################################
# Not required

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