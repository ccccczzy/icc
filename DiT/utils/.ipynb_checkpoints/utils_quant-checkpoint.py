import torch
import torch.nn as nn
####################################################
#                Fake Data Format                  #
####################################################
# Fake precision wrapped in torch.float32
# Use when some data format is not supported in current device
class fake_float():
    # Fake floating-point 
    def __init__(self, wd_exp, wd_mant, func_round=torch.round):
        self.wd_mant = wd_mant
        self.wd_exp = wd_exp
        self.wd = 1 + wd_mant + wd_exp
        self.func_round = func_round
    
    def check(self, x):
        '''
        bit field:   S    E   M; integer values 
        value field: sign exp mant; refer to torch;
        bias = 2^(wd_exp-1)-1, scale = 2^(wd_mant)
        x = (-1)^S * 2^(E - bias) * (1 + M/scale)
        x =          2^exp        * mant, signed mant belongs to (-1, 1)
        mant = (-1)^S * (1 + M/scale)/2 --> M = (2*abs(mant) - 1) * scale
        exp = E - bias + 1              --> E = exp + bias - 1
        '''
        # Calculate scale & bias
        scale = 2**self.wd_mant
        bias = 2**(self.wd_exp-1) - 1
        # Value decomp
        mant, exp = torch.frexp(x)
        # Tuncate M
        M = (2*mant.abs() - 1) * scale
        M = self.func_round(M)
        mant = x.sign() * (1 + M / scale) / 2
        # Saturation E
        min_exp = 1 - bias + 1
        max_exp = (2**self.wd_exp - 1 - 1) - bias + 1        
        mant = torch.where(exp > max_exp, 1, mant)
        mant = torch.where(exp < min_exp, 0, mant)
        exp = torch.where(exp > max_exp, max_exp, exp)
        exp = torch.where(exp < min_exp, min_exp, exp)
        # Re-combine
        x = (2**exp.float()) * mant
        return x
    
class fake_int():
    # Fake integer
    def __init__(self, sign, wd, scale, zero_pt=0, func_round=torch.round):
        self.sign = sign  # True: signed
        self.wd = wd
        self.scale = scale
        self.zero_pt = zero_pt
        self.func_round = func_round
    
    def quant(self, x):
        x = x / self.scale 
        x = self.func_round(x)
        x = x + self.zero_pt
        if self.sign is True:
            x = x.clamp(-2**(self.wd-1), 2**(self.wd-1)-1)
        else:
            x = x.clamp(0, 2**self.wd-1)
        return x
    
    def dequant(self, x):
        x = x - self.zero_pt
        x = x * self.scale
        return x
    
    def check(self, x):
        x = self.quant(x)
        x = self.dequant(x)
        return x

fake_fp32 = fake_float(8, 23)
fake_fp16 = fake_float(5, 10)
fake_bf16 = fake_float(8, 7)
fake_tf32 = fake_float(8, 10)


####################################################
#                    Module                        #
####################################################

@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w

@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w

# weight only quantization
class q_Linear(nn.Linear):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 bias, 
                 # Weight quantization variable
                 q_wt = "per_tensor",       # "per_tensor", "per_channel", None
                 q_wt_group_size = 128,
                 q_wt_
                 
                 quant_in, 
                 quant_wt, ):
        pass
    
    
    def forward(x):
        # input & output: default float
        float_prec = x.dtype
        pass