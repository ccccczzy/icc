import torch
import torch.nn as nn
import math

''' 
T: Tiny
S: Small
M: Middle
L: Large
XL, XXL, ...

Gate:  do or donot compute 
'''

def gumbel_sigmoid(logits, tau=1, hard=False, noise=True, thr=0.5):
    '''
    reference: torch.nn.functional.gumbel_softmax
    '''
    if(noise):
        gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        gumbels = (logits + gumbels) / tau
    else:
        gumbels = logits / tau

    y_soft = gumbels.sigmoid()

    if(hard):
        y_hard = (y_soft > thr) * 1.0
        ret = (y_hard - y_soft).detach() + y_soft
    else:
        ret = y_soft
    return ret

class GumbelSigmoid(nn.Module):
    def __init__(self, tau=1, thr_eval=0.5):
        super().__init__()
        self.tau = tau
        self.thr_eval = thr_eval
    def forward(self, logits):
        if(self.training):  
            y = gumbel_sigmoid(logits, tau=self.tau, hard=True, noise=False, thr=0.5)
        else:     
            y = gumbel_sigmoid(logits, tau=self.tau, hard=True, noise=False, thr=self.thr_eval)
        return y

class feature_fuse(nn.Module):
    def __init__(self, fuse_type="cat"):
        super().__init__()
        self.fuse_type = fuse_type
    def forward(self, x, cache_x):
        '''
        x/cache_x:         tensor, (batch_size, (num_wnds), hidden_size)
        return: "cat":     tensor, (batch_size, (num_wnds), 2*hidden_size)
                "add/sub": tensor, (batch_size, (num_wnds), hidden_size)
        '''
        if(self.fuse_type == "cat"):
            return torch.cat([x, cache_x], dim=-1)
        elif(self.fuse_type == "add"):
            return x + cache_x
        elif(self.fuse_type == "sub"):
            return x - cache_x
        elif(self.fuse_type == "sub_cat"):
            # sub & concat x
            return torch.cat([x - cache_x, x], dim=-1)
        elif(self.fuse_type == "sub_catc"):
            # sub & concat cache_x
            return torch.cat([x - cache_x, cache_x], dim=-1)
        else:
            raise NotImplementedError(self.fuse_type)

#################################################################################
#                                   Gate                                        #
#################################################################################

class gate_NULL(nn.Module):
    def __init__(self, hidden_size=1024, fuse_type="cat", use_cond=False, tau=1, use=True):
        super().__init__()
        self.switch = use
        self.use = use
        self.use_cond = use_cond
        if(self.use == True):
            self.gumbel_sigmoid = GumbelSigmoid(tau)
    def switch_on(self, switch=True):
        self.switch = switch
    def tau_assign(self, tau=1):
        if(self.gumbel_sigmoid != None):
            self.gumbel_sigmoid.tau = tau
    def thr_eval_assign(self, thr_eval=0.5):
        if(self.gumbel_sigmoid != None):
            self.gumbel_sigmoid.thr_eval = thr_eval
    def forward(self, x, cond, cache, i_name, o_name):
        return torch.ones(x.shape[0], device=x.device)

class gate_RAND(gate_NULL):
    def __init__(self, hidden_size=1024, fuse_type="cat", use_cond=False, tau=1, use=True):
        super().__init__()
        self.switch = use
        self.use = use
        self.use_cond = use_cond
        
    def forward(self, x, cond, cache, i_name, o_name):
        dense_prob = 0.5
        batch_size = x.shape[0]
        dense_ids = torch.rand(x.shape[0], device=x.device) < dense_prob
        gate = torch.zeros(x.shape[0], device=x.device)
        gate = torch.where(dense_ids, 1, gate)
        return gate

class gate_ACCU(gate_NULL):
    def __init__(self, hidden_size=1024, fuse_type="cat", use_cond=False, tau=1, use=True):
        super().__init__()
        # Setting
        self.switch = use
        self.use = use
        self.use_cond = use_cond
        # Module: build modules when used
        if(use == True):
            self.feature_clas = nn.Sequential(nn.Linear(hidden_size, 1, bias=True))
            self.gumbel_sigmoid = GumbelSigmoid(tau)
            self.initialize_weights()
    
    def initialize_weights(self):
        nn.init.constant_(self.feature_clas[-1].weight, 0)
        nn.init.constant_(self.feature_clas[-1].bias, 0.4)
        
    def forward(self, x, cond, cache, i_name, o_name, flag):
        # Cache read
        cache_x = cache[i_name]
        # Flatten
        delta = (x - cache_x).flatten(start_dim=1)          #(batch_size, XXX)
        x = x.flatten(start_dim=1)                          #(batch_size, XXX)
        # Compute error                                     #(batch_size)
        error = torch.norm(delta, dim=1, p=1) / torch.norm(x, dim=1, p=1)   #Relative L1
        #error = torch.norm(delta, dim=1, p=2) / torch.norm(x, dim=1, p=2)   #Relative L2
        # Accumulate error
        accu_error = cache.get("accu_error", None)
        accu_error = error if (accu_error == None) else (error + accu_error)
        # Thr
        dyn_thr = self.feature_clas(cond).squeeze(dim=1)    #(batch_size)
        logits = accu_error - dyn_thr
        if(flag == True):
            import torch.nn.functional as F
            dyn_thr2 = F.linear(cond, self.feature_clas[0].weight)
            dyn_thr3 = F.linear(cond, self.feature_clas[0].weight, self.feature_clas[0].bias)
            dyn_thr4 = dyn_thr2
            print(f"weight: {self.feature_clas[-1].weight[0, 0]}")
            print(f"bias: {self.feature_clas[-1].bias[0]}")
            print(f"dyn_thr: {dyn_thr.cpu()}")
            print(f"dyn_thr2: {dyn_thr2.cpu()}")
            print(f"dyn_thr3: {dyn_thr3.cpu()}")
        gate = self.gumbel_sigmoid(logits)
        # Update accumulated error
        cache["accu_error"] = (1-gate) * accu_error + gate * torch.zeros_like(accu_error)
        return gate

class gate_ACCUv2(gate_NULL):
    def __init__(self, hidden_size=1024, fuse_type="cat", use_cond=False, tau=1, use=True):
        super().__init__()
        # Setting
        self.switch = use
        self.use = use
        self.use_cond = use_cond
        # Module: build modules when used
        if(use == True):
            self.feature_clas = nn.Sequential(nn.Linear(2*hidden_size, 1, bias=True))
            self.gumbel_sigmoid = GumbelSigmoid(tau)

    def forward(self, x, cond, cache, i_name, o_name):
        # Cache read
        cache_x = cache[i_name]
        cache_cond = cache["cond"]
        # Flatten
        delta_cond = cond - cache_cond
        delta = (x - cache_x).flatten(start_dim=1)          #(batch_size, XXX)
        x = x.flatten(start_dim=1)                          #(batch_size, XXX)
        # Compute error                                     #(batch_size)
        #error = torch.norm(delta, dim=1, p=1) / torch.norm(x, dim=1, p=1)   #Relative L1
        error = torch.norm(delta, dim=1, p=2) / torch.norm(x, dim=1, p=2)   #Relative L2
        # Accumulate error
        accu_error = cache.get("accu_error", None)
        accu_error = error if (accu_error == None) else (error + accu_error)
        # Thr
        dyn_thr = self.feature_clas(torch.cat([cond, delta_cond], dim=-1)).squeeze(dim=1)    #(batch_size)
        #print(accu_error)
        #print(dyn_thr)
        logits = accu_error - dyn_thr
        gate = self.gumbel_sigmoid(logits)
        # Update accumulated error
        cache["accu_error"] = (1-gate) * accu_error + gate * torch.zeros_like(accu_error)
        return gate    
    
class gate_DELTA(gate_NULL):
    def __init__(self, hidden_size=1024, fuse_type="cat", use_cond=False, tau=1, use=True):
        super().__init__()
        # Setting
        self.switch = use
        self.use = use
        self.use_cond = use_cond
        # Module: build modules when used
        if(use == True):
            self.feature_clas = nn.Sequential(nn.Linear(hidden_size, 1, bias=True))
            self.gumbel_sigmoid = GumbelSigmoid(tau)
    
    def forward(self, x, cond, cache, i_name, o_name):
        # Not use cache_y for now
        cache_x = cache[i_name]
        delta = (x - cache_x).flatten(start_dim=1)          #(batch_size, XXX)
        x = x.flatten(start_dim=1)                          #(batch_size, XXX)
        # Compute error                                     #(batch_size)
        error = torch.norm(delta, dim=1, p=1) / torch.norm(x, dim=1, p=1)   #Relative L1
        #error = torch.norm(delta, dim=1, p=2) / torch.norm(x, dim=1, p=2)   #Relative L2
        dyn_thr = self.feature_clas(cond).squeeze(dim=1)    #(batch_size)
        logits = error - dyn_thr
        gate = self.gumbel_sigmoid(logits)
        return gate
    
class gate_COND(gate_NULL):
    # Only use condition as input
    # Condition can be 1)time embedding; 2)label embedding; 3)time embedding + label embedding
    def __init__(self, hidden_size=1024, fuse_type="cat", use_cond=False, tau=1, use=True):
        super().__init__()
        # Setting
        self.switch = use
        self.use = use
        self.use_cond = use_cond
        # Module: build modules when used
        if(use == True):
            if(use_cond == True):
                self.feature_clas = nn.Sequential(nn.Linear(hidden_size, 1, bias=True))
            self.gumbel_sigmoid = GumbelSigmoid(tau)
    
    def forward(self, x, cond, cache, i_name, o_name):
        if((self.use is False) or (self.switch is False)):
            batch_size, _, _ = x.shape
            return torch.ones(batch_size, device=cond.device)
        else:
            if(self.use_cond == True):
                logits = self.feature_clas(cond).squeeze(dim=1)
                gate = self.gumbel_sigmoid(logits)
            else:
                batch_size, _, _ = x.shape
                gate = torch.ones(batch_size, device=cond.device)
            return gate
                
class gate_T(gate_NULL):
    # A simple classifier head
    def __init__(self, hidden_size=1024, fuse_type="cat", use_cond=False, tau=1, use=True):
        super().__init__()
        # Setting
        self.switch = use
        self.use = use
        self.use_cond = use_cond
        # Module: build modules when used
        if(use == True):
            if(fuse_type in ["cat", "sub_cat", "sub_catc"]):
                clas_in_size = 2*hidden_size
            else:
                clas_in_size = hidden_size
            clas_in_size = clas_in_size + hidden_size   #cache_y
            if(use_cond == True):
                clas_in_size = clas_in_size + hidden_size
            self.feature_fuse = feature_fuse(fuse_type)
            # Multiple layer
            '''
            clas_mid_size = hidden_size // 8
            self.feature_clas = nn.Sequential(nn.Linear(clas_in_size, clas_mid_size, bias=True), 
                                              nn.GELU(),
                                              nn.Linear(clas_mid_size, 1, bias=True))
            '''
            # Single layer
            self.feature_clas = nn.Sequential(nn.Linear(clas_in_size, 1, bias=True)) 
            self.gumbel_sigmoid = GumbelSigmoid(tau)
        # Initialize weights
        #self.initialize_weights()
        
    def initialize_weights(self):
        pass
        #nn.init.constant_(self.feature_clas[-1].weight, 0)
        #nn.init.constant_(self.feature_clas[-1].bias, 0)
    
    def forward(self, x, cond, cache, i_name, o_name):
        if((self.use is False) or (self.switch is False)):
            batch_size, _, _ = x.shape
            return torch.ones(batch_size, device=x.device)
        else:
            cache_x = cache[i_name]
            cache_y = cache[o_name]
            #cache_y = torch.zeros_like(cache[o_name])
            # Pool 
            x, cache_x = x.mean(dim=1), cache_x.mean(dim=1) #(batch_size, hidden_size)
            cache_y = cache_y.mean(dim=1)                   #(batch_size, hidden_size)
            # Feature fuse
            x = self.feature_fuse(x, cache_x)               #(batch_size, 2*hidden_size or hidden_size)
            x = torch.cat([x, cache_y], dim=-1)             #(batch_size, 3*hidden_size or 2*hidden_size)
            if(self.use_cond):
                x = torch.cat([x, cond], dim=-1)            #(batch_size, 4*hidden_size or 3*hidden_size)
            # Classifier
            logits = self.feature_clas(x).squeeze(dim=1)    #(batch_size)
            # Gumbel Sigmoid
            gate = self.gumbel_sigmoid(logits)
            return gate

class gate_L(gate_NULL):
    # fuse + extract{DWCONV + PWCONV} + GAP + clas 
    def __init__(self, hidden_size=1024, fuse_type="cat", use_cond=False, tau=1, use=True):
        super().__init__()
        # Setting
        self.switch = use
        self.use = use
        self.use_cond = use_cond
        # Module: build modules when used
        if(use == True):
            # Channel size compute
            x_in_size = hidden_size if (fuse_type=="sub") else hidden_size*2
            x_out_size = hidden_size // 16
            y_out_size = hidden_size // 16
            clas_in_size = x_out_size + y_out_size
            if(use_cond):
                cond_out_size = hidden_size // 16
                clas_in_size = x_out_size + y_out_size + cond_out_size
            # Feature fuse of x
            self.feature_fuse_x = feature_fuse(fuse_type)
            # Feature extract
            dwconv_kernel_size = 3
            dwconv_stride = 1
            dwconv_padding = (dwconv_kernel_size - dwconv_stride) // 2
            self.feature_extract_x = nn.Sequential(nn.Conv2d(in_channels=x_in_size, out_channels=x_in_size, kernel_size=dwconv_kernel_size, stride=dwconv_stride, padding=dwconv_padding, groups=x_in_size, bias=False),
                                                   nn.ReLU(),
                                                   nn.Conv2d(in_channels=hidden_size, out_channels=x_out_size, kernel_size=1, stride=1, padding=0, bias=False),
                                                   nn.ReLU(),
                                                   nn.AdaptiveAvgPool2d((1, 1)))
            self.feature_extract_y = nn.Sequential(nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=dwconv_kernel_size, stride=dwconv_stride, padding=dwconv_padding, groups=hidden_size, bias=False),
                                                   nn.ReLU(),
                                                   nn.Conv2d(in_channels=hidden_size, out_channels=y_out_size, kernel_size=1, stride=1, padding=0, bias=False),
                                                   nn.ReLU(),
                                                   nn.AdaptiveAvgPool2d((1, 1)))
            if(use_cond == True):
                self.feature_extract_cond = nn.Sequential(nn.Linear(hidden_size, cond_out_size),
                                                          nn.ReLU())
            # Classifier
            self.feature_clas = nn.Sequential(nn.Linear(clas_in_size, 1, bias=True)) 
            self.gumbel_sigmoid = GumbelSigmoid(tau)
        # Initialize weights
        #self.initialize_weights()
        
    def initialize_weights(self):
        pass
        #nn.init.constant_(self.feature_clas[-1].weight, 0)
        #nn.init.constant_(self.feature_clas[-1].bias, 0)
    
    def forward(self, x, cond, cache, i_name, o_name):
        if((self.use is False) or (self.switch is False)):
            batch_size, _, _ = x.shape
            return torch.ones(batch_size, device=x.device)
        else:
            cache_x = cache[i_name]
            cache_y = cache[o_name]
            batch_size, num_tokens, hidden_size = x.shape
            height_width = int(math.sqrt(num_tokens))
            # Feature fuse of x & cache_x
            x = self.feature_fuse_x(x, cache_x)                  #(batch_size, num_tokens, x_in_size)
            # Reshape: (batch_size, num_tokens, C) --> (batch_size, C, H, W)
            x = x.transpose(-1, -2).reshape(batch_size, -1, height_width, height_width).contiguous()
            y = cache_y.transpose(-1, -2).reshape(batch_size, -1, height_width, height_width).contiguous()
            # Feature extract of x, y
            x = self.feature_extract_x(x).flatten(start_dim=1)   #(batch_size, x_out_size)
            y = self.feature_extract_y(y).flatten(start_dim=1)   #(batch_size, y_out_size)     
            # Feature fuse of x, y
            x = torch.cat([x, y], dim=-1)                  
            # Condition
            if(self.use_cond):
                cond = self.feature_extract_cond(cond)
                x = torch.cat([x, cond], dim=-1)           
            # Classifier
            logits = self.feature_clas(x).squeeze(dim=1)    #(batch_size)
            # Gumbel Sigmoid
            gate = self.gumbel_sigmoid(logits)
            return gate

class gate_DiTBlock(nn.Module):
    '''
    The collection of gates for a single DiT block
    Cannot be used alone
    '''
    def __init__(self, hidden_size=1024, fuse_type="cat", use_cond=False, tau=1, use_mha=True, use_ffn=True, type_mha="T", type_ffn="T"):
        super().__init__()
        name_gate_mha = "gate_" + type_mha
        name_gate_ffn = "gate_" + type_ffn
        self.gate_mha = globals()[name_gate_mha](hidden_size, fuse_type, use_cond, tau, use_mha)
        self.gate_ffn = globals()[name_gate_ffn](hidden_size, fuse_type, use_cond, tau, use_ffn)
    def switch_on(self, switch_mha=True, switch_ffn=True):
        self.gate_mha.switch_on(switch_mha)
        self.gate_ffn.switch_on(switch_ffn)
    def tau_assign(self, tau=1):
        self.gate_mha.tau_assign(tau)
        self.gate_ffn.tau_assign(tau)
    def thr_eval_assign(self, thr_eval=0.5):
        self.gate_mha.thr_eval_assign(thr_eval)
        self.gate_ffn.thr_eval_assign(thr_eval)

class gate_DiT(nn.Module):
    '''
    The collection of gates for a DiT model
    Cannot be used alone
    '''
    def __init__(self, num_blocks, hidden_size=1024, fuse_type="cat", use_cond=False, tau=1, use_mha=None, use_ffn=None, type_mha="T", type_ffn="T"):
        super().__init__()
        if(use_mha==None):
            use_mha = [True for _ in range(num_blocks)]
        if(use_ffn==None):
            use_ffn = [True for _ in range(num_blocks)]
        self.gates_blocks = nn.ModuleList()
        for idx_block in range(num_blocks):
            self.gates_blocks.append(gate_DiTBlock(hidden_size, fuse_type, use_cond, tau, use_mha[idx_block], use_ffn[idx_block], type_mha, type_ffn))

    def switch_on(self, switch_mha, switch_ffn):
        for idx_block in range(num_blocks):
            self.gates_blocks[idx_block].switch_on(switch_mha[idx_block], switch_ffn[idx_block])
    def tau_assign(self, tau=1):
        for gate in self.gates_blocks:
            gate.tau_assign(tau)
    def thr_eval_assign(self, thr_eval=0.5):
        for gate in self.gates_blocks:
            gate.thr_eval_assign(thr_eval)

            
#################################################################################
#                               Gate Config                                     #
#################################################################################
# Only gate for DiT-XL/2 is implemented
def gate_DiT_XL_2(**kwargs):
    return gate_DiT(num_blocks=28, hidden_size=1152, **kwargs)

gate_DiT_models = {'DiT-XL/2': gate_DiT_XL_2}