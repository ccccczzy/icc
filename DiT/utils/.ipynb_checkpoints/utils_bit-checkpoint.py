import torch

def to_2tuple(x):
    if isinstance(x, tuple) and len(x) == 2:
        return x
    elif isinstance(x, int):
        return (x, x)
    raise ValueError("Input must be an integer or a tuple of length 2")
    

    

def shared_exp_extract(x, wnd_size, num_tokens_hw, group_size=None):
    '''
    x:                  tensor, (batch_size, num_tokens, hidden_size)
    return: exp_share:  tensor, (batch_size, num_wnds, 1, num_groups, 1)
            exp_delta:  tensor, (batch_size, num_wnds, num_tokens_wnd, num_groups, group_size)
            mant_shift: tensor, (batch_size, num_wnds, num_tokens_wnd, num_groups, group_size)
    '''
    wnd_size_dim1, wnd_size_dim2 = to_2tuple(wnd_size)
    num_tokens_dim1, num_tokens_dim2 = to_2tuple(num_tokens_hw)
    num_wnds_dim1 = num_tokens_dim1 // wnd_size_dim1
    num_wnds_dim2 = num_tokens_dim2 // wnd_size_dim2
    num_wnds = num_wnds_dim1 * num_wnds_dim2
    num_tokens_wnd = wnd_size_dim1 * wnd_size_dim2
    
    batch_size, num_tokens, hidden_size = x.shape
    group_size = hidden_size if (group_size is None) else group_size
    # (batch_size, num_wnds, num_tokens_wnd, num_groups, group_size)
    x = x.reshape(batch_size, num_wnds, num_tokens_wnd, hidden_size // group_size, group_size)
    # Field decomp
    x = x.to(torch.float32)
    mant, exp = torch.frexp(x)
    # Delta & shift
    exp_share = exp.max(dim=-1, keepdim=True)[0]       #(batch_size, num_wnds, num_tokens_wnd, num_groups, 1)
    exp_share = exp_share.max(dim=2, keepdim=True)[0] #(batch_size, num_wnds, 1, num_groups, 1)
    exp_delta = exp_share - exp                        
    mant_shift = mant / 2**exp_delta         
    return exp_share, exp_delta, mant_shift

###################################################
#                    Delta                        #
###################################################

def snake_scan(x):
    '''
    scan dim2 first --> then scan dim1
    x:      tensor, (batch_size, num_wnds, wnd_size_dim1, wnd_size_dim2, hidden_size)
    return: tensor, (batch_size, num_wnds, wnd_size_dim1, wnd_size_dim2, hidden_size)
    '''
    size_dim1 = x.shape[2]
    flat_x = torch.clone(x)
    for idx_dim1 in range(size_dim1):
        if(idx_dim1 % 2 == 1):
            flat_x[:, :, idx_dim1, :, :] = flat_x[:, :, idx_dim1, :, :].flip(dims=[2])
        else:
            pass
            # flat_x[:, :, idx_dim1, :, :] = flat_x[:, :, idx_dim1, :, :]
    return flat_x

def intra_wnd_delta(x, wnd_size, num_tokens_hw, group_size, func_scan = snake_scan):
    '''
    x:                tensor, (batch_size, num_tokens, hidden_size), has been shuffled
    return: start_pt: tensor, (batch_size, num_wnds, 1, num_groups, group_size)
            delta:    tensor, (batch_size, num_wnds, num_tokens_wnd-1, num_groups, group_size)
    '''
    wnd_size_dim1, wnd_size_dim2 = to_2tuple(wnd_size)
    num_tokens_dim1, num_tokens_dim2 = to_2tuple(num_tokens_hw)
    num_wnds_dim1 = num_tokens_dim1 // wnd_size_dim1
    num_wnds_dim2 = num_tokens_dim2 // wnd_size_dim2
    num_wnds = num_wnds_dim1 * num_wnds_dim2
    batch_size, num_tokens, hidden_size = x.shape
    # Reshape
    # (batch_size, num_wnds, wnd_size_dim1, wnd_size_dim2, hidden_size)
    x = x.reshape(batch_size, num_wnds, wnd_size_dim1, wnd_size_dim2, hidden_size)
    
    # Scan and flatten
    # (batch_size, num_wnds, wnd_size_dim1, wnd_size_dim2, hidden_size)
    x = func_scan(x)   
    # (batch_size, num_wnds, num_tokens_wnd, num_groups, group_size)
    x = x.flatten(start_dim=2, end_dim=3).reshape(batch_size, num_wnds, -1, hidden_size // group_size, group_size)

    # Calculate delta
    x_last = x[:, :, :-1, :, :]              #(batch_size, num_wnds, num_tokens_wnd-1, num_groups, group_size)
    x_next = x[:, :, 1:, :, :]               #(batch_size, num_wnds, num_tokens_wnd-1, num_groups, group_size)
    start_pt = x[:, :, 0, :, :].unsqueeze(2) #(batch_size, num_wnds, 1, num_groups, group_size)
    delta = x_next - x_last                  #(batch_size, num_wnds, num_tokens_wnd-1, num_groups, group_size)
    return start_pt, delta



def proj_scale(vec1, vec2, dim=-1, keepdim=True):
    '''
    Project vec1 to vec2 direction, vec2_proj_vec1 = scale * vec1, where scale is a scalar
    vec1:   tensor, (XXX, size_dim)
    vec2:   tensor, (XXX, size_dim)
    return: tensor, (XXX) if keepdim=False else (XXX, 1)
    '''
    inner_prod = (vec1 * vec2).sum(dim=dim, keepdim=keepdim)
    square_vec2 = (vec2 ** 2).sum(dim=dim, keepdim=keepdim)
    scale = inner_prod / square_vec2
    return scale

def intra_cube_scale_delta(x, wnd_size, num_tokens_hw, group_size=None, use_scale = True, func_scan = snake_scan):
    '''
    x:                tensor, (batch_size, num_tokens, hidden_size), has been shuffled
    return: start_pt: tensor, (batch_size, num_wnds, 1, num_groups, group_size)
            delta:    tensor, (batch_size, num_wnds, num_tokens_wnd-1, num_groups, group_size)
    '''
    wnd_size_dim1, wnd_size_dim2 = to_2tuple(wnd_size)
    num_tokens_dim1, num_tokens_dim2 = to_2tuple(num_tokens_hw)
    num_wnds_dim1 = num_tokens_dim1 // wnd_size_dim1
    num_wnds_dim2 = num_tokens_dim2 // wnd_size_dim2
    num_wnds = num_wnds_dim1 * num_wnds_dim2
    batch_size, num_tokens, hidden_size = x.shape
    # Reshape
    # (batch_size, num_wnds, wnd_size_dim1, wnd_size_dim2, hidden_size)
    x = x.reshape(batch_size, num_wnds, wnd_size_dim1, wnd_size_dim2, hidden_size)
    group_size = hidden_size if (group_size is None) else group_size
    
    # Scan and flatten
    # (batch_size, num_wnds, wnd_size_dim1, wnd_size_dim2, hidden_size)
    x = func_scan(x)   
    # (batch_size, num_wnds, num_tokens_wnd, num_groups, group_size)
    x = x.flatten(start_dim=2, end_dim=3).reshape(batch_size, num_wnds, -1, hidden_size // group_size, group_size)

    # Calculate delta
    x_last = x[:, :, :-1, :, :]                              #(batch_size, num_wnds, num_tokens_wnd-1, num_groups, group_size)
    x_next = x[:, :, 1:, :, :]                               #(batch_size, num_wnds, num_tokens_wnd-1, num_groups, group_size)
    start_pt = x[:, :, 0, :, :].unsqueeze(2)                 #(batch_size, num_wnds, 1, num_groups, group_size)
    if(use_scale):
        #(batch_size, num_wnds, num_tokens_wnd-1, num_groups, 1)
        scale = proj_scale(x_next, x_last, dim=-1, keepdim=True)
    else:
        scale = 1
    delta = x_next - scale * x_last                          #(batch_size, num_wnds, num_tokens_wnd-1, num_groups, group_size)
    return start_pt, delta, scale
    
