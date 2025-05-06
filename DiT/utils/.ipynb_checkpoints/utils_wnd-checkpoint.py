# Window-based operations
import torch
import torch.nn.functional as F

def to_2tuple(x):
    if isinstance(x, tuple) and len(x) == 2:
        return x
    elif isinstance(x, int):
        return (x, x)
    raise ValueError("Input must be an integer or a tuple of length 2")


###################################################
#                    Shuffle                      #
###################################################

def wnd_shuffle(x, wnd_size, num_tokens_hw):
    '''
    x:             tensor, (batch_size, num_tokens, hidden_size), where num_tokens = num_tokens_dim1 x num_tokens_dim2
    wnd_size:      int or 2tuple, size of the window
    num_tokens_hw: int ot 2tuple, number of tokens along the height and/or width
    '''
    wnd_size_dim1, wnd_size_dim2 = to_2tuple(wnd_size)
    num_tokens_dim1, num_tokens_dim2 = to_2tuple(num_tokens_hw)
    num_wnds_dim1 = num_tokens_dim1 // wnd_size_dim1
    num_wnds_dim2 = num_tokens_dim2 // wnd_size_dim2

    batch_size, num_tokens, hidden_size = x.shape
    # (batch_size, num_wnds_dim1, wnd_size_dim1, num_wnds_dim2, wnd_size_dim2, hidden_size)
    x = x.reshape(batch_size, num_wnds_dim1, wnd_size_dim1, num_wnds_dim2, wnd_size_dim2, hidden_size)
    # (batch_size, num_wnds_dim1, num_wnds_dim2, wnd_size_dim1, wnd_size_dim2, hidden_size)
    x = x.permute(0, 1, 3, 2, 4, 5)
    # (batch_size, num_tokens, hidden_size)
    x = x.flatten(start_dim=1, end_dim=4)  
    return x

def wnd_deshuffle(x, wnd_size, num_tokens_hw):
    '''
    x:             tensor, (batch_size, num_tokens, hidden_size), where num_tokens = num_wnds x num_tokens_wnd
    wnd_size:      int or 2tuple
    num_tokens_hw: int ot 2tuple
    '''
    wnd_size_dim1, wnd_size_dim2 = to_2tuple(wnd_size)
    num_tokens_dim1, num_tokens_dim2 = to_2tuple(num_tokens_hw)
    num_wnds_dim1 = num_tokens_dim1 // wnd_size_dim1
    num_wnds_dim2 = num_tokens_dim2 // wnd_size_dim2

    batch_size, num_tokens, hidden_size = x.shape
    # (batch_size, num_wnds_dim1, num_wnds_dim2, wnd_size_dim1, wnd_size_dim2, hidden_size)
    x = x.reshape(batch_size, num_wnds_dim1, num_wnds_dim2, wnd_size_dim1, wnd_size_dim2, hidden_size)
    # (batch_size, num_wnds_dim1, wnd_size_dim1, num_wnds_dim2, wnd_size_dim2, hidden_size)
    x = x.permute(0, 1, 3, 2, 4, 5)  
    # (batch_size, num_tokens, hidden_size)                                             
    x = x.flatten(start_dim=1, end_dim=4)
    return x


###################################################
#                    Delta                        #
###################################################

def snake_scan_flatten(x):
    # scan dim2 first --> then scan dim1
    # x: tensor, (batch_size, num_wnds, wnd_size_dim1, wnd_size_dim2, hidden_size)
    size_dim1 = x.shape[2]
    flat_x = torch.clone(x)
    for idx_dim1 in range(size_dim1):
        if(idx_dim1 % 2 == 1):
            flat_x[:, :, idx_dim1, :, :] = flat_x[:, :, idx_dim1, :, :].flip(dims=[2])
        else:
            pass
            # flat_x[:, :, idx_dim1, :, :] = flat_x[:, :, idx_dim1, :, :]
    flat_x = flat_x.flatten(start_dim=2, end_dim=3)
    return flat_x

def intra_wnd_delta(x, wnd_size, num_tokens_hw, scan_mode="snake"):
    '''
    scan_mode:   str, 
    1) "snake":   dim2 --> dim1
    2) 
    '''
    wnd_size_dim1, wnd_size_dim2 = to_2tuple(wnd_size)
    num_tokens_dim1, num_tokens_dim2 = to_2tuple(num_tokens_hw)
    num_wnds_dim1 = num_tokens_dim1 // wnd_size_dim1
    num_wnds_dim2 = num_tokens_dim2 // wnd_size_dim2
    num_wnds = num_wnds_dim1 * num_wnds_dim2
    batch_size, num_tokens, hidden_size = x.shape
    # (batch_size, num_wnds, wnd_size_dim1, wnd_size_dim2, hidden_size)
    x = x.reshape(batch_size, num_wnds, wnd_size_dim1, wnd_size_dim2, hidden_size)
    
    # Flatten as scan_mode
    # (batch_size, num_wnds, num_tokens_wnd, hidden_size)
    if(scan_mode == "snake"):
        x = snake_scan_flatten(x)
    else:
        x = snake_scan_flatten(x)

    # Calculate delta
    x_last = x[:, :, :-1, :]
    x_next = x[:, :, 1:, :]
    delta = x_next - x_last                  #(batch_size, num_wnds, num_tokens_wnd-1, hidden_size)
    start_point = x[:, :, 0, :].unsqueeze(2) #(batch_size, num_wnds, 1, hidden_size)
    return start_point, delta


def token_sim(x):
    '''
    x:      tensor, (batch_size, num_tokens, hidden_size)
    return: tensor, (batch_size, num_tokens, num_tokens)
    '''
    x1 = x.unsqueeze(2) #(batch_size, num_tokens, 1, hidden_size)
    x2 = x.unsqueeze(1) #(batch_size, 1, num_tokens, hidden_size)
    sim = F.cosine_similarity(x1, x2, dim=-1)
    return sim
    




    
    













###################################################
#                Scatter-Gather                   #
###################################################
def wnd_gather():
    pass

def wnd_scatter():
    pass





    