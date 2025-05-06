import torch
import math

def batch_jvp(f, x, v):
    '''
    f:      func, y = f(x), where y is tensor, (batch_size, YYY) 
    x:      tensor, (batch_size, XXX) 
    v:      tensor, (batch_size, XXX)
    return: tensor, (batch_size, YYY), jvp = j * v, where j is tensor, (batch_size, YYY, XXX)
    '''
    jvp = torch.autograd.functional.jvp(f, x, v)  #tuple of tensor, len=batch_size
    return jvp[1]

def deriv_gelu(x, approximate=""):
    # The derivative of GELU
    if(approximate == ""):
        cdf = 0.5 * (1 + torch.erf(x / math.sqrt(2)))
        pdf = torch.exp(-0.5 * x ** 2) / math.sqrt(2 * torch.pi)
        return cdf + x * pdf
    elif(approximate == "tanh"):
        # y = 0.5x(1 + tanh(sqrt(2/pi)(x + 0.047715x^3))
        pass
    else:
        raise NotImplementError(approximate)
        
def jacob_softmax():
    pass