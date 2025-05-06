import torch
import matplotlib.pyplot as plt

class stat_dist():
    def __init__(self, min_val, max_val, prec, func_preprcs=None):
        self.func_preprcs = func_preprcs   # pre-process function
        self.min_val = min_val
        self.max_val = max_val
        self.prec = prec
        self.num_bins = int((max_val - min_val) / prec) + 1
        self.boundaries = torch.arange(min_val, max_val+prec, prec)
        self.hist = torch.zeros_like(self.boundaries)
        self.true_max_val = None
        self.true_min_val = None
    
    def clr(self):
        self.hist = torch.zeros_like(self.hist)
        self.true_max_val = None
        self.true_min_val = None
    
    def plt(self, path="./dist.png"):
        plt.plot(self.boundaries.cpu().numpy(), self.hist.cpu().numpy())
        plt.grid(True)
        plt.savefig(path)
    
    def upd(self, tensor):
        if(self.func_preprcs != None):
            tensor = self.func_preprcs(tensor)
        if(self.true_max_val is None):
            self.true_max_val = tensor.max().item()
        else:
            self.true_max_val = max(self.true_max_val, tensor.max().item())
        if(self.true_min_val is None):
            self.true_min_val = tensor.min().item()
        else:
            self.true_min_val = min(self.true_min_val, tensor.min().item())
        
        tensor = tensor.clamp(self.min_val, self.max_val)
        self.boundaries = self.boundaries.to(tensor.device)
        self.hist = self.hist.to(tensor.device)
        tmp_bin_indices = torch.bucketize(tensor, boundaries = self.boundaries)
        tmp_hist = torch.bincount(tmp_bin_indices, minlength = self.num_bins)
        self.hist += tmp_hist
    
    def mean(self):
        return (self.boundaries * self.hist).sum() / self.hist.sum()
    
    def var(self):
        square_mean = (self.boundaries**2 * self.hist).sum() / self.hist.sum()
        return square_mean - self.mean()**2
            

class stat_delta_dist():
    def __init__(self, min_val, max_val, prec, func_preprcs=None):
        self.func_preprcs = func_preprcs   # pre-process function
        self.min_val = min_val
        self.max_val = max_val
        self.prec = prec
        self.num_bins = int((max_val - min_val) / prec) + 1
        self.boundaries = torch.arange(min_val, max_val+prec, prec)
        self.hist = None

class stat_mag():
    pass

class stat_val():
    pass
            