import torch
import torch.nn as nn
import math

# For static graph 

class sched_PERIOD():
    # Decide compute or not based on a fixed period
    def __init__(self, num_steps=100, period=2):
        self.num_steps = num_steps #int
        self.period = period       #int
        # Gate lists init: 1 means update
        self.gate_list = [0] * num_steps
        for i in range(num_steps):
            self.gate_list[i] = 1 if (i % period == 0) else 0
        
    def clr(self):
        self.idx_step_rev = 0
        
    def step(self):
        gate = self.gate_list[self.idx_step_rev]
        self.idx_step_rev += 1
        return gate
    

def get_sched(num_steps, sched_type, sched_kwargs):
    sched_name = "sched_" + sched_type
    sched = globals()[sched_name](num_steps, **sched_kwargs)
    return sched


class sched_PERIOD_v2():
    def __init__(self, num_steps=100, period=2):
        self.num_steps = num_steps
        self.period = period       #int
        # Gate lists init: 1 means update
        self.gate_list_label = [0] * num_steps
        self.gate_list_null = [0] * num_steps
        # Labeled: 1 1 0 1 0 1 0 1
        for i in range(num_steps):
            self.gate_list_label[i] = 1 if ((i+1) % period == 0) else 0
        self.gate_list_label[0] = 1
        # Null: 1 0 1 0 1 0 1 0
        for i in range(num_steps):
            self.gate_list_null[i] = 1 if (i % period == 0) else 0
    
    def clr(self):
        self.idx_step_rev = 0
        
    def step(self):
        gate_label = self.gate_list_label[self.idx_step_rev]
        gate_null = self.gate_list_null[self.idx_step_rev]
        self.idx_step_rev += 1
        return gate_label, gate_null
            
    
    