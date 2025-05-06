

def print_cache_life(SpDiT):
    num_blocks = len(SpDiT.blocks)
    # Multi-head Attention
    for idx_block in range(num_blocks):
        life = SpDiT.blocks[idx_block].attn.cache.life
        print(f"block{idx_block:2d}_mha: {life}")
    # Feed-forward Network
    for idx_block in range(num_blocks):
        life = SpDiT.blocks[idx_block].mlp.cache.life
        print(f"block{idx_block:2d}_ffn: {life}")

class cache(object):
    # cache of layer
    def __init__(self):
        # Cached value: dict, {str: tensor}
        self.content = dict()
        # Status: int
        self.status = 0      #int, 1 for active, 0 for inactive
        # Monitor var: tensor or int
        self.life = 0        #(batch_size), times that curret cached value is reused
    
    def detach(self):
        # Detach all contents
        for key, value in self.content.items():
            if(value != None):
                self.content[key] = value.detach()
        return True

    def clr(self):
        # Clear all contents, reset status and life
        for key, value in self.content.items():
            self.content[key] = None
        self.status = 0
        self.life = 0
        return True
    
    def upd(self, interm_act, flag=1):
        '''
        interm_act: dict, {str: tensor}
        flag:       int(0/1) or tensor, (batch_size)
        '''
        # Cached value
        if(flag == 1):
            for key, value in interm_act.items():
                self.content[key] = value
        elif(flag == 0):
            pass
        else:
            gate = flag.unsqueeze(-1).unsqueeze(-1)
            # Cached value
            for key, value in interm_act.items():
                self.content[key] = gate * value + (1-gate) * self.content[key]
        # Status
        self.status = 1
        # Reuse life monitor: 1: life->0; 0: life+1
        neg_flag = 1 - flag
        self.life = self.life * neg_flag + neg_flag
        return True

