import numpy as np


class Meter(object):
    '''Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    '''
    def add(self):
        '''Log a new value to the meter.'''
        pass

    def value(self):
        '''Get the value of the meter in the current state.'''
        pass
      
    def reset(self):
      '''Resets the meter to default settings.'''
      pass


class AverageMeter(Meter):
    def __init__(self):
        super(AverageMeter, self).__init__()
        self.reset()

    def add(self, value, n=1):
        self.val = value
        self.sum += value * n
        self.n += n
        
        if self.n == 0:
            self.avg = self.sum
        else:
            self.avg = self.sum / self.n
        
    def value(self):
        return self.avg

    def reset(self):
        self.n = 0
        self.sum = 0
        self.val = 0
        self.avg = 0
        
        
def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
        
    assert len(base_lrs) == len(optimizer.param_groups)
    
    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
            
    return _lr_adjuster
