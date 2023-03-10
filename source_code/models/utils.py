## Adapted from: https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/

import torch

class LRScheduler ():

    def __init__ (self, optimizer, patience = 10,
    min_lr = 1e-6, factor = 0.5):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
 
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode = 'min',
            patience = self.patience,
            factor = self.factor,
            min_lr = self.min_lr,
            verbose = True
            )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)

class EarlyStopping():

    def __init__(self, patience = 10, min_delta = 0):
        self.patience = patience
        self.min_delta = min_delta 
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self,  val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss -  val_loss > self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


    
