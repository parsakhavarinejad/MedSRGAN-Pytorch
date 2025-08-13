import numpy as np
import torch
import os

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=False, path='checkpoint.pth', trace_func=print):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_metric_max = -np.Inf
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_metric, generator, discriminator, epoch):
        score = val_metric
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, generator, discriminator, epoch)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, generator, discriminator, epoch)
            self.counter = 0

    def save_checkpoint(self, val_metric, generator, discriminator, epoch):
        if self.verbose:
            self.trace_func(f'Validation metric improved ({self.val_metric_max:.4f} --> {val_metric:.4f}). Saving best model...')
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'best_val_metric': val_metric
        }, self.path)
        self.val_metric_max = val_metric