
```python
# utils/early_stopping.py
import torch
import numpy as np
import os

class EarlyStopping:
    """
    Early stops the training if validation metric (e.g., PSNR) doesn't improve after a given patience.
    """
    def __init__(self, patience=7, min_delta=0, verbose=False, path='checkpoint.pth', trace_func=print):
        """
        Args:
            patience (int): How many epochs to wait after last improvement before stopping.
                            Default: 7
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                               Default: 0
            verbose (bool): If True, prints a message for each validation metric improvement.
                            Default: False
            path (str): Path for the checkpoint to be saved to.
                        Default: 'checkpoint.pth'
            trace_func (function): trace print function.
                                   Default: print
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_metric_max = -np.Inf # We want to maximize PSNR/SSIM
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_metric, generator, discriminator):
        """
        Checks if training should be stopped based on the validation metric.

        Args:
            val_metric (float): Current validation metric (e.g., PSNR).
            generator (torch.nn.Module): The generator model to save.
            discriminator (torch.nn.Module): The discriminator model to save.
        """
        score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, generator, discriminator)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, generator, discriminator)
            self.counter = 0

    def save_checkpoint(self, val_metric, generator, discriminator):
        """
        Saves model when validation metric improves.
        """
        if self.verbose:
            self.trace_func(f'Validation metric improved ({self.val_metric_max:.4f} --> {val_metric:.4f}). Saving model ...')
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save({
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'best_val_metric': val_metric
        }, self.path)
        self.val_metric_max = val_metric
