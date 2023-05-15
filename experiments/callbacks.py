import torch
import tempfile
from copy import deepcopy


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a
    given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.delta = delta
        # self.temp_dir = tempfile.TemporaryDirectory(dir=".")
        # self.save_path = f"{self.temp_dir.name}/checkpoint.pt"

    def __call__(self, val_loss, model, epoch):
        score = -val_loss
        score_tensor = torch.Tensor([score])

        if self.best_score is None:
            # we always want to save the first set of weights
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"reached patience of {self.patience}")
                self.early_stop = True
        elif torch.isnan(score_tensor):
            print(f"Encountered NaN value at epoch {epoch}")
            self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        # torch.save(model.state_dict(), self.save_path)
        self.best_weights = deepcopy(model.state_dict())
        self.val_loss_min = val_loss

    def restore_best_weights(self, model):
        """retrieves the model's best weights and restores them"""
        print(f"Retrieving best weights for model from epoch {self.best_epoch}.")
        model.load_state_dict(self.best_weights)
        return model
