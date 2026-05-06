import numpy as np
import torch
import os

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=200, verbose=False, delta=0):
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, classifier, classifier_e):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, classifier, classifier_e)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, classifier, classifier_e)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, classifier, classifier_e):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path1 = os.path.join(self.save_path, 'best_network.pth')
        path2 = os.path.join(self.save_path, 'classifier.pth')
        path3 = os.path.join(self.save_path, 'classifier_e.pth')
        torch.save(model, path1)
        torch.save(classifier, path2)
        torch.save(classifier_e, path3)
        self.val_loss_min = val_loss

        torch.save(model.state_dict(), os.path.join(self.save_path, 'best_network_state_dict.pth'))
        torch.save(classifier.state_dict(), os.path.join(self.save_path, 'classifier_state_dict.pth'))
        if classifier_e is not None:
            torch.save(classifier_e.state_dict(), os.path.join(self.save_path, 'classifier_e_state_dict.pth'))


