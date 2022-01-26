import os
import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.accs = 0
        self.F1 = 0
        self.F2 = 0
        self.F3 = 0
        self.F4 = 0
        self.val_loss_min = np.Inf
        # this help to gather preds later
        self.preds = []

    def __call__(self, report, model, modelname, str):
        score = -report.get("val_loss")
        if self.best_score is None:
            self.best_score = score
            self.accs = report.get("accuracy")
            self.F1 = report["0"]["f1-score"] if report.get("0") else 0
            self.F2 = report["1"]["f1-score"] if report.get("1") else 0
            self.F3 = report["2"]["f1-score"] if report.get("2") else 0
            self.F4 = report["3"]["f1-score"] if report.get("3") else 0
            self.preds = report["preds"]
            self.save_checkpoint(report["val_loss"], model, modelname, str)
        elif score < self.best_score:
            self.counter += 1
            # print('EarlyStopping counter: {} out of {}'.format(self.counter,self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                print(
                    "BEST Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(
                        self.accs, self.F1, self.F2, self.F3, self.F4
                    )
                )
        else:
            self.best_score = score
            self.accs = report.get("accuracy")
            self.F1 = report["0"]["f1-score"] if report.get("0") else 0
            self.F2 = report["1"]["f1-score"] if report.get("1") else 0
            self.F3 = report["2"]["f1-score"] if report.get("2") else 0
            self.F4 = report["3"]["f1-score"] if report.get("3") else 0
            self.preds = report["preds"]
            self.save_checkpoint(report["val_loss"], model, modelname, str)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, modelname, str):
        """Saves model when validation loss decrease."""
        # if self.verbose:
        #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.val_loss_min,val_loss))
        torch.save(model.state_dict(), os.path.join("output", modelname + str + ".m"))
        self.val_loss_min = val_loss
