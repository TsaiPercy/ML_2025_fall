import typing as t
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_curve, auc


# my
import matplotlib.pyplot as plt

class WeakClassifier(nn.Module):
    """
    Use pyTorch to implement a 1 ~ 2 layer model.
    No non-linear activation in the `intermediate layers` allowed.
    """
    def __init__(self, input_dim):
        super(WeakClassifier, self).__init__()

        self.fc = nn.Linear(input_dim, 1)
        """
        hidden_dim = 8
        output_dim = 1

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # """

    def forward(self, x):

        x = self.fc(x)
        
        """
        x = self.fc1(x)
        x = self.fc2(x)
        # """
        return x


def entropy_loss(outputs, targets):

    """
    outputs ===> torch
    targets ===> torch
    
    """
    # sigmoid
    probs = torch.sigmoid(outputs)

    # avoid log(0)
    eps = 1e-10
    probs = torch.clamp(probs, eps, 1 - eps)

    loss_torch = - (targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))

    # loss return torch.tensor
    # ===> (n_samples,)

    return loss_torch
    raise NotImplementedError


def plot_learners_roc(
    y_preds: t.List[t.Sequence[float]],
    y_trues: t.Sequence[int],
    fpath='./tmp.png',
):
    
    plt.figure(figsize=(6, 6))

    for i, preds in enumerate(y_preds):
        FPR, TPR, _ = roc_curve(y_trues, preds)
        roc_auc = auc(FPR, TPR)

        plt.plot(FPR, TPR, label=f'AUC = {roc_auc:.2f}')

    plt.plot([0, 1], [0, 1], 'k--') #, label='Random Guess')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curves of Weak Learners')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(fpath)
    plt.close()
    return
    raise NotImplementedError


def plot_feature_importance(
    feature_importance: t.Sequence[float],
    catagories: t.Sequence[str],
    fpath='./FI.png',
):
    
    plt.figure(figsize=(10, 8))

    plt.barh(
        catagories, # y
        feature_importance, # x
        label='importance'
    )

    plt.xlabel('Importance')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()
    return
    raise NotImplementedError