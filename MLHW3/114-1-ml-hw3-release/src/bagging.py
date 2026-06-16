import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import WeakClassifier

from src import utils
import random
class BaggingClassifier:
    def __init__(self, input_dim: int) -> None:
        """Free to add args as you need, like batch-size, learning rate, etc."""

        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(10)
        ]

        # my
        self.batch_size = 128

    def fit(self, X_train, y_train, num_epochs: int, learning_rate: float):
        """
        TODO: Implement the training part
        """

        n_samples = y_train.shape[0]

        
        batch_size = self.batch_size

        batch_num = int(n_samples / batch_size)
        
        
        for wc_model in self.learners:

            optimizer = optim.Adam(wc_model.parameters(), lr=learning_rate)

            # default replace=True
            idxs = np.random.randint(0, n_samples, size=n_samples)


            for epoch in range(num_epochs):
                
                np.random.shuffle(idxs)

                batch_X_train = X_train[ idxs, :]
                batch_y_train = y_train[ idxs]


                for b in range(batch_num):
                
                    X_train_mini_batch = batch_X_train[ b * batch_size: (b + 1) * batch_size, :]
                    y_train_mini_batch = batch_y_train[ b * batch_size: (b + 1) * batch_size]


                    optimizer.zero_grad()
                    outputs = wc_model(X_train_mini_batch)

                    loss_torch = utils.entropy_loss(outputs, y_train_mini_batch).squeeze()

                    loss = torch.mean(loss_torch)
            
                    loss.backward()
                    
                    optimizer.step()
                    
        return 
        
        raise NotImplementedError

    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        """
        TODO: Implement the training part
        """
        # get two type class, 
        # y is 01

        n_samples = X.shape[0]
        n_wc_models = float(len(self.learners))

        y_pred_prob_each_wc = []
        y_pred_total_score = torch.zeros(n_samples).squeeze()

        for wc_model in self.learners:
            with torch.no_grad():
                outputs = wc_model(X).squeeze()

            pred_class_01 = (outputs > -1.25).float()

            y_pred_total_score += pred_class_01

            y_pred_prob_each_wc.append(outputs)

        
        y_pred_classes_01 = ((y_pred_total_score/n_wc_models) > 0.5).float().squeeze()

        return y_pred_classes_01, y_pred_prob_each_wc
        raise NotImplementedError

    def compute_feature_importance(self) -> t.Sequence[float]:
        """
        TODO: Implement the feature importance calculation
        """
        feature_important = []

        for wc_model in self.learners:
            
            feature_weights = wc_model.fc.weight.data.abs()
            feature_important.append(feature_weights.numpy())

        return np.sum(np.array(feature_important), axis=0).reshape(-1)
        raise NotImplementedError
