import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import WeakClassifier

from src import utils

class AdaBoostClassifier:
    def __init__(self, input_dim: int, num_learners: int = 10, batch_size: int = 16) -> None:
        """Free to add args as you need, like batch-size, learning rate, etc."""

        self.sample_weights = None
        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(num_learners)
        ]
        self.alphas = []

        # my
        self.batch_size = batch_size
        


    def fit(self, X_train, y_train, num_epochs: int = 1000, learning_rate: float = 0.01):
        """
        TODO: Implement the training part
        """

        """
        X_train ===> torch tensor
        y_train ===> torch tensor
        """

    
        n_samples = y_train.shape[0]

        # D init
        self.sample_weights = torch.full((n_samples,), 1.0 /n_samples).squeeze()

        batch_size = self.batch_size

        batch_num = int(n_samples / batch_size)
        
        # print(batch_num)

        for wc_model in self.learners:

            optimizer = optim.Adam(wc_model.parameters(), lr=learning_rate)

            for epoch in range(num_epochs):
                
                idxs = np.arange(n_samples)
                np.random.shuffle(idxs)
                batch_X_train = X_train[idxs]
                batch_y_train = y_train[idxs]
                batch_sample_weights = self.sample_weights[idxs]

                for b in range(batch_num):
                
                    X_train_mini_batch = batch_X_train[ b * batch_size: (b + 1) * batch_size, :]
                    y_train_mini_batch = batch_y_train[ b * batch_size: (b + 1) * batch_size]

                    optimizer.zero_grad()
                    outputs = wc_model(X_train_mini_batch)

                    loss_torch = utils.entropy_loss(outputs, y_train_mini_batch).squeeze()

                    loss = torch.sum(batch_sample_weights[b * batch_size: (b + 1) * batch_size] * loss_torch)
            
                    loss.backward()
                    
                    optimizer.step()


            # get two type class, 
            # y_train is 01
            # but adaboost formula is +-1

            with torch.no_grad():
                outputs = wc_model(X_train).squeeze()

            pred_class_01 = (outputs > 0).float()
            pred_class_np1 = (pred_class_01 * 2) - 1.0


            # compute epsilon
            F_idx = (pred_class_01 != y_train)
            epsilon = torch.sum(self.sample_weights * F_idx)

            ## prevent divide 0
            epsilon = torch.clamp(epsilon, 1e-10, 1 - 1e-10)

            # compute alpha
            alpha = 0.5 * torch.log((1 - epsilon) / epsilon)
            self.alphas.append(alpha)

            y_train_np1 = (y_train * 2) - 1.0
            ayh = torch.exp(-alpha * y_train_np1 * pred_class_np1)


            self.sample_weights *= ayh
            self.sample_weights /= torch.sum(self.sample_weights)

            # print(f"error={epsilon:.4f}, alpha={alpha:.4f}")

        return 
        raise NotImplementedError

    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        """
        TODO: Implement the prediction
        """
        
        """
        X ===> torch
        """
        n_samples = X.shape[0]

        y_pred_prob_each_wc = []
        y_pred_total_score = torch.zeros(n_samples).squeeze()

        for alpha, wc_model in zip(self.alphas, self.learners):

            with torch.no_grad():
                outputs = wc_model(X).squeeze()

            pred_class_01 = (outputs > 0).float()
            pred_class_np1 = (pred_class_01 * 2) - 1.0

            y_pred_total_score += alpha * pred_class_np1
        
            y_pred_prob_each_wc.append(outputs)


        
        y_pred_classes_01 = (y_pred_total_score > 0).float().squeeze()


        return y_pred_classes_01, y_pred_prob_each_wc
        raise NotImplementedError


    def compute_feature_importance(self) -> t.Sequence[float]:
        """
        TODO: Implement the feature importance calculation
        """
        feature_important = []

        for alpha, wc_model in zip(self.alphas, self.learners):
            
            feature_weights = wc_model.fc.weight.data.abs()
            feature_important.append((feature_weights * alpha).numpy())

        return np.sum(np.array(feature_important), axis=0).reshape(-1)
    
        raise NotImplementedError
