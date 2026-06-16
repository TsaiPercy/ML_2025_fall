import numpy as np
import pandas as pd
from loguru import logger
import random

import torch
from src import AdaBoostClassifier, BaggingClassifier, DecisionTree, utils

from src import decision_tree


def main():
    """You can control the seed for reproducibility"""
    # 777
    random.seed(82)
    np.random.seed(82)
    torch.manual_seed(82)

    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    X_train = train_df.drop(['target'], axis=1)
    y_train = train_df['target'].to_numpy()  # (n_samples, )

    X_test = test_df.drop(['target'], axis=1)
    y_test = test_df['target'].to_numpy()

    feature_names = list(train_df.drop(['target'], axis=1).columns)

    """
    TODO: Implement you preprocessing function.
    """

    def norm(X_tn, X_tt):

        MEAN = X_tn.mean(axis=0)
        STD = X_tn.std(axis=0)

        X_tn = (X_tn - MEAN) / STD
        X_tt = (X_tt - MEAN) / STD

        return X_tn, X_tt

    for c in feature_names:

        if isinstance(X_train[c][0], str):

            column_c = X_train[c].unique()

            mapping_dict = {cc: float(i) for i, cc in enumerate(column_c)}

            X_train[c] = X_train[c].map(mapping_dict)
            X_test[c] = X_test[c].map(mapping_dict)

        X_train[c], X_test[c] = norm(X_train[c], X_test[c])

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    n_feature = X_train.shape[1]

    # torch.tensor, default requires_grad=False
    # X_train = torch.tensor(X_train, requires_grad=False)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).squeeze()

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).squeeze()

    # print("preprocess finish")

    """
    TODO: Implement your ensemble methods.
    1. You can modify the hyperparameters as you need.
    2. You must print out logs (e.g., accuracy) with loguru.
    """

    # AdaBoost
    clf_adaboost = AdaBoostClassifier(input_dim=n_feature, num_learners=10, batch_size=16)

    _ = clf_adaboost.fit(X_train, y_train, num_epochs=2, learning_rate=0.01)

    y_pred_classes, y_pred_probs = clf_adaboost.predict_learners(X_test)
    accuracy_ = torch.mean((y_pred_classes == y_test).float())
    logger.info(f'AdaBoost - Accuracy: {accuracy_:.4f}')
    utils.plot_learners_roc(
        y_preds=y_pred_probs,
        y_trues=y_test,
        fpath="./ada_roc.png"
    )
    feature_importance = clf_adaboost.compute_feature_importance()

    utils.plot_feature_importance(
        feature_importance=feature_importance,
        catagories=feature_names,
        fpath="./ada_FI.png"
    )
    # print("AdaBoost finish")

    # Bagging
    clf_bagging = BaggingClassifier(input_dim=n_feature)
    _ = clf_bagging.fit(X_train, y_train, num_epochs=5, learning_rate=0.005)

    y_pred_classes, y_pred_probs = clf_bagging.predict_learners(X_test)
    accuracy_ = torch.mean((y_pred_classes == y_test).float())
    logger.info(f'Bagging - Accuracy: {accuracy_:.4f}')
    utils.plot_learners_roc(
        y_preds=y_pred_probs,
        y_trues=y_test,
        fpath="./bag_roc.png"
    )
    feature_importance = clf_bagging.compute_feature_importance()

    utils.plot_feature_importance(
        feature_importance=feature_importance,
        catagories=feature_names,
        fpath="./bag_FI.png"
    )

    X_train = X_train.numpy()
    y_train = y_train.numpy()
    X_test = X_test.numpy()
    y_test = y_test.numpy()

    # Decision Tree
    clf_tree = DecisionTree(max_depth=7)
    clf_tree.fit(X_train, y_train)
    y_pred_classes = clf_tree.predict(X_test)

    arr = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1]
    print(f"arr: {arr}")
    print(f"gini_index: {decision_tree.gini_index(np.array(arr))}")
    print(f"entropy: {decision_tree.entropy(np.array(arr))}")

    accuracy_ = np.mean(y_pred_classes == y_test)
    logger.info(f'DecisionTree - Accuracy: {accuracy_:.4f}')

    utils.plot_feature_importance(
        feature_importance=clf_tree.feature_importance,
        catagories=feature_names,
        fpath="./DT_FI.png"
    )


if __name__ == '__main__':
    main()
