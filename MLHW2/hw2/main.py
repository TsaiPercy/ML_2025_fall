import typing as t

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger

from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, learning_rate: float = 1e-4, num_iterations: int = 100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.intercept = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:
        """
        Implement your fitting function here.
        The weights and intercept should be kept in self.weights and self.intercept.
        """
        targets_new = targets.reshape(-1)

        inputs_ext = np.hstack([inputs, np.ones((inputs.shape[0], 1))])

        params = np.zeros(inputs_ext.shape[1])

        '''
        print(f"inputs_ext.shape: {inputs_ext.shape}")
        print(f"targets_new.shape: {targets_new.shape}")
        print(f"params.shape: {params.shape}")
        '''

        # Gradient_descent
        for iter in range(self.num_iterations):

            a = params @ inputs_ext.T
            y = self.sigmoid(a)

            gradient_loss = inputs_ext.T @ (y - targets_new)
            params -= self.learning_rate * gradient_loss

            '''
            loss = (-1.0) * np.sum(targets_new * np.log(y) + (1.0 - targets_new) * np.log(1.0 - y))
            '''

            y_label = (y >= 0.5).astype(int)

            Acc = accuracy_score(targets_new, y_label)

            if iter % 1000 == 1:
                # logger.info(f'{loss=:.4f}')
                logger.info(f'Iteration {iter}, Train-{Acc=:.4f}')

        self.weights = params[:-1]
        self.intercept = params[-1]
        return
        raise NotImplementedError

    def predict(
            self,
            inputs: npt.NDArray[float]
    ) -> t.Tuple[t.Sequence[np.float64], t.Sequence[int]]:
        """
        Implement your prediction function here.
        The return should contains
        1. sample probabilty of being class_1
        2. sample predicted class
        """

        a = self.weights @ inputs.T + self.intercept

        y_pred_probs = self.sigmoid(a)
        y_pred_classes = (y_pred_probs >= 0.5).astype(int)

        return y_pred_probs, y_pred_classes
        raise NotImplementedError

    def sigmoid(self, x):
        """
        Implement the sigmoid function.
        """
        return 1.0 / (1.0 + np.exp(-x))
        raise NotImplementedError


class FLD:
    """Implement FLD
    You can add arguments as you need,
    but don't modify those already exist variables.
    """
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None
        self.itrcp_DB = None
        self.itrcp_PL = None
        self.threshold = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:

        '''
        print(f"inputs.shape: {inputs.shape}")
        print(f"targets.shape: {targets.shape}")
        '''

        class_0 = inputs[targets == 0, :]
        class_1 = inputs[targets == 1, :]

        self.m0 = np.mean(class_0, axis=0)
        self.m1 = np.mean(class_1, axis=0)

        '''
        print(f"self.m0.shape: {self.m0.shape}")
        print(f"self.m1.shape: {self.m1.shape}")
        '''

        diff_0_mean = class_0 - self.m0
        diff_1_mean = class_1 - self.m1

        '''
        print(f"diff_0_mean.shape: {diff_0_mean.shape}")
        print(f"diff_1_mean.shape: {diff_1_mean.shape}")
        '''

        self.sw = diff_0_mean.T @ diff_0_mean + diff_1_mean.T @ diff_1_mean

        diff = (self.m1 - self.m0).reshape(-1, 1)
        self.sb = diff @ diff.T

        '''
        print(f"diff.shape: {diff.shape}")
        print(f"self.sw.shape: {self.sw.shape}")
        print(f"self.sb.shape: {self.sb.shape}")
        '''

        self.w = np.linalg.pinv(self.sw) @ diff
        self.w = (self.w / np.linalg.norm(self.w)).reshape(-1)

        # print(f"self.w.shape: {self.w.shape}")

        self.slope = -self.w[0] / self.w[1]

        proj_m0 = self.m0 @ self.w
        proj_m1 = self.m1 @ self.w

        self.threshold = (proj_m0 + proj_m1) / 2

        self.itrcp_DB = self.threshold / self.w[1]

        m = 0.5 * (self.m0 + self.m1)

        self.itrcp_PL = m[1] - (-1.0 / self.slope) * m[0]

        return
        raise NotImplementedError

    def predict(
            self,
            inputs: npt.NDArray[float]
    ) -> t.Sequence[t.Union[int, bool]]:

        y = inputs @ self.w
        y_pred_classes = (y >= self.threshold).astype(int)

        return y_pred_classes
        raise NotImplementedError

    def plot_projection(
            self,
            x_test: npt.NDArray[float],
            y_test: t.Sequence[t.Union[int]]
    ) -> None:

        y_pred_classes = self.predict(x_test)
        PL_slope = (-1.0 / self.slope)
        accuracy = accuracy_score(y_test, y_pred_classes)

        '''
        print(f"x_test.shape: {x_test.shape}")
        print(f"y_test.shape: {y_test.shape}")
        print(f"y_pred_classes.shape: {y_pred_classes.shape}")
        '''

        plt.xlim(-1.0, 1.5)
        plt.ylim(-1.0, 1.5)
        plt.axis('equal')

        x_vals = np.linspace(-1.0, 2.0, 100)
        y_vals = self.slope * x_vals + self.itrcp_DB
        plt.plot(x_vals, y_vals, color='blue', linestyle='--', label='Decision boundary')

        x_vals = np.linspace(-0.7, 1.3, 100)
        y_vals = PL_slope * x_vals + self.itrcp_PL
        plt.plot(x_vals, y_vals, color='gray', label='Projection Line')

        up_PL_idx = ((-1.0 / self.slope) * x_test[:, 0] + self.itrcp_PL) < 0
        dw_PL_idx = ((-1.0 / self.slope) * x_test[:, 0] + self.itrcp_PL) >= 0

        Dot_Line_color = [0.8, 0.8, 0.8]

        up_PL_x = (self.itrcp_PL - x_test[up_PL_idx, 1] + self.slope * x_test[up_PL_idx, 0]) / (self.slope - PL_slope)
        up_PL_y = self.slope * (up_PL_x - x_test[up_PL_idx, 0]) + x_test[up_PL_idx, 1]

        x_vals = np.linspace(x_test[up_PL_idx, 0], up_PL_x, 100)
        y_vals = np.linspace(x_test[up_PL_idx, 1], up_PL_y, 100)
        plt.plot(x_vals, y_vals, color=Dot_Line_color, label='Dot Line')

        dw_PL_x = (self.itrcp_PL - x_test[dw_PL_idx, 1] + self.slope * x_test[dw_PL_idx, 0]) / (self.slope - PL_slope)
        dw_PL_y = self.slope * (dw_PL_x - x_test[dw_PL_idx, 0]) + x_test[dw_PL_idx, 1]

        x_vals = np.linspace(x_test[dw_PL_idx, 0], dw_PL_x, 100)
        y_vals = np.linspace(x_test[dw_PL_idx, 1], dw_PL_y, 100)
        plt.plot(x_vals, y_vals, color=Dot_Line_color, label='Dot Line')

        idx_0 = (y_test == 0)
        idx_1 = (y_test == 1)

        plt.scatter(
            x_test[idx_0, 0],
            x_test[idx_0, 1],
            marker='o',
            color=np.where(y_pred_classes[idx_0] == y_test[idx_0], 'green', 'red'),
            label='Class 0',
            zorder=3
        )

        plt.scatter(
            x_test[idx_1, 0],
            x_test[idx_1, 1],
            marker='^',
            color=np.where(y_pred_classes[idx_1] == y_test[idx_1], 'green', 'red'),
            label='Class 1',
            zorder=3
        )

        point_size = 10

        plt.scatter(
            up_PL_x,
            up_PL_y,
            marker='o',
            color="gray",
            s=point_size,
            zorder=3
        )

        plt.scatter(
            dw_PL_x,
            dw_PL_y,
            marker='o',
            color="gray",
            s=point_size,
            zorder=3
        )

        plt.title(f"Projection onto FLD axis (w={self.w})\n\
        Slope={PL_slope:4f}, Intercept_of_PL={self.itrcp_PL:4f}, Accuracy={accuracy:4f}")

        plt.savefig("FLD.png", dpi=300)

        return
        raise NotImplementedError


def compute_auc(y_trues, y_preds):
    return roc_auc_score(y_trues, y_preds)
    raise NotImplementedError


def accuracy_score(y_trues, y_preds):
    return np.sum(y_trues == y_preds).astype(np.float64) / y_trues.shape[0]
    raise NotImplementedError


def main():
    # Read data
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    # Part1: Logistic Regression
    x_train = train_df.drop(['target'], axis=1).to_numpy()  # (n_samples, n_features)
    y_train = train_df['target'].to_numpy()  # (n_samples, )
    print(y_train.shape)

    x_test = test_df.drop(['target'], axis=1).to_numpy()
    y_test = test_df['target'].to_numpy()

    LR = LogisticRegression(
        learning_rate=1e-3,  # You can modify the parameters as you want
        num_iterations=1000,  # You can modify the parameters as you want
    )
    LR.fit(x_train, y_train)
    y_pred_probs, y_pred_classes = LR.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)
    auc_score = compute_auc(y_test, y_pred_probs)
    logger.info(f'LR: Weights: {LR.weights[:5]}, Intercep: {LR.intercept}')
    logger.info(f'LR: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}')

    # Part2: FLD
    cols = ['27', '30']  # Dont modify
    x_train = train_df[cols].to_numpy()
    y_train = train_df['target'].to_numpy()
    x_test = test_df[cols].to_numpy()
    y_test = test_df['target'].to_numpy()

    FLD_ = FLD()
    """
    (TODO): Implement your code to
    1) Fit the FLD model
    2) Make prediction
    3) Compute the evaluation metrics

    Please also take care of the variables you used.
    """
    FLD_.fit(x_train, y_train)
    y_pred_classes = FLD_.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)

    logger.info(f'FLD: m0={FLD_.m0}, m1={FLD_.m1} of {cols=}')
    logger.info(f'FLD: \nSw=\n{FLD_.sw}')
    logger.info(f'FLD: \nSb=\n{FLD_.sb}')
    logger.info(f'FLD: \nw=\n{FLD_.w}')
    logger.info(f'FLD: Accuracy={accuracy:.4f}')

    """
    (TODO): Implement your code below to plot the projection
    """
    FLD_.plot_projection(x_test, y_test)


if __name__ == '__main__':
    main()
