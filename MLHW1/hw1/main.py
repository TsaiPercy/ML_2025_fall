"""
1. Complete the implementation for the `...` part
2. Feel free to take strategies to make faster convergence
3. You can add additional params to the Class/Function as you need. But the key print out should be kept.
4. Traps in the code. Fix common semantic/stylistic problems to pass the linting
"""

from loguru import logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearRegressionBase:
    def __init__(self):
        self.weights = None
        self.intercept = None

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class LinearRegressionCloseform (LinearRegressionBase):
    def fit(self, X, y):
        """Question1
        Complete this function
        """

        '''
        y = n * 1
        X = n * d
        W = (d+1) * 1
        '''
        X_ext = np.hstack([X, np.ones((X.shape[0], 1))])
        params = np.linalg.pinv(X_ext.T @ X_ext) @ X_ext.T @ y
        self.weights = params[:-1]
        self.intercept = params[-1]

    def predict(self, X):
        """Question4
        Complete this function
        """
        return X @ self.weights + self.intercept


class LinearRegressionGradientdescent (LinearRegressionBase):
    def fit(
        self,
        X,
        y,
        learning_rate,
        epochs: int
    ):
        """Question2
        Complete this function
        """

        losses, lr_history = [], []
        lr = learning_rate
        batch_size = 32

        params = np.zeros(X.shape[1] + 1)
        y_new = (np.array((y.reshape(-1)))).astype(np.float64)

        n = y_new.shape[0]
        X_ext = (np.hstack([X, np.ones((X.shape[0], 1))])).astype(np.float64)

        # lr = 0.0003, epoch = 1000000, 0.005%
        # after std, lr = 0.01, epoch = 1500 0.000%
        '''for epoch in range(epochs):

            now_val = X_ext @ params

            loss = compute_mse(now_val, y_new)

            losses.append(loss)
            lr_history.append(lr)

            gradient_w = np.array((-1 / n) * (X_ext.T @ (y_new - now_val)))

            gradient_w = (gradient_w.reshape(-1)).astype(np.float64)

            params = params - lr * gradient_w

            if epoch % 1000 == 0:
                logger.info(f'EPOCH {epoch}, {loss=:.4f}, {lr=:.4f}')'''

        # seed 60, epoch 20000, diff 0.056%
        np.random.seed(60)
        for epoch in range(epochs):

            indices = np.arange(n)
            np.random.shuffle(indices)
            X_ext = X_ext[indices]
            y_new = y_new[indices]

            for i in range(int(n / batch_size)):

                mn_b_X = X_ext[i * batch_size: (i + 1) * batch_size, :]
                mn_b_y = y_new[i * batch_size: (i + 1) * batch_size]

                b_val = mn_b_X @ params

                r = (mn_b_y - b_val)

                lr = ((r.T @ r) / (r.T @ mn_b_X @ mn_b_X.T @ r) / batch_size)

                gradient_w = np.array((-1 / batch_size) * (mn_b_X.T @ r))

                gradient_w = (gradient_w.reshape(-1)).astype(np.float64)

                params = params - lr * gradient_w

            now_val = X_ext @ params

            loss = compute_mse(now_val, y_new)

            losses.append(loss)
            lr_history.append(lr)

            if epoch % 1000 == 0:
                logger.info(f'EPOCH {epoch}, {loss=:.4f}, {lr=:.4f}')

        self.weights = params[:-1]
        self.intercept = params[-1]

        return losses, lr_history

    def predict(self, X):
        """Question4
        Complete this
        """
        return X @ self.weights + self.intercept


def compute_mse(prediction, ground_truth):
    mse = np.mean((prediction - ground_truth) ** 2)
    return mse


def main():
    train_df = pd.read_csv('./train.csv')  # Load training data
    test_df = pd.read_csv('./test.csv')  # Load test data
    train_x = train_df.drop(["Performance Index"], axis=1).to_numpy()
    train_y = train_df["Performance Index"].to_numpy()
    test_x = test_df.drop(["Performance Index"], axis=1).to_numpy()
    test_y = test_df["Performance Index"].to_numpy()

    '''# std
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std
    print(mean.shape)'''

    LR_CF = LinearRegressionCloseform()
    LR_CF.fit(train_x, train_y)

    """This is the print out of question1"""
    logger.info(f'{LR_CF.weights=}, {LR_CF.intercept=:.4f}')

    LR_GD = LinearRegressionGradientdescent()
    losses, lr_history = LR_GD.fit(train_x, train_y, learning_rate=0.01, epochs=20000)
    # losses, lr_history = LR_GD.fit(train_x, train_y, learning_rate=0.01, epochs=1500)

    """
    This is the print out of question2
    Note: You need to screenshot your hyper-parameters as well.
    """
    logger.info(f'{LR_GD.weights=}, {LR_GD.intercept=:.4f}')

    """
    Question3: Plot the learning curve.
    Implement here
    """

    plt.plot(np.arange(0, len(losses)), losses, label='Train MSE loss')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training loss")
    plt.legend()
    plt.show()

    """Question4"""
    y_preds_cf = LR_CF.predict(test_x)
    y_preds_gd = LR_GD.predict(test_x)
    y_preds_diff = np.abs(y_preds_gd - y_preds_cf).mean()
    logger.info(f'Prediction difference: {y_preds_diff:.4f}')

    mse_cf = compute_mse(y_preds_cf, test_y)
    mse_gd = compute_mse(y_preds_gd, test_y)
    diff = (np.abs(mse_gd - mse_cf) / mse_cf) * 100
    logger.info(f'{mse_cf=:.4f}, {mse_gd=:.4f}. Difference: {diff:.3f}%')


if __name__ == '__main__':
    main()
