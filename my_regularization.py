import numpy as np
import math

class Regularization:
    def __init__(self):
        self.is_fitted = False

    def fit(self, X, y, epochs=1000, learning_rate=0.01, loss_function='mse', lambada=0.1, mode='ridge'):
        p = X.shape[1]
        np.random.seed(42)

        b = np.random.random(1)
        w = np.random.random(p)

        losses = []

        for epoch in range(epochs):
            y_pred = self.__predict(X, w, b)

            loss = self.__calculate_loss(y, y_pred, loss_function=loss_function)
            losses.append(loss)

            grad_b, grad_w = self.__calculate_gradients(X, y, y_pred)

            b = b - learning_rate * grad_b
            w = w - learning_rate * grad_w

            if epoch % 100 == 0:
                print(f'{loss_function.upper()} loss at epoch {epoch} = {loss}')

        print(f'{loss_function.upper()} loss at epoch {epochs} = {losses[-1]}')

        self.losses_ = losses
        self.w_ = w
        self.b_ = b
        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError('You need to fit the model first before prediction')

        y_pred = np.dot(X, self.w_.T) + self.b_
        return y_pred

    def __predict(self, X, w, b):
        y_pred = np.dot(X, w.T) + b
        return y_pred

    def __calculate_loss(self, y_true, y_pred, loss_function):
        n = len(y_true)

        if loss_function == 'mse':
            loss = np.sum((y_pred - y_true) ** 2) / n
        elif loss_function == 'rmse':
            loss = math.sqrt(np.sum((y_pred - y_true) ** 2) / n)
        elif loss_function == 'mae':
            loss = np.sum(abs(y_pred - y_true)) / n

        return loss

    def __calculate_gradients(self, X, y_true, y_pred):
        n = len(y_true)

        grad_b = 2 * (np.sum(y_pred - y_true)) / n
        grad_w = 2 * (np.dot((y_pred - y_true).T, X)) / n

        return grad_b, grad_w