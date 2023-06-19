import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator

import plotting
from datasets import get_toy_dataset


def loss(w, b, C, X, y):
    # TODO: implement the loss function (eq. 1)
    # useful methods: np.sum, np.clip
    margin = y * (np.dot(X, w) + b)  # shape: (n,)
    loss = np.sum(np.clip(1 - margin, 0, None)) + C * np.sum(w ** 2)  # shape: (1,)
    return loss


def grad(w, b, C, X, y):
    # TODO: implement the gradients with respect to w and b.
    # useful methods: np.sum, np.where, numpy broadcasting

    margin = y * (np.dot(X, w) + b)  # shape: (n,)
    ind = np.where(margin <= 1)[0]
    grad_w = w - C * np.sum(y[ind][:, None] * X[ind], axis=0)  # shape: (d,)
    grad_b = - C * np.sum(y[ind])  # shape: (1,)
    return grad_w, grad_b


def check_convergence(loss_list):
    if len(loss_list) > 5:
        if np.allclose(loss_list[-5:], loss_list[-1], rtol=1e-4):
            return True
    return False


class LinearSVM(BaseEstimator):
    def __init__(self, C=1, eta=0.001, max_iter=100000):
        self.C = C  # regularization parameter
        self.max_iter = max_iter
        self.eta = eta  # learning rate
        self.w = None
        self.b = None
        self.n = 0
        self.dim = 0

    def __decision_function(self, X):
        return np.dot(X, self.w) + self.b

    def get_mapping(self, y):
        return np.where(y <= 0, -1, 1)

    def fit(self, X, y):
        # convert y: {0,1} -> -1, 1
        y_map = self.get_mapping(y)
        self.n, self.dim = X.shape
        self.w = np.zeros(self.dim)
        self.b = 0

        loss_list = []

        for j in range(self.max_iter):
            grad_w, grad_b = grad(self.w, self.b, self.C, X, y_map)

            self.w -= self.eta * grad_w
            self.b -= self.eta * grad_b

            loss_list.append(loss(self.w, self.b, self.C, X, y_map))
            if check_convergence(loss_list):
                print(f"Converged after {j} iterations")
                break

        return loss_list

    def predict(self, X):
        return np.where(self.__decision_function(X) <= 0, -1, 1)

    def score(self, X, y):
        y_pred = self.predict(X)
        score = np.sum(y_pred == y) / len(y)
        return score

    def plot_decision_boundary(self, X, y):
        fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
        ax1.scatter(X[:, 0], X[:, 1], c=y, s=50, alpha=.7)
        ax1 = plt.gca()
        xlim = ax1.get_xlim()
        ylim = ax1.get_ylim()

        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = self.__decision_function(xy).reshape(XX.shape)
        ax1.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                    linestyles=['--', '-', '--'])

        ax1.set_xlabel('X1')
        ax1.set_ylabel('X2')
        ax1.set_title('Decision Boundary')
        plt.savefig('task2_1.png')
        plt.show()


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_toy_dataset(1, remove_outlier=True)
    clf = LinearSVM()
    fit = clf.fit(X_train, y_train)
    # the `fit` method returns a list of scores that you should plot in order
    # to monitor the convergence. When does the classifier converge?
    # Answer in the report or in .
    test_score = clf.score(X_test, y_test)
    print(f"Test Score: {test_score}")

    plt.plot(fit)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss vs. Iterations')
    plt.savefig('task2_1_loss.png')
    plt.show()

    # TODO plot the decision boundary!
    clf.plot_decision_boundary(X_train, y_train)
