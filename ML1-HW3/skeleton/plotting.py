import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# use tkgg
plt.switch_backend('TkAgg')


def plot_decision_boundary(X_train, t_train, k, score, label, classifier):
    """ Plots the decision boundary for a given training set and a given k.

    Input: X_train ... training features
           t_train ... training classes
           k ... number of neighbors to be taken into account for classifying new data

    Output: y_estimate ... estimated classes of all new data points
    """

    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

    # Create a meshgrid for the feature space
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # Create new data points
    X_new = np.c_[xx.ravel(), yy.ravel()]

    # make the prediction for the new data points
    y_estimate = classifier.predict(X_new)

    # Plot the decision boundary
    # reshape the estimated classes according to the meshgrid
    Z = y_estimate.reshape(xx.shape)

    # plot the contour lines
    plt.contourf(xx, yy, Z, cmap=cm.coolwarm, alpha=0.8)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=t_train, cmap=cm.coolwarm, s=20, edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'{label} for k={k} (score: {score:.2f})')
    plt.savefig(f'2.3_{label}_{score}.png')
    plt.show()


def plot_dataset(X_train, X_test, y_train, y_test):
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm.coolwarm, s=20, edgecolors='k')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm.coolwarm, s=20, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Dataset')

    plt.show()

def plot_training_test_over_k(k, training_scores, test_scores):
    plt.plot(k, training_scores, label='Training Scores')
    plt.plot(k, test_scores, label='Test Scores')
    plt.xlabel('k')
    plt.ylabel('Score')
    plt.title('Training and Test Scores over k')
    plt.legend()
    plt.savefig(f'training_test_over_k_1.3.png')
    plt.show()