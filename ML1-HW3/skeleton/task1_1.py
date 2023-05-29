import numpy as np
from matplotlib import cm
from sklearn.base import BaseEstimator
from datasets import get_toy_dataset, get_heart_dataset
import matplotlib.pyplot as plt

from plotting import plot_decision_boundary


class KNearestNeighborsClassifier(BaseEstimator):
  def __init__(self, k=1):
    self.k = k
    self.X_train = None
    self.y_train = None

  def fit(self, X, y):
    # TODO IMPLEMENT ME
    self.X_train = np.array(X)
    self.y_train = np.array(y)

    return

  def score(self, X, y):
    y_pred = self.predict(X)
    return np.mean(y_pred == y)

  def predict(self, X):
    # TODO: assign class labels
    # useful numpy methods: np.argsort, np.unique, np.argmax, np.count_nonzero
    # pay close attention to the `axis` parameter of these methods
    # broadcasting is really useful for this task!
    # See https://numpy.org/doc/stable/user/basics.broadcasting.html

    distances = np.sqrt(np.sum((X[:, np.newaxis] - self.X_train[np.newaxis, :]) ** 2, axis=2))
    idx = np.argpartition(distances, self.k, axis=1)[:, :self.k]
    nearest_classes = self.y_train[idx]
    counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=len(np.unique(self.y_train))), axis=1, arr=nearest_classes)
    y_estimate = np.argmax(counts, axis=1)

    return y_estimate

def main():
    #load data
    X_train, X_test, y_train, y_test = get_toy_dataset(1)

    #create classifier
    clf = KNearestNeighborsClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    print("Accuracy:", score)

    #plt.savefig(f'{label}_{score}.png')
    plot_decision_boundary(X_train, y_train, 1, score, 'Task 1.1', classifier=clf)






if __name__ == '__main__':
    main()
