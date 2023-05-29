import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV

import plotting
from datasets import get_toy_dataset
from task1_1 import KNearestNeighborsClassifier

if __name__ == '__main__':
  for idx in [1, 2, 3]:
    X_train, X_test, y_train, y_test = get_toy_dataset(idx)
    knn = KNearestNeighborsClassifier()
    #TODO: use the `GridSearchCV` meta-classifier and search over different values of `k`!
    # include the `return_train_score=True` option to get the training accuracies

    clf = GridSearchCV(knn, param_grid={'k': np.arange(1, 100)}, return_train_score=True)
    clf.fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print(f"Test Score: {test_score}")
    print(f"Dataset {idx}: {clf.best_params_}")

    plt.figure()
    plotting.plot_decision_boundary(X_train, y_train, idx, test_score, 'Task 1.2', classifier=clf)
    plotting.plot_dataset(X_train, X_test, y_train, y_test)
    # TODO you should use the plt.savefig(...) function to store your plots before calling plt.show()
    plotting.plot_training_test_over_k(clf.cv_results_['param_k'], clf.cv_results_['mean_train_score'], clf.cv_results_['mean_test_score'])

    plt.show()

