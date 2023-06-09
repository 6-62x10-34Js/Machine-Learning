import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC

import plotting
from datasets import get_toy_dataset

name = []
mean_v_array = []


if __name__ == '__main__':
  for idx in [1, 2, 3]:
    X_train, X_test, y_train, y_test = get_toy_dataset(idx)
    svc = SVC(tol=1e-4)
    param_grid = {'C': [1, 10, 100], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    clf = GridSearchCV(svc, param_grid=param_grid, return_train_score=True)

    clf.fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    cross = cross_val_score(clf, X_train, y_train, cv=5)


    print(f"Dataset {idx}: {clf.best_params_}")
    print("Test Score:", test_score)
    print("Mean Cross Validation Score:", np.mean(cross))

    mean_V = np.mean(clf.cv_results_['mean_test_score'])
    mean_v_array.append(mean_V)


    print(f"Mean Test Score: {mean_V:.3f}")
    plotting.plot_decision_boundary(X_test, y_test, clf.best_params_['C'], clf.best_score_, f'Task 2.3, kernel: ' + clf.best_params_['kernel'], classifier=clf)


