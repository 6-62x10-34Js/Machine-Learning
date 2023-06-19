import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV

import plotting
from datasets import get_toy_dataset
from task1_1 import KNearestNeighborsClassifier
from sklearn.model_selection import cross_val_score

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_toy_dataset(2, apply_noise=True)
    cross_val_array = []
    for k in [1, 20, 100]:
        clf = KNearestNeighborsClassifier(k)
        cross_val = cross_val_score(clf, X_train, y_train, cv=5)
        cross_val_array.append(cross_val)
        print(f"Cross Validation Score for k={k}: {cross_val}")
        clf.fit(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        print(f"Test Score for k={k}: {test_score:.2f}")
        plotting.plot_decision_boundary(X_train, y_train, k, test_score, f'Task 1.3', classifier=clf)

    knn = KNearestNeighborsClassifier()
    param_grid = {'k': np.arange(1, 100)}
    clf = GridSearchCV(knn, param_grid=param_grid, return_train_score=True)
    clf.fit(X_train, y_train)
    #determine the best parameters for the classifier using 5-fold cross validation
    cross_scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(f"Cross Validation Scores: {cross_scores}")
    print(f'Mean Cross Validation Score: {np.mean(cross_scores)}, Standard Deviation: {np.std(cross_scores)}')

    # such as the `mean_train_score` and `mean_test_score`. Plot these values as a function of `k` and report the best
    # parameters. Is the classifier very sensitive to the choice of k?
    plotting.plot_training_test_over_k(clf.cv_results_['param_k'], clf.cv_results_['mean_train_score'], clf.cv_results_['mean_test_score'])
    print(f"Best Parameters: {clf.best_params_}")

    plt.show()

