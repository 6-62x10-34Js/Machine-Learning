import matplotlib.pyplot as plt

import plotting
from datasets import get_toy_dataset
from task2_1 import LinearSVM
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_toy_dataset(1, remove_outlier=True)
    svm = LinearSVM()
    param_dict = {'C': [1, 10, 100, 1000], 'eta': [1e-8, 1e-4, 1e-2]}
    clf = GridSearchCV(svm, param_grid=param_dict, return_train_score=True)

    clf.fit(X_train, y_train)
    # find the best model
    print(f"Best parameters: {clf.best_params_}")



    # TODO Use the parameters you have found to instantiate a LinearSVM.
    # the `fit` method returns a list of scores that you should plot in order
    # to monitor the convergence. When does the classifier converge?
    svm = LinearSVM(clf.best_params_['C'], clf.best_params_['eta'])
    scores = svm.fit(X_train, y_train)
    plt.plot(scores)
    test_score = clf.score(X_test, y_test)
    print(f"Test Score: {test_score}")

    svm.plot_decision_boundary(X_train, y_train)


    # TODO plot the decision boundary!
