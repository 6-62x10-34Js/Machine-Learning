import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from datasets import get_heart_dataset, get_toy_dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFECV
import pickle as pkl

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_toy_dataset(4)

    # TODO fit a random forest classifier and check how well it performs on the test set after tuning the parameters,
    # report your results
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    test_score = rf.score(X_test, y_test)
    print(f"Feature importances: {rf.feature_importances_} ")
    print("Test Score Random Forest:", test_score)

    # TODO fit a SVC and find suitable parameters, report your results
    svc = SVC(random_state=42)
    svc.fit(X_train, y_train)
    test_score_svc = svc.score(X_test, y_test)
    print("Test Score Support Vector Machine:", test_score_svc)

    # TODO create a bar plot (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.barh.html#matplotlib.pyplot.barh)
    # of the `feature_importances_` of the RF classifier.
    plt.barh(range(len(rf.feature_importances_)), rf.feature_importances_)
    plt.title("Feature importances")
    plt.xlabel("Importances")
    plt.ylabel("Feature Index")
    plt.show()
    # TODO create another RF classifier
    # Use recursive feature elimination (https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV)
    # to automatically choose the best number of parameters
    # set `scoring = 'accuracy'` to look for the feature subset with highest accuracy
    # and fit the RFECV to the training data
    rf = RandomForestClassifier(random_state=42)
    rfecv = RFECV(rf, scoring='accuracy')
    rfecv.fit(X_train, y_train)
    test_score = rfecv.score(X_test, y_test)
    mean_test_score = rfecv.cv_results_["mean_test_score"]
    print("Mean cross-validated Score of the RFECV:", mean_test_score[rfecv.n_features_ - 1])
    print("Test Score of the RFECV:", test_score)
    print(f"Optimal number of features: {rfecv.n_features_}")

    X_train_trans = rfecv.transform(X_train)
    X_test_trans = rfecv.transform(X_test)

    # TODO use the RFECV to transform the training and test dataset -- it automatically removes the least important
    # feature columns from the datasets. You don't have to change y_train or y_test
    # Fit a SVC classifier on the new dataset. Do you see a difference in performance?
    svc = SVC(random_state=42)
    svc.fit(X_train_trans, y_train)
    test_score_svc = svc.score(X_test_trans, y_test)
    print("Test Score Support Vector Machine transformed:", test_score_svc)
