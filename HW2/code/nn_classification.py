import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt


def reduce_dimension(features):
    """
    :param features: Data to reduce the dimensionality. Shape: (n_samples, n_features)
    :param n_components: Number of principal components
    :return: Data with reduced dimensionality. Shape: (n_samples, n_components)
    """

    # pca = # TODO # Create an instance of PCA from sklearn.decomposition (already imported).
    # Set the parameters (some of them are specified in the HW2 sheet).
    pca = PCA(random_state=1)

    # fit the model with features, and apply the transformation on the features
    pca.fit(features)

    # use pca inbuilt function to get the explained variance
    explained_variance = pca.explained_variance_ratio_

    # calculate the cumulative sum of the explained variance
    cum_sum = np.cumsum(explained_variance)

    # find the number of components that explain 95% of the variance
    n_components = np.argmax(cum_sum >= 0.95) + 1

    print(f'Number of components: {n_components}')

    # apply the transformation on the features
    X_reduced = pca.transform(features)[:, :n_components]

    # print the explained variance
    print(f'Explained variance: {pca.explained_variance_ratio_}')
    print(f'Number of components: {n_components}')

    return X_reduced


def train_nn(features, targets):
    """
    Train MLPClassifier with different number of neurons in one hidden layer.

    :param features:
    :param targets:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)

    param_dict = {'early_stopping': [True, False], 'hidden_layer_sizes': [(2,), (10,), (100,), (200,)]}
    result_dict = pd.DataFrame(columns=['early_stopping', 'n_hid', 'train_acc', 'test_acc', 'loss'])
    for n_hid in param_dict['hidden_layer_sizes']:
        for early_stopping in param_dict['early_stopping']:
            model = MLPClassifier(hidden_layer_sizes=n_hid, early_stopping=early_stopping, random_state=1, max_iter=500)
            model.fit(X_train, y_train)
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            loss = model.loss_

            # add results to dictionary
            result_dict = result_dict.append(
                {'early_stopping': early_stopping, 'n_hid': n_hid, 'train_acc': train_acc, 'test_acc': test_acc,
                 'loss': loss}, ignore_index=True)

            print(f'Number of neurons: {n_hid}')
            print(f'Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')
            print(f'Loss: {loss:.4f}')
            print('------------------')

    # result_dict.to_csv('result_dict.csv')
    return result_dict


def train_nn_with_regularization(features, targets):
    """
    Train MLPClassifier using regularization.

    :param features:
    :param targets:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)

    # Copy your code from train_nn, but experiment now with regularization (alpha, early_stopping).
    param_dict = {'early_stopping': [True, False], 'hidden_layer_sizes': [(2,), (10,), (100,), (200,)],
                  'alpha': [0.0001, 0.001, 0.01, 0.1]}
    result_dict = pd.DataFrame(columns=['early_stopping', 'n_hid', 'alpha', 'train_acc', 'test_acc', 'loss'])
    for n_hid in param_dict['hidden_layer_sizes']:
        # try all combinations of parameters
        for early_stopping in param_dict['early_stopping']:
            for a in param_dict['alpha']:
                model = MLPClassifier(hidden_layer_sizes=n_hid, early_stopping=early_stopping, random_state=1,
                                      max_iter=500, alpha=a)
                model.fit(X_train, y_train)
                train_acc = model.score(X_train, y_train)
                test_acc = model.score(X_test, y_test)
                loss = model.loss_

                # add results to dictionary
                result_dict = result_dict.append(
                    {'early_stopping': early_stopping, 'n_hid': n_hid, 'alpha': 0, 'train_acc': train_acc,
                     'test_acc': test_acc, 'loss': loss}, ignore_index=True)

                print(f'Number of neurons: {n_hid}')
                print(f'Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')
                print(f'Loss: {loss:.4f}')
                print('------------------')

    # resultdict saved as csv
    # result_dict.to_csv('result_dict.csv')
    #
    return result_dict


def train_nn_with_different_seeds(features, targets):
    """
    Train MLPClassifier using different seeds.
    Print (mean +/- std) accuracy on the training and test set.
    Print confusion matrix and classification report.

    :param features:
    :param targets:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)
    seeds = [240, 34, 39, 188, 44]  # TODO create a list of different seeds of your choice

    train_acc_arr = np.zeros(len(seeds))
    test_acc_arr = np.zeros(len(seeds))

    # TODO create an instance of MLPClassifier, check the perfomance for different seeds
    for i, seed in enumerate(seeds):
        model = MLPClassifier(hidden_layer_sizes=(200,), alpha=0.1, random_state=seed, max_iter=500)
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        loss = model.loss_
        print(f'Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')
        print(f'Loss: {loss:.4f}')
        train_acc_arr[i] = train_acc
        test_acc_arr[i] = test_acc

    train_acc_mean = np.mean(train_acc_arr)
    train_acc_std = np.std(train_acc_arr)
    train_acc_min = np.min(train_acc_arr)
    train_acc_max = np.max(train_acc_arr)

    test_acc_mean = np.mean(test_acc_arr)
    test_acc_std = np.std(test_acc_arr)
    test_acc_min = np.min(test_acc_arr)
    test_acc_max = np.max(test_acc_arr)
    print(f'On the train set: {train_acc_mean:.4f} +/- {train_acc_std:.4f}')
    print(f'On the test set: {test_acc_mean:.4f} +/- {test_acc_std:.4f}')
    print(f'Minimum and Maximum Test: Min:{test_acc_min:.4f}&Max:{test_acc_max:.4f}')
    print(f'Minimum and Maximum Train:MIn:{train_acc_min:.4f}& Max{train_acc_max:.4f}')

    loss_curve = model.loss_curve_

    plt.plot(loss_curve)
    plt.ylabel('Loss')
    plt.xlabel('Number of iterrations')

    print("Predicting on the test set")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred, labels=range(10)))


def perform_grid_search(features, targets):
    """
    BONUS task: Perform GridSearch using GridSearchCV.
    Create a dictionary of parameters, then a MLPClassifier (e.g., nn, set default values as specified in the HW2 sheet).
    Create an instance of GridSearchCV with parameters nn and dict.
    Print the best score and the best parameter set.

    :param features:
    :param targets:
    :return:
    """

    regularization_dict = {
        'value1': 0.0001,
        'value2': 0.01,
        'value3': 1,
        'value4': 10
    }
    optimizer_dict = {
        'adam': 'adam',
        'lbfgs': 'lbfgs'
    }
    n_hidden_neurons_dict = {
        'layer1': (100,),
        'layer2': (200,)
    }
    activations_dict = {
        'relu': 'relu',
        'logistic': 'logistic'
    }

    param_grid = {
        'hidden_layer_sizes': list(n_hidden_neurons_dict.values()),
        'activation': list(activations_dict.values()),
        'solver': list(optimizer_dict.values()),
        'alpha': list(regularization_dict.values())
    }
    param_values = list(param_grid.values())
    num_combinations = np.prod([len(values) for values in param_values])
    print(f"Number of combinations that will be checked: {num_combinations}")

    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=1)
    mlp = MLPClassifier(max_iter=100, learning_rate_init=0.01, random_state=1)
    grid_search = GridSearchCV(mlp, param_grid, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    accuracy = cross_val_score(grid_search, X_train, y_train, cv=5)
    print('------------------Overall results------------------')
    print(f'Accuracy: {accuracy.mean():.4f} +/- {accuracy.std():.4f}')
    print(f'Best Cross Validation Score: {np.max(accuracy):.4f}')
    print(f'Best score: {grid_search.best_score_}')
    print(f'Best parameters: {grid_search.best_params_}')

    # get test accuracy for best model
    best_model = grid_search.best_estimator_
    test_acc = best_model.score(X_test, y_test)
    print(f'Test accuracy of the best model: {test_acc:.4f}')
