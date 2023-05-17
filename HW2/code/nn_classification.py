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

    #calculate the cumulative sum of the explained variance
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

def train_nn(features, targets, n_hidden_neurons):
    """
    Train MLPClassifier with different number of neurons in one hidden layer.

    :param features:
    :param targets:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)
    result_dict = {}
    # TODO create an instance of MLPClassifier from sklearn.neural_network (already imported).
    # Set the parameters (some of them are specified in the HW2 sheet).
    # For each number of neurons in n_hidden_neurons, train the model and print the accuracy on the training and test set.

    param_dict = {'early_stopping': [True, False], 'hidden_layer_sizes': [(2,), (10,), (100,), (200,)]}
    result_dict = pd.DataFrame(columns=['early_stopping', 'n_hid', 'train_acc', 'test_acc', 'loss'])
    for n_hid in param_dict['hidden_layer_sizes']:

        for early_stopping in param_dict['early_stopping']:
            model = MLPClassifier(hidden_layer_sizes=n_hid, early_stopping=early_stopping, random_state=1, max_iter= 500)
            model.fit(X_train, y_train)
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            loss = model.loss_

            # add results to dictionary
            result_dict = result_dict.append({ 'early_stopping': early_stopping, 'n_hid': n_hid, 'train_acc': train_acc, 'test_acc': test_acc, 'loss': loss}, ignore_index=True)

            print(f'Number of neurons: {n_hid}')
            print(f'Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')
            print(f'Loss: {loss:.4f}')
            print('------------------')

    result_dict.to_csv('result_dict.csv')
    return result_dict

def plot_result_dicts(result_dict1, result_dict2):
    # Extract the alpha, early_stopping, and n_hid values from result_dict1
    alphas = list(result_dict1.keys())
    early_stopping_values = list(result_dict1[alphas[0]].keys())
    n_hid_values = list(result_dict1[alphas[0]][early_stopping_values[0]].keys())

    # Define the bar width and offset
    bar_width = 0.35
    offset = bar_width

    # Set up the figure and axis
    fig, ax = plt.subplots()

    # Iterate over the alpha, early_stopping, and n_hid values
    for i, alpha in enumerate(alphas):
        for j, early_stopping in enumerate(early_stopping_values):
            train_acc1 = [result_dict1[alpha][early_stopping][n_hid]['train_acc'] for n_hid in n_hid_values]
            test_acc1 = [result_dict1[alpha][early_stopping][n_hid]['test_acc'] for n_hid in n_hid_values]
            train_acc2 = [result_dict2[alpha][early_stopping][n_hid]['train_acc'] for n_hid in n_hid_values]
            test_acc2 = [result_dict2[alpha][early_stopping][n_hid]['test_acc'] for n_hid in n_hid_values]

            # Calculate the x position for each group
            x = np.arange(len(n_hid_values))

            # Plot the train accuracy bars for result_dict1
            ax.bar(x - offset, train_acc1, bar_width, label=f'Dict1 (Alpha: {alpha}, Early Stopping: {early_stopping})')

            # Plot the test accuracy bars for result_dict1
            ax.bar(x, test_acc1, bar_width, label=f'Dict1 (Alpha: {alpha}, Early Stopping: {early_stopping}), Early Stopping')

            # Plot the train accuracy bars for result_dict2
            ax.bar(x + offset, train_acc2, bar_width, label=f'Dict2 (Alpha: {alpha}, Early Stopping: {early_stopping})')

            # Plot the test accuracy bars for result_dict2
            ax.bar(x + 2 * offset, test_acc2, bar_width, label=f'Dict2 (Alpha: {alpha}, Early Stopping: {early_stopping}), Early Stopping')

            # Increment the offset for the next alpha
            offset += 4 * bar_width

    # Set the x-axis labels
    ax.set_xticks(np.arange(len(n_hid_values)) + bar_width * (len(early_stopping_values) - 1) / 2)
    ax.set_xticklabels(n_hid_values)

    # Set the y-axis label and title
    ax.set_ylabel('Accuracy')
    ax.set_title('Train and Test Accuracy Comparison')

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()
def compare_train_methods(result_dict1, result_dict2):
    """
    Compare the results of training with different methods.

    :param res_arr_1: Results of training without regularization
    :param res_arr_2: Results of training with regularization
    :return:
    """
    for alpha, alpha_dict1 in result_dict1.items():
        alpha_dict2 = result_dict2.get(alpha, {})

        for early_stopping, result1 in alpha_dict1.items():
            result2 = alpha_dict2.get(early_stopping, {})

            train_acc1 = result1.get('train_acc', None)
            test_acc1 = result1.get('test_acc', None)


            train_acc2 = result2.get('train_acc', None)
            test_acc2 = result2.get('test_acc', None)

            print(f"Alpha: {alpha}, Early Stopping: {early_stopping}")
            print(f"Dict1 - Train Accuracy: {train_acc1}")
            print(f"Dict1 - Test Accuracy: {test_acc1}")
            print(f"Dict2 - Train Accuracy: {train_acc2}")
            print(f"Dict2 - Test Accuracy: {test_acc2}")
            print("------------------")


def train_nn_with_regularization(features, targets):
    """
    Train MLPClassifier using regularization.

    :param features:
    :param targets:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)

    # Copy your code from train_nn, but experiment now with regularization (alpha, early_stopping).
    param_dict = {'early_stopping': [True, False], 'hidden_layer_sizes': [(2,), (10,), (100,), (200,)]}
    result_dict = pd.DataFrame(columns=['early_stopping', 'n_hid', 'train_acc', 'test_acc', 'loss'])
    for n_hid in param_dict['hidden_layer_sizes']:
        #try all combinations of parameters
        for early_stopping in param_dict['early_stopping']:
            model = MLPClassifier(hidden_layer_sizes=(n_hid,), random_state=1, max_iter=500, solver= 'adam', early_stopping=early_stopping)
            model.fit(X_train, y_train)
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            loss = model.loss_
            print(f'Number of neurons: {n_hid}')
            if early_stopping == False:
                early_stopping = 'default'

            # add results to dictionary
            result_dict = result_dict.append({'early_stopping': early_stopping, 'n_hid': n_hid, 'train_acc': train_acc, 'test_acc': test_acc, 'loss': loss}, ignore_index=True)
            print(f'Early stopping: {early_stopping}')
            print(f'Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')
            print(f'Loss: {loss:.4f}')
            print('------------------')

    #resultdict saved as csv
    result_dict.to_csv('result_dict.csv')

    # print the results
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
    seeds = [0] # TODO create a list of different seeds of your choice

    train_acc_arr = np.zeros(len(seeds))
    test_acc_arr = np.zeros(len(seeds))

    # TODO create an instance of MLPClassifier, check the perfomance for different seeds

    train_acc = 0 # TODO 
    test_acc =  0 # TODO for each seed
    loss =  0 # TODO for each seed (for you as a sanity check that the loss stays similar for different seeds, no need to include it in the report)
    print(f'Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')
    print(f'Loss: {loss:.4f}')


    train_acc_mean = 0 # TODO
    train_acc_std = 0 # TODO
    
    test_acc_mean = 0 # TODO
    test_acc_std = 0 # TODO
    print(f'On the train set: {train_acc_mean:.4f} +/- {train_acc_std:.4f}')
    print(f'On the test set: {test_acc_mean:.4f} +/- {test_acc_std:.4f}')
    # TODO: print min and max accuracy as well

    # TODO: plot the loss curve 
    # TODO: Confusion matrix and classification report (for one classifier that performs well)
    print("Predicting on the test set")
    # y_pred = 0 # TODO calculate predictions
    # print(classification_report(y_test, y_pred)) 
    # print(confusion_matrix(y_test, y_pred, labels=range(10)))


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
        'value3': 1
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
    mlp = MLPClassifier(max_iter=100, learning_rate_init = 0.01, random_state=1)
    grid_search = GridSearchCV(mlp, param_grid, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    accuracy = cross_val_score(grid_search, X_train, y_train, cv=5)
    print('------------------Overall results------------------')
    print(f'Accuracy: {accuracy.mean():.4f} +/- {accuracy.std():.4f}')
    print(f'Best Cross Validation Score: {np.max(accuracy):.4f}')
    print(f'Best score: {grid_search.best_score_}')
    print(f'Best parameters: {grid_search.best_params_}')

    #get test accuracy for best model
    best_model = grid_search.best_estimator_
    test_acc = best_model.score(X_test, y_test)
    print(f'Test accuracy of the best model: {test_acc:.4f}')



