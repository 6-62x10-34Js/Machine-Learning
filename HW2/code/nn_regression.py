from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.model_selection import learning_curve


def calculate_mse(targets, predictions):
    """
    :param targets:
    :param predictions: Predictions obtained by using the model
    :return:
    """
    mse = mean_squared_error(targets, predictions)
    mse = ((targets - predictions) ** 2).mean(axis=0)

    return mse


def solve_regression_task(features, targets, n_hidden_neurons_list, optimizer_list, regulization_list, activation_list,stopping_list):
    """
    :param features:
    :param targets:
    :param n_hidden_neurons_list:
    :param optimizer_list:
    :param regulization_list:
    :return: 
    """

    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=33)
    # (try at least 3 different numbers of neurons) with grid search and add the results to the table.

    data_frame = pd.DataFrame(
        columns=['Name', 'Hidden neurons', 'Optimizer', 'Regulization', 'Activation', 'Early Stopping', 'Train loss', 'Test loss', 'R^2'])

    for activation in activation_list:
        for n_hidden_neurons in n_hidden_neurons_list:
            for optimizer in optimizer_list:
                for regulization in regulization_list:
                    for early_stopping in stopping_list:
                        model = MLPRegressor(hidden_layer_sizes=n_hidden_neurons, activation=activation, solver=optimizer,
                                             alpha=regulization, max_iter=200, random_state=33, early_stopping=early_stopping)
                        model.fit(X_train, y_train)
                        expected_y = y_test
                        predicted_y = model.predict(X_test)
                        train_loss = calculate_mse(expected_y, predicted_y)
                        test_loss = calculate_mse(y_train, model.predict(X_train))
                        r_squared = model.score(X_test, y_test)
                        name = str(n_hidden_neurons) + ', ' + str(optimizer) + ', ' + str(regulization) + ', ' + str(
                            activation) + ', ' + str(early_stopping)
                        data_frame = data_frame.append(
                            {'Name': name, 'Hidden neurons': n_hidden_neurons, 'Optimizer': optimizer,
                             'Regulization': regulization, 'Activation': activation, 'Early Stopping': early_stopping ,'Train loss': train_loss,
                             'Test loss': test_loss, 'R^2': r_squared}, ignore_index=True)

    best_model = data_frame.loc[data_frame['R^2'].idxmax()]
    #print_data_to_csv(best_model)
    # report train and test loss of the best model
    print('Best model: ' + str(best_model))
    print('With R^2: ' + str(best_model['R^2']))
    print('Train loss: ' + str(best_model['Train loss']))
    print('Test loss: ' + str(best_model['Test loss']))


    return


def plot_expected_predicted_values(expected_y, predicted_y, n_hidden_neurons, optimizer, regulization, r_squared):
    """
    :param expected_y:
    :param predicted_y:
    :param n_hidden_neurons:
    :param optimizer:
    :param regulization:
    :return:
    """
    sns.regplot(expected_y, predicted_y, fit_reg=True, scatter_kws={"s": 10}, line_kws={"color": "red"}, ci=None,
                truncate=True)
    plt.title('Hidden neurons: ' + str(n_hidden_neurons) + ', Optimizer: ' + str(optimizer) + ', Regulization: ' + str(
        regulization) + ',' + '\n' + 'R^2: ' + str(round(r_squared, 3)))
    plt.xlabel('Expected y')
    plt.ylabel('Predicted y')
    plt.legend()
    plt.savefig(f'plot_{str(n_hidden_neurons), str(optimizer), str(regulization)}.png')

    plt.show()

    return


def print_data_to_csv(data):
    """
    :param data as dataframe:
    :return:
    """
    data_frame = pd.DataFrame(data)
    data_frame.to_csv('data.csv', index=False)

    return





