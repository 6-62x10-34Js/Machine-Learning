import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


def calculate_mse(targets, predictions):
    """
    :param targets:
    :param predictions: Predictions obtained by using the model
    :return:
    """
    mse = 0  # TODO Calculate MSE using mean_squared_error from sklearn.metrics (alrady imported)
    mse = mean_squared_error(targets, predictions)

    return mse


def plot_compare_train_and_test_loss(data):
    """
    :param data:
    :return:
    """
    # plot a 1 x 3 grid of subplots, where each subplot shows the train and test loss for a specific combination of the parameters
    # (hidden neurons, optimizer, regulization)
    # The x-axis should be the number of epochs and the y-axis the MSE.
    # The title of each subplot should be the combination of the parameters.

    #split data into 3 lists
    hidden_neurons = []
    optimizer = []
    regulization = []
    train_loss = []
    test_loss = []

    for i in range(len(data)):
        hidden_neurons.append(data[i][0])
        optimizer.append(data[i][1])
        regulization.append(data[i][2])
        train_loss.append(data[i][3])
        test_loss.append(data[i][4])

    #plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].plot(train_loss[0], label='train loss')
    axs[0].plot(test_loss[0], label='test loss')
    axs[0].set_title('hidden neurons: ' + str(hidden_neurons[0]) + ', optimizer: ' + str(optimizer[0]) + ', regulization: ' + str(regulization[0]))
    axs[0].legend()

    axs[1].plot(train_loss[1], label='train loss')
    axs[1].plot(test_loss[1], label='test loss')
    axs[1].set_title('hidden neurons: ' + str(hidden_neurons[1]) + ', optimizer: ' + str(optimizer[1]) + ', regulization: ' + str(regulization[1]))
    axs[1].legend()

    axs[2].plot(train_loss[2], label='train loss')
    axs[2].plot(test_loss[2], label='test loss')
    axs[2].set_title('hidden neurons: ' + str(hidden_neurons[2]) + ', optimizer: ' + str(optimizer[2]) + ', regulization: ' + str(regulization[2]))
    axs[2].legend()
    plt.show()

    return



def solve_regression_task(features, targets):
    """
    :param features:
    :param targets:
    :return: 
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)

    # (try at least 3 different numbers of neurons) with grid search and add the results to the table.

    n_hidden_neurons_list = [(100,), (50, 50), (100, 50, 25)]

    optimizer_list = ['adam', 'sgd', 'lbfgs']

    regulization_list = [0.0001, 0.01, 1]

    data = []

    for n_hidden_neurons in n_hidden_neurons_list:
        train_losses = []
        test_losses = []
        for optimizer in optimizer_list:
            for regulization in regulization_list:
                model = MLPRegressor(hidden_layer_sizes=n_hidden_neurons, activation='relu', solver=optimizer,
                                     alpha=regulization, max_iter=200, random_state=33)
                model.fit(X_train, y_train)

                # record train and test loss arrays for plotting


                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                train_loss = calculate_mse(y_train, y_train_pred)
                test_loss = calculate_mse(y_test, y_test_pred)
                print(
                    f'Hidden neurons: {n_hidden_neurons}, Optimizer: {optimizer}, Regulization: {regulization}, Train loss: {train_loss}, Test loss: {test_loss}')
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                data.append([n_hidden_neurons, optimizer, regulization, train_loss, test_loss])
    #plot_compare_train_and_test_loss(data)
    analyze_data(np.array(data))
    return

def analyze_data(data):
    """
    :param data:

    :return:
    """

    train_loss = data[:, 3]
    test_loss = data[:, 4]

    n_neurons = data[:, 0]
    optimizer = data[:, 1]
    regulization = data[:, 2]

    #find the best train loss and test loss
    best_train_loss_index = np.argmin(train_loss)
    best_test_loss_index = np.argmin(test_loss)

    #find the worst train loss and test loss
    worst_train_loss_index = np.argmax(train_loss)
    worst_test_loss_index = np.argmax(test_loss)

    #find the difference between the best and worst train loss and test loss
    train_loss_difference = abs(train_loss[best_train_loss_index] - train_loss[worst_train_loss_index])
    test_loss_difference = abs(test_loss[best_test_loss_index] - test_loss[worst_test_loss_index])

    # median of train loss and test loss
    median_train_loss = np.median(train_loss)
    median_test_loss = np.median(test_loss)


    #print the results
    print('---------------------------RESULTS---------------------------------')
    print('The best train loss is: ' + str(train_loss[best_train_loss_index]))
    print('The best test loss is: ' + str(test_loss[best_test_loss_index]))
    print('The best number of neurons is: ' + str(n_neurons[best_train_loss_index]))
    print('The best optimizer is: ' + str(optimizer[best_train_loss_index]))
    print('The best regulization is: ' + str(regulization[best_train_loss_index]))
    print('....................................................................')

    print('The worst train loss is: ' + str(train_loss[worst_train_loss_index]))
    print('The worst test loss is: ' + str(test_loss[worst_test_loss_index]))
    print('The worst number of neurons is: ' + str(n_neurons[worst_train_loss_index]))
    print('The worst optimizer is: ' + str(optimizer[worst_train_loss_index]))
    print('The worst regulization is: ' + str(regulization[worst_train_loss_index]))
    print('....................................................................')
    print('The median train loss is: ' + str(median_train_loss))
    print('The median test loss is: ' + str(median_test_loss))
    print('....................................................................')
    print('The difference between the best and worst train loss is: ' + str(train_loss_difference))
    print('The difference between the best and worst test loss is: ' + str(test_loss_difference))





    return