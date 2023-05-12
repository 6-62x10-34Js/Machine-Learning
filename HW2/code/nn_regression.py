import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def calculate_mse(targets, predictions):
    """
    :param targets:
    :param predictions: Predictions obtained by using the model
    :return:
    """
    mse = mean_squared_error(targets, predictions)
    mse =((targets - predictions) ** 2).mean(axis=0)


    return mse


def solve_regression_task(features, targets, n_hidden_neurons_list, optimizer_list, regulization_list):
    """
    :param features:
    :param targets:
    :param n_hidden_neurons_list:
    :param optimizer_list:
    :param regulization_list:
    :return: 
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)

    # (try at least 3 different numbers of neurons) with grid search and add the results to the table.

    data = []

    for n_hidden_neurons in n_hidden_neurons_list:
        train_losses = []
        test_losses = []
        r_q = []
        for optimizer in optimizer_list:
            for regulization in regulization_list:
                model = MLPRegressor(hidden_layer_sizes=n_hidden_neurons, activation='logistic', solver=optimizer,
                                     alpha=regulization, max_iter=200, random_state=33)
                model.fit(X_train, y_train)
                expected_y = y_test
                predicted_y = model.predict(X_test)
                train_loss = calculate_mse(expected_y, predicted_y)
                test_loss = calculate_mse(y_train, model.predict(X_train))
                r_squared = model.score(X_test, y_test)
                test_losses.append(test_loss)
                train_losses.append(train_loss)
                r_q.append(r_squared)

                # sns.regplot(expected_y, predicted_y, fit_reg=True, scatter_kws={"s": 10}, line_kws={"color": "red"}, ci=None, truncate=True)
                # plt.title('Hidden neurons: ' + str(n_hidden_neurons) + ', Optimizer: ' + str(optimizer) + ', Regulization: ' + str(regulization)+ ','+'\n'+ 'R^2: ' + str(round(r_squared, 3)))
                # plt.xlabel('Expected y')
                # plt.ylabel('Predicted y')
                # plt.show()

                data.append([n_hidden_neurons, optimizer, regulization, train_loss, test_loss, r_q])
    best_params, worst_params, mean_params = analyze_data(np.array(data), n_hidden_neurons_list, optimizer_list, regulization_list)

    # run the model with the best parameters and compare the results with the results obtained by using the model
    # with the default parameters (you can use the same random_state in all runs).
    model = MLPRegressor(hidden_layer_sizes=best_params[2], activation='logistic', solver=best_params[0],
                         alpha=best_params[1], max_iter=200, random_state=33)
    model.fit(X_train, y_train)
    expected_y = y_test
    predicted_y = model.predict(X_test)
    train_loss = calculate_mse(expected_y, predicted_y)
    test_loss = calculate_mse(y_train, model.predict(X_train))
    r_squared = model.score(X_test, y_test)
    df_best_regression = pd.DataFrame({'Expected y': expected_y, 'Predicted y': predicted_y})

    model_worst = MLPRegressor(hidden_layer_sizes=worst_params[2], activation='logistic', solver=worst_params[0],
                            alpha=worst_params[1], max_iter=200, random_state=33)
    model_worst.fit(X_train, y_train)
    expected_y_worst = y_test
    predicted_y_worst = model_worst.predict(X_test)
    df_worst_regression = pd.DataFrame({'Expected y': expected_y_worst, 'Predicted y': predicted_y_worst})

    model_mean = MLPRegressor(hidden_layer_sizes=mean_params[2], activation='logistic', solver=mean_params[0],
                            alpha=mean_params[1], max_iter=200, random_state=33)
    model_mean.fit(X_train, y_train)
    expected_y_mean = y_test
    predicted_y_mean = model_mean.predict(X_test)
    df_mean_regression = pd.DataFrame({'Expected y': expected_y_mean, 'Predicted y': predicted_y_mean})

    df = pd.concat([df_best_regression.assign(model = 'model best'), df_worst_regression.assign(model = 'model worst'), df_mean_regression.assign(model = 'model mean')])

    fig, ax = plt.subplots()
    models = ['model best', 'model worst', 'model mean']
    for model in models:
        sns.regplot(x='Expected y', y='Predicted y', data=df[df['model'] == model], fit_reg=True, line_kws={'label': 'Regression for ' + model}, ci=None, ax=ax, truncate=True)
    plt.title('Comparison of the best, worst and mean model')
    plt.xlabel('Expected y')
    plt.ylabel('Predicted y')
    plt.legend()
    plt.show()

    print('The best train loss is: ' + str(train_loss))
    print('The best test loss is: ' + str(test_loss))
    print('The best R^2 is: ' + str(r_squared))


    return

def print_data_to_csv(data):
    """
    :param data:
    :return:
    """
    np.savetxt("data.csv", data, delimiter=",", fmt='%s')
    return
def analyze_data(data, n_hidden_neurons_list, optimizer_list, regulization_list):
    """
    :param data:

    :return:
    """
    print_data_to_csv(data)
    train_loss = data[:, 3]
    test_loss = data[:, 4]
    r_squared = data[:, 5]

    n_neurons = data[:, 0]
    optimizer = data[:, 1]
    regulization = data[:, 2]

    best_r_squared_index = np.argmax(r_squared)
    worst_r_squared_index = np.argmin(r_squared)
    mean_r_squared_index = np.argsort(r_squared)[len(r_squared) // 2]



    best_r_squared = r_squared[best_r_squared_index]
    worst_r_squared = r_squared[worst_r_squared_index]
    mean_r_squared = r_squared[mean_r_squared_index]

    # find the best train loss and test loss
    best_train_loss_index = np.argmin(train_loss)

    # find the worst train loss and test loss
    worst_train_loss_index = np.argmax(train_loss)

    # find the difference between the best and worst train loss and test loss
    train_loss_difference = abs(train_loss[best_train_loss_index] - train_loss[worst_train_loss_index])

    # median of train loss and test loss
    median_train_loss = np.median(train_loss)



    # print the results
    print('---------------------------RESULTS---------------------------------')
    print('The best train loss is: ' + str(train_loss[best_train_loss_index]))
    print('The best number of neurons is: ' + str(n_neurons[best_train_loss_index]))
    print('The best optimizer is: ' + str(optimizer[best_train_loss_index]))
    print('The best regulization is: ' + str(regulization[best_train_loss_index]))
    print(f'The best R^2 is: {best_r_squared}')
    print('....................................................................')

    print('The worst train loss is: ' + str(train_loss[worst_train_loss_index]))
    print('The worst number of neurons is: ' + str(n_neurons[worst_train_loss_index]))
    print('The worst optimizer is: ' + str(optimizer[worst_train_loss_index]))
    print('The worst regulization is: ' + str(regulization[worst_train_loss_index]))
    print(f'The worst R^2 is: {worst_r_squared}')
    print('....................................................................')
    print('The median train loss is: ' + str(median_train_loss))
    print('....................................................................')
    print('The difference between the best and worst train loss is: ' + str(train_loss_difference))

    # mean r_squared for each optimizer
    r_squared_optimizers_adam = []
    r_squared_optimizers_sgd = []
    r_squared_optimizers_lbfgs = []
    for i in range(0, len(optimizer)):
        if optimizer[i] == optimizer_list[0]:
            r_squared_optimizers_adam.append(r_squared[i])
        if optimizer[i] == optimizer_list[1]:
            r_squared_optimizers_sgd.append(r_squared[i])
        if optimizer[i] == optimizer_list[2]:
            r_squared_optimizers_lbfgs.append(r_squared[i])

    mean_r_squared_adam = np.mean(r_squared_optimizers_adam)
    mean_r_squared_sgd = np.mean(r_squared_optimizers_sgd)
    mean_r_squared_lbfgs = np.mean(r_squared_optimizers_lbfgs)

    highest_r_squared_adam = np.max(r_squared_optimizers_adam)
    highest_r_squared_sgd = np.max(r_squared_optimizers_sgd)
    highest_r_squared_lbfgs = np.max(r_squared_optimizers_lbfgs)

    lowest_r_squared_adam = np.min(r_squared_optimizers_adam)
    lowest_r_squared_sgd = np.min(r_squared_optimizers_sgd)
    lowest_r_squared_lbfgs = np.min(r_squared_optimizers_lbfgs)

    print('The mean R^2 for each optimizer')
    print(f'The mean R^2 for {optimizer_list[0]} is: {mean_r_squared_adam}')
    print(f'The mean R^2 for {optimizer_list[1]} is: {mean_r_squared_sgd}')
    print(f'The mean R^2 for {optimizer_list[2]}is: {mean_r_squared_lbfgs}')
    print('....................................................................')
    print('The highest R^2 for each optimizer')
    print(f'The highest R^2 for {optimizer_list[0]} is: {highest_r_squared_adam}')
    print(f'The highest R^2 for {optimizer_list[1]} is: {highest_r_squared_sgd}')
    print(f'The highest R^2 for {optimizer_list[2]} is: {highest_r_squared_lbfgs}')
    print('....................................................................')
    print('The lowest R^2 for each optimizer')
    print(f'The lowest R^2 for {optimizer_list[0]} is: {lowest_r_squared_adam}')
    print(f'The lowest R^2 for {optimizer_list[1]} is: {lowest_r_squared_sgd}')
    print(f'The lowest R^2 for {optimizer_list[2]} is: {lowest_r_squared_lbfgs}')
    print('....................................................................')

    # mean r_squared for each regulization
    r_squared_regulization_0_0001 = []
    r_squared_regulization_0_01 = []
    r_squared_regulization_1 = []
    for i in range(0, len(regulization)):
        if regulization[i] == regulization_list[0]:
            r_squared_regulization_0_0001.append(r_squared[i])
        if regulization[i] == regulization_list[1]:
            r_squared_regulization_0_01.append(r_squared[i])
        if regulization[i] == regulization_list[2]:
            r_squared_regulization_1.append(r_squared[i])

    mean_r_squared_regulization_0_0001 = np.mean(r_squared_regulization_0_0001)
    mean_r_squared_regulization_0_01 = np.mean(r_squared_regulization_0_01)
    mean_r_squared_regulization_1 = np.mean(r_squared_regulization_1)

    highest_regulization_0_0001 = np.max(r_squared_regulization_0_0001)
    highest_regulization_0_01 = np.max(r_squared_regulization_0_01)
    highest_regulization_1 = np.max(r_squared_regulization_1)

    lowest_regulization_0_0001 = np.min(r_squared_regulization_0_0001)
    lowest_regulization_0_01 = np.min(r_squared_regulization_0_01)
    lowest_regulization_1 = np.min(r_squared_regulization_1)

    print('The mean R^2 for each regulization')
    print(f'The mean R^2 for regulization {regulization_list[0]} is: {mean_r_squared_regulization_0_0001}')
    print(f'The mean R^2 for regulization {regulization_list[1]} is: {mean_r_squared_regulization_0_01}')
    print(f'The mean R^2 for regulization {regulization_list[2]} is: {mean_r_squared_regulization_1}')
    print('....................................................................')
    print('The highest R^2 for each regulization')
    print(f'The highest R^2 for regulization {regulization_list[0]} is: {highest_regulization_0_0001}')
    print(f'The highest R^2 for regulization {regulization_list[1]} is: {highest_regulization_0_01}')
    print(f'The highest R^2 for regulization {regulization_list[2]} is: {highest_regulization_1}')
    print('....................................................................')
    print('The lowest R^2 for each regulization')
    print(f'The lowest R^2 for regulization {regulization_list[0]} is: {lowest_regulization_0_0001}')
    print(f'The lowest R^2 for regulization {regulization_list[1]} is: {lowest_regulization_0_01}')
    print(f'The lowest R^2 for regulization {regulization_list[2]} is: {lowest_regulization_1}')

    print('....................................................................')

    # mean r_squared for each number of neurons
    r_squared_neurons_1 = []
    r_squared_neurons_2 = []
    r_squared_neurons_3 = []

    for i in range(0, len(n_neurons)):
        if n_neurons[i] == n_hidden_neurons_list[0]:
            r_squared_neurons_1.append(r_squared[i])
        if n_neurons[i] == n_hidden_neurons_list[1]:
            r_squared_neurons_2.append(r_squared[i])
        if n_neurons[i] == n_hidden_neurons_list[2]:
            r_squared_neurons_3.append(r_squared[i])

    mean_r_squared_neurons_1 = np.mean(r_squared_neurons_1)
    mean_r_squared_neurons_2 = np.mean(r_squared_neurons_2)
    mean_r_squared_neurons_3 = np.mean(r_squared_neurons_3)

    print('The mean R^2 for each number of neurons')
    print(f'The mean R^2 for {n_hidden_neurons_list[0]} hidden layer is: {mean_r_squared_neurons_1}')
    print(f'The mean R^2 for {n_hidden_neurons_list[1]} hidden layers is: {mean_r_squared_neurons_2}')
    print(f'The mean R^2 for {n_hidden_neurons_list[2]} hidden layers is: {mean_r_squared_neurons_3}')
    print('....................................................................')

    print('Overall best combination of parameters')
    print(f'Optimizer: {optimizer[best_train_loss_index]}')
    print(f'Regulization term: {regulization[best_train_loss_index]}')
    print(f'Number of neurons: {n_neurons[best_train_loss_index]}')
    print(f'With a train loss of: {train_loss[best_train_loss_index]}')
    print(f'And an $R^{2}$ of: {np.max(r_squared[best_train_loss_index])}')

    best_parameters = [optimizer[best_train_loss_index], regulization[best_train_loss_index], n_neurons[best_train_loss_index]]
    worst_paramters = [optimizer[worst_r_squared_index], regulization[worst_r_squared_index], n_neurons[worst_r_squared_index]]
    mean_parameters = [optimizer[mean_r_squared_index], regulization[mean_r_squared_index], n_neurons[mean_r_squared_index]]
    return best_parameters, worst_paramters, mean_parameters
