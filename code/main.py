import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from gradient_descent import eggholder, gradient_eggholder, gradient_descent, plot_eggholder_function, \
    plot_contour_w_gradient
from lin_reg_memristors import test_fit_zero_intercept_lin_model, test_fit_lin_model_with_intercept, \
    fit_zero_intercept_lin_model, fit_lin_model_with_intercept, predict_fault_type

from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import timeit

cm_blue_orange = ListedColormap(['blue', 'orange'])


def task_1():
    print('---- Task 1.1 ----')
    test_fit_zero_intercept_lin_model()
    test_fit_lin_model_with_intercept()

    # Load the data from 'data/memristor_measurements.npy'
    data = np.load('data/memristor_measurements.npy')

    n_memristor = data.shape[0]

    ### --- Use Model 1 (zero-intercept lin. model, that is, fit the model using fit_zero_intercept_lin_model)
    estimated_theta_per_memristor = np.zeros(n_memristor)
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.5, wspace=0.4)
    plt.title('Task 1.1')

    zero_prediction = []
    for i in range(n_memristor):
        # Implement an approprate function call
        x = data[i, :, 0]
        y = data[i, :, 1]

        theta = fit_zero_intercept_lin_model(x, y)
        estimated_theta_per_memristor[i] = theta
        prediction = predict_fault_type(theta)
        # print(f'{prediction} wilde SHAPE')
        zero_prediction.append(prediction)

        # Visualize the data and the best fit for each memristor
        row_idx = i // 4
        col_idx = i % 4
        ax_mem = axs[row_idx, col_idx]
        plt.figure()
        ax_mem.plot(x, y, 'ko')
        x_line = np.array([np.min(x), np.max(x)])
        y_line = theta * x_line
        ax_mem.set_xlabel('Delta_R_ideal')  # Expected
        ax_mem.set_ylabel('Delta_R')  # Achieved
        ax_mem.set_title(f'Memristor {i + 1}' "\n" f'{prediction}')
        ax_mem.plot(x_line, y_line, label=f'Delta_R = {theta:.2f} * Delta_R_ideal')
        ax_mem.plot(x_line, y_line, label=f'Delta_R = {theta:.2f} * Delta_R_ideal')
        ax_mem = plt.gca()
        ax_mem.spines['right'].set_visible(False)
        ax_mem.spines['top'].set_visible(False)
        # plt.show()
        plt.close()  # Comment/Uncomment

    # plt.savefig(f'plots/model_1_memristor_collection_YES_centered.jpg', dpi=120)
    # print('Fault Prediction List')
    # print(zero_prediction)

    print('\nModel 1 (zero-intercept linear model).')
    print(f'Estimated theta per memristor: {estimated_theta_per_memristor}')

    non_zero_prediction = []

    fig1, ax1 = plt.subplots(nrows=2, ncols=4, figsize=(12, 8))
    fig1.subplots_adjust(hspace=0.5, wspace=0.4)
    plt.title('fit_lin_model_with_intercept')

    ### --- Use Model 2 (lin. model with intercept, that is, fit the model using fit_lin_model_with_intercept)    
    estimated_params_per_memristor = np.zeros((n_memristor, 2))
    for i in range(n_memristor):
        # Implement an approprate function call
        x = data[i, :, 0]
        y = data[i, :, 1]
        theta_0, theta_1 = fit_lin_model_with_intercept(x, y)
        prediction = predict_fault_type(theta_1)
        non_zero_prediction.append(prediction)
        estimated_params_per_memristor[i, :] = theta_0, theta_1

        # Visualize the data and the best fit for each memristor
        row_idx = i // 4
        col_idx = i % 4
        ax = ax1[row_idx, col_idx]
        plt.figure()
        ax.plot(x, y, 'ko')
        x_line = np.array([np.min(x), np.max(x)])
        y_line = theta_0 + theta_1 * x_line
        ax.set_xlabel('Delta_R_ideal')  # Expected
        ax.set_ylabel('Delta_R')  # Achieved
        ax.set_title(f'Memristor {i + 1}' "\n" f'{prediction}')
        ax.plot(x_line, y_line, label=f'Delta_R = {theta_0:.2f} + {theta_1:.2f} * Delta_R_ideal')
        plt.legend()
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # plt.show()
        plt.close()  # Comment/Uncomment

    print('\nModel 2 (linear model with intercept).')
    print(f"Estimated params (theta_0, theta_1) per memristor: {estimated_params_per_memristor}")

    # TODO: Use either Model 1 or Model 2 for the decision on memristor fault type. 
    # This should be a piece of code with if-statements and thresholds on parameters (you have to decide which thresholds make sense).
    print('Fault Prediction List')
    print(zero_prediction)

    # for this task the functions "decide_zero" and "decide_non_zero" have been implemented

    print(estimated_params_per_memristor.shape)
    # for i in range(len(estimated_theta_per_memristor))
    plt.show()

    plt.savefig(f'plots/model_2_memristor_collection_standart.jpg', dpi=120)


def task_2():
    print('\n---- Task 2 ----')

    def plot_datapoints(X, y, title, fig_name='fig.png'):
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
        fig.suptitle(title, y=0.93)

        p = axs.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_blue_orange)

        axs.set_xlabel('x1')
        axs.set_ylabel('x2')
        axs.legend(*p.legend_elements(), loc='best', bbox_to_anchor=(0.96, 1.15))

        # fig.savefig(fig_name) # TODO: Uncomment if you want to save it
        plt.show()  # if you want to see it now
        plt.close()  # Comment/Uncomment

    for task in [0, 1, 2]:
        print(f'---- Logistic regression task {task + 1} ----')
        if task == 0:
            # Load the data set 1 (X-1-data.npy and targets-dataset-1.npy)
            X_data = np.load('data/X-1-data.npy')
            y = np.load('data/targets-dataset-1.npy')

            # X_feature = np.array([X_data[:,0]+X_data[:,1]]).T
            X_feature = np.array([np.sign(X_data[:, 0] - 10 + 1) * np.sign(21 - X_data[:, 1]) * (
                        1 - np.sign(X_data[:, 0] - 30) * np.sign(X_data[:, 1])) * (
                                              1 - np.sign(X_data[:, 1] - 20) * np.sign(X_data[:, 0]))]).T
            # X = np.concatenate([np.ones((X_data.shape[0],1)),X_data], axis=1)# create the design matrix based on the features in X_data
            X = np.concatenate([np.ones((X_feature.shape[0], 1)), X_feature], axis=1)
        elif task == 1:
            # Load the data set 2 (X-1-data.npy and targets-dataset-2.npy)
            X_data = np.load('data/X-1-data.npy')
            y = np.load('data/targets-dataset-2.npy')

            X_feature = np.array([np.sqrt(X_data[:, 0] ** 2 + X_data[:, 1] ** 2)]).T
            # X = np.concatenate([np.ones((X_feature.shape[0],1)),X_feature], axis=1) # create the design matrix based on the features in X_data
            X = np.concatenate([np.ones((X_feature.shape[0], 1)), X_feature], axis=1)

        elif task == 2:
            # Load the data set 3 (X-2-data.npy and targets-dataset-3.npy)
            X_data = np.load('data/X-2-data.npy')
            y = np.load('data/targets-dataset-3.npy')
            X_feature = np.column_stack((X_data[:, 0], X_data[:, 1], X_data[:, 0] ** 2, X_data[:, 0] ** 3,
                                         X_data[:, 0] ** 4, X_data[:, 0] ** 5, X_data[:, 0] ** 6))
            # X = np.concatenate([np.ones((X_data.shape[0],1)),X_data], axis=1) # create the design matrix based on the features in X_data
            X = np.concatenate([np.ones((X_feature.shape[0], 1)), X_feature], axis=1)

        plot_datapoints(X_data, y, 'Targets',
                        'plots/targets_' + str(task) + '.png')  # Uncomment to generate plots as in the exercise sheet

        # Split the data into train and test sets, using train_test_split function that is already imported
        # We want 20% of the data to be in the test set. Fix the random_state parameter (use value 0)).
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
        X_data_train, X_data_test, y_data_train, y_data_test = train_test_split(X_data, y, test_size=0.20,
                                                                                random_state=0)

        print(
            f'Shapes of: X_train {X_train.shape}, X_test {X_test.shape}, y_train {y_train.shape}, y_test {y_test.shape}')

        # Create a classifier, and fit the model to the data
        # clf = # TODO use LogisticRegression from sklearn.linear_model (already imported)
        clf = LogisticRegression().fit(X_train, y_train)
        acc_train = clf.score(X_train, y_train)  # TODO
        acc_test = clf.score(X_test, y_test)
        print(f'Train accuracy: {acc_train * 100:.2f}. Test accuracy: {100 * acc_test:.2f}.')

        # Calculating the loss.
        # Calculate PROBABILITIES of predictions. Output will be with the second dimension that equals 2, because we have 2 classes.
        # (The returned estimates for all classes are ordered by the label of classes.)
        # When calculating log_loss, provide yhat_train and yhat_test of dimension (n_samples, ). That means, "reduce" the dimension,
        # simply by selecting (indexing) the probabilities of the positive class.
        y_pred_test = clf.predict_proba(X_test)
        y_pred_train = clf.predict_proba(X_train)
        loss_test = log_loss(y_test, y_pred_test)  # TODO use log_loss from sklearn.metrics (already imported)
        loss_train = log_loss(y_train, y_pred_train)
        print(f'Train loss: {loss_train:.4f}. Test loss: {loss_test:.4f}.')

        # Calculate the predictions, we need them for the plots.
        yhat_train = clf.predict(X_train)
        yhat_test = clf.predict(X_test)

        plot_datapoints(X_data_train, yhat_train, 'Predictions on the train set',
                        fig_name='logreg_train' + str(task + 1) + '.png')
        plot_datapoints(X_data_test, yhat_test, 'Predictions on the test set',
                        fig_name='logreg_test' + str(task + 1) + '.png')

        # TODO: Print the theta vector (and also the bias term). Hint: check Attributes of the classifier
        theta_vector = clf.coef_
        print('theta_vector:', theta_vector)

        # Implement Bias Term
        theta_zero = clf.intercept_
        print('theta_zero:', theta_zero)
def task_3():
    print('\n---- Task 3 ----')
    # choose starting point and define problem points
    x0 = np.array([random.randint(-512, 512), random.randint(-512, 512)])
    problem_x0_1 = np.array([47, 0])
    problem_x0_2 = np.array([52, 5])

    # Choose max_iter and learning_rates.
    learning_rates = np.array([0.05, 0.1])
    max_iters = np.array([100, 1000])

    print(f'Starting point: x={x0[0]}')
    print(f'Starting point: y={x0[1]}')
    plot_eggholder_function(eggholder)


    for learning_rate in learning_rates:
        for max_iter in max_iters:
            x, E_list, recorder = gradient_descent(eggholder, gradient_eggholder, x0, learning_rate, max_iter)
            print(f'Minimum found for {learning_rate}: f({x}) = {eggholder(x)}')
            plot_contour_w_gradient(eggholder, x0, recorder, E_list, learning_rate, max_iter)

    x_min = np.array([512, 404.2319])
    print(f'Global minimum: f({x_min}) = {eggholder(x_min)}')

    # Test 1 - Problematic point 1. See HW1, Tasks 3.6 and 3.7.
    print('A problematic point: ', gradient_eggholder([problem_x0_1[0], problem_x0_1[1]]))

    # Test 2 - Problematic point 2. See HW1, Tasks 3.6 and 3.7.
    print('Another problematic point: ', gradient_eggholder([problem_x0_2[0], problem_x0_2[1]]))


def main():
    task_1()

    task_2()
    task_3()


if __name__ == '__main__':
    main()
