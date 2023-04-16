import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from gradient_descent import eggholder, gradient_eggholder, gradient_descent, plot_eggholder_function, \
    plot_contour_w_gradient
from lin_reg_memristors import test_fit_zero_intercept_lin_model, test_fit_lin_model_with_intercept, \
    fit_zero_intercept_lin_model, fit_lin_model_with_intercept, predict_fault_type
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
        ax_mem.legend()
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
        plt.close()  # Comment/Uncomment

    for task in [0, 1, 2]:
        print(f'---- Logistic regression task {task + 1} ----')
        if task == 0:
            # Load the data set 1 (X-1-data.npy and targets-dataset-1.npy)
            X_data = np.zeros((900, 2))  # TODO: change me
            y = np.zeros((900,))  # TODO: change me

            # X = TODO # create the design matrix based on the features in X_data
        elif task == 1:
            # Load the data set 2 (X-1-data.npy and targets-dataset-2.npy)
            X_data = np.zeros((900, 2))  # TODO: change me
            y = np.zeros((900,))  # TODO: change me

            # X = TODO # create the design matrix based on the features in X_data
        elif task == 2:
            # Load the data set 3 (X-2-data.npy and targets-dataset-3.npy)
            X_data = np.zeros((800, 2))  # TODO: change me
            y = np.zeros((800,))  # TODO: change me

            # X = TODO # create the design matrix based on the features in X_data

        # plot_datapoints(X, y, 'Targets', 'plots/targets_' + str(task) + '.png')  # Uncomment to generate plots as in the exercise sheet

        # Split the data into train and test sets, using train_test_split function that is already imported 
        # We want 20% of the data to be in the test set. Fix the random_state parameter (use value 0)).
        # X_train, X_test, y_train, y_test = TODO
        # print(f'Shapes of: X_train {X_train.shape}, X_test {X_test.shape}, y_train {y_train.shape}, y_test {y_test.shape}')

        # Create a classifier, and fit the model to the data
        # clf = # TODO use LogisticRegression from sklearn.linear_model (already imported)

        # acc_train, acc_test = # TODO
        # print(f'Train accuracy: {acc_train * 100:.2f}. Test accuracy: {100 * acc_test:.2f}.')

        # Calculating the loss.
        # Calculate PROBABILITIES of predictions. Output will be with the second dimension that equals 2, because we have 2 classes. 
        # (The returned estimates for all classes are ordered by the label of classes.)
        # When calculating log_loss, provide yhat_train and yhat_test of dimension (n_samples, ). That means, "reduce" the dimension, 
        # simply by selecting (indexing) the probabilities of the positive class. 

        # loss_train, loss_test = # TODO use log_loss from sklearn.metrics (already imported)
        # print(f'Train loss: {loss_train:.4f}. Test loss: {loss_test:.4f}.')

        # Calculate the predictions, we need them for the plots.
        # yhat_train = # TODO
        # yhat_test = # TODO

        # plot_datapoints(X_train, yhat_train, 'Predictions on the train set', fig_name='logreg_train' + str(task + 1) + '.png')
        # plot_datapoints(X_test, yhat_test, 'Predictions on the test set', fig_name='logreg_test' + str(task + 1) + '.png')

        # TODO: Print the theta vector (and also the bias term). Hint: check Attributes of the classifier


def task_3():
    print('\n---- Task 3 ----')

    # Plot the function, to see how it looks like
    # plot_eggholder_function(eggholder)

    # Done: choose a 2D random point from randint (-512, 512)
    x0 = np.array([random.randint(-512, 512), random.randint(-512, 512)])
    x0 = np.array([-39, -482])
    print(f'Starting point: x={x0[0]}')
    print(f'Starting point: y={x0[1]}')

    # Call the function gradient_descent. Choose max_iter, learning_rate.
    learning_rates = np.array([0.01])
    max_iter = 1000

    # i thnínk that the gradient function is wrongly implemented in the plot function

    for learning_rate in learning_rates:
        #x, E_list, recorder = gradient_descent(eggholder, gradient_eggholder, x0, learning_rate, max_iter)
        x, E_list, recorder = gradient_descent(eggholder, gradient_eggholder, x0, learning_rate, max_iter)

        print(f'Minimum found for {learning_rate}: f({x}) = {eggholder(x)}')
        plot_contour_w_gradient(eggholder, x0, recorder, E_list, learning_rate, max_iter)

    x_min = np.array([512, 404.2319])
    print(f'Global minimum: f({x_min}) = {eggholder(x_min)}')

    # Test 1 - Problematic point 1. See HW1, Tasks 3.6 and 3.7.
    x, y = 0, 0  # TODO: change me
    print('A problematic point: ', gradient_eggholder([x, y]))

    # Test 2 - Problematic point 2. See HW1, Tasks 3.6 and 3.7.
    x, y = 0, 0  # TODO: change me
    print('Another problematic point: ', gradient_eggholder([x, y]))


def main():
    # task_1()

    # task_2()
    task_3()


if __name__ == '__main__':
    main()