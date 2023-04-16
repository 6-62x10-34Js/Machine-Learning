import math

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_contour_w_gradient(f, start_point, recorder, E_list, learning_rate, max_iter):
    n = 1000
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(E_list, label=f"lr={learning_rate}, iters={max_iter}")
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost')
    ax1.autoscale(axis='y')
    ax1.set_title('Cost over Iteration')

    # get last item
    last = recorder[-1]

    # prepwork for plot scaling
    start_point = np.array([start_point[0], start_point[1]])
    last = np.array([last[0], last[1]])
    distance = math.dist(start_point, last)

    # plot contour plot with history andy_termination point
    x_range = np.linspace(start_point[0] - 2 * distance, last[0] + 2 * distance, n)
    y_range = np.linspace(start_point[1] - 2 * distance, last[1] + 2 * distance, n)
    # x_range = np.linspace(-20, 20, n)
    # y_range = np.linspace(-20, 20, n)
    X, Y = np.meshgrid(x_range, y_range)
    Z = f([X, Y])

    cp = ax2.contourf(X, Y, Z, levels=25, alpha=0.5)


    ax2.scatter(start_point[0], start_point[1], c='red', marker='x', label='Starting Point')
    for i, x in enumerate(recorder):
        ax2.scatter(x[0], x[1], color='black', marker='o', s=2)

    ax2.scatter(last[0], last[1], color='green', marker='x',
                label=f'Last Iteration' "\n" f'minimum at {round(last[0], 2), round(last[1], 2)}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    fig2.colorbar(cp)
    fig2.legend()
    ax2.set_title('Contour Plot with History')

    plt.title('Gradient Decent on $f(x,y)=x^2*y^2$')
    plt.savefig(f'plots/Task3/contour_and_cost{max_iter}_{learning_rate}v2.jpg', dpi=120)


def plot_eggholder_function(f):
    '''
    Plotting the 3D surface of a given cost function f.
    :param f: The function to visualize
    :return:
    '''
    n = 1000
    bounds = [-512, 512]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    x_ax = np.linspace(bounds[0], bounds[1], n)
    y_ax = np.linspace(bounds[0], bounds[1], n)
    XX, YY = np.meshgrid(x_ax, y_ax)

    ZZ = np.zeros(XX.shape)
    print(ZZ.shape)
    ZZ = f([XX, YY])

    ax.plot_surface(XX, YY, ZZ, cmap='jet')

    plt.show()


def gradient_descent(f, df, x, learning_rate, max_iter):
    """
    Find the optimal solution of the function f(x) using gradient descent:
    Until the max number of iteration is reached, decrease the parameter x by the gradient times the learning_rate.
    The function should return the minimal argument x and the list of errors at each iteration in a numpy array.

    :param f: function to minimize
    :param df: function representing the gradient of f
    :param x: vector, initial point
    :param learning_rate:
    :param max_iter: maximum number of iterations
    :return: x (solution, vector), E_list (array of errors over iterations)
    """

    E_list = np.zeros(max_iter)
    recorder = []
    record_interval = max_iter / 100
    for i in range(max_iter):

        grad = df(x)
        x = x - learning_rate * - grad.T
        error = f(x)
        E_list[i] = error
        recorder.append(x)
        if i % record_interval == 0:
            recorder.append(x)

    return x, E_list, recorder


def eggholder(x):
    # Implement the cost function specified in the HW1 sheet
    z = - (x[1] + 47) * np.sin(np.sqrt(np.abs(x[0] / 2 + (x[1] + 47)))) - x[0] * np.sin(
        np.sqrt(np.abs(x[0] - (x[1] + 47))))
    # print('x, y')
    # print(f'{x[0]}, {x[1]}')
    # z = x[0] ** 2 + x[1] ** 2
    # print('z')
    # print(z)
    return z


def div_check(x, y):
    try:
        x / y
    except ZeroDivisionError:
        return True
    else:
        return False


def gradient_eggholder(f):
    #working even though it uses wrong formula
    x = f[0]
    y = f[1]

    # Implement gradients of the Eggholder function w.r.t. x and y
    x_term1 = - (x * (-x + y + 47) * np.cos(np.sqrt(np.abs(-x + y + 47))) / (2 * np.abs(-x + y + 47) ** (3 / 2)))
    x_term2 = - ((47 + y) * (x + 2 * y + 94) * np.cos(np.sqrt(np.abs(47 + x / 2 + y))) / (
            8 * np.abs(47 + x / 2 + y) ** (3 / 2)))
    x_term3 = np.sin(np.sqrt(np.abs(-x + y + 47)))

    grad_x = - x_term3 + x_term1 - x_term2

    y_term1 = - (x * (x - y - 47) * np.cos(np.sqrt(np.abs(-x + y + 47))) / (2 * np.abs(-x + y + 47) ** (3 / 2)))
    y_term2 = - ((47 + y) * (x + 2 * y + 94) * np.cos(np.sqrt(np.abs(x / 2 + y + 47))) / (
            4 * np.abs(x / 2 + y + 47) ** (3 / 2)))
    y_term3 = np.sin(np.sqrt(np.abs(x / 2 + y + 47)))
    grad_y = - y_term3 + y_term1 - y_term2

    # grad_x = 2 * x
    # grad_y = 2 * y

    return np.array([grad_x, grad_y])


def generic_GD_solver(x):
    # TODO: bonus task
    return 0