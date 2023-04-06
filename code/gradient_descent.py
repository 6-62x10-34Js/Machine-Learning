import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_eggholder_function(f):
    '''
    Plotting the 3D surface of a given cost function f.
    :param f: The function to visualize
    :return:
    '''
    n = 1000
    bounds = [-512, 512]

    fig = plt.figure(figsize=(10,6))
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

    for i in range(max_iter):
        x = x - learning_rate * df(x)
        E_list[i] = f(x)

    return x, E_list


def eggholder(x):
    # Implement the cost function specified in the HW1 sheet
    z = - (x[1] + 47) * np.sin(np.sqrt(abs(x[0] / 2 + (x[1] + 47)))) - x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47))))
    #z = x[0] + x[1]  # TODO: change me

    return z


def gradient_eggholder(f):
    x = f[0]
    y = f[1]
    # Implement gradients of the Eggholder function w.r.t. x and y
    common_term = abs(47 + x / 2 + y) ** (3 / 2)
    x_first = x * (-47 + x - y) * np.cos(np.sqrt(abs(47 + x - y))) / (2 * common_term)
    x_second = (47 + y) * np.cos(np.sqrt(abs(47 + x / 2 + y))) / (4 * common_term)
    x_third = np.sin(np.sqrt(47 - x + y))

    grad_x = x_first - x_second - x_third

    common_term = abs(-47 + x - y) ** (3 / 2)
    y_first = x * (-47 + x - y) * np.cos(np.sqrt(abs(-47 + x - y))) / (2 * common_term)
    y_second = (47 + y) * np.cos(np.sqrt(abs(47 + x / 2 + y))) / (2 * abs(47 + x / 2 + y) ** (3 / 2))
    y_third = np.sin(np.sqrt(abs(47 - x / 2 + y)))

    grad_y = y_first - y_second - y_third
                                      
    return np.array([grad_x, grad_y])


def generic_GD_solver(x):
    # TODO: bonus task
    return 0
