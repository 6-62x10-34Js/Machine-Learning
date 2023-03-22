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
    # Implement the gradient descent algorithm
    # E_list should be appended in each iteration, with the current value of the cost
    
    return x, E_list


def eggholder(x):
    # Implement the cost function specified in the HW1 sheet
    z = x[0] + x[1]  # TODO: change me
    return z


def gradient_eggholder(x):
    # Implement gradients of the Eggholder function w.r.t. x and y
    grad_x = 0  # TODO: change me 
    grad_y = 0  # TODO: change me
                                      
    return np.array([grad_x, grad_y])


def generic_GD_solver(x):
    # TODO: bonus task
    return 0
