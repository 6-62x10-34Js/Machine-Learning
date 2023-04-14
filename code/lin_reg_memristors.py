import numpy as np
from numpy.linalg import pinv  # if you decide to implement the equations using the matrix notation

def test_fit_zero_intercept_lin_model():
    # test negative values
    x1 = np.array([0., 1., 2., 3., 4., 5.])
    y1 = np.array([0, -2, -4, -6, -8, -10])
    expected_theta_1 = -2.0
    #assert fit_zero_intercept_lin_model(x1, y1) == expected_theta_1

    # test floats
    x2 = np.array([1, 3, 5, 7, 9])
    y2 = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    expected_theta_2 = 0.5
    #assert round(fit_zero_intercept_lin_model(x2, y2), 2) == expected_theta_2

def test_fit_lin_model_with_intercept():
    x1 = np.array([1, 2, 3, 4, 5])
    y1 = np.array([1, 2, 3, 4, 5])
    expected_theta_1 = 1.0
    expected_theta_0 = 0
    assert fit_lin_model_with_intercept(x1, y1) == (expected_theta_0, expected_theta_1)

def fit_zero_intercept_lin_model(x, y):
    """
    :param x: x coordinates of data points (i.e., $\Delta R_i^\text{ideal}$)
    :param y: y coordinates of data points (i.e., \Delta R_i$)
    :return: theta 
    """


    n = len(x)
    theta = (sum(x * y) / (sum(x)) ** 2) / n

    # This is the code that gives the logical results

    #theta = (sum(x * y) - sum(x) * sum(y) / n) / (sum(x ** 2) - sum(x) ** 2 / n)

    return theta


def fit_lin_model_with_intercept(x, y):
    """
    :param x: x coordinates of data points (i.e., $\Delta R_i^\text{ideal}$)
    :param y: y coordinates of data points (i.e., \Delta R_i$)
    :return: theta_0, theta_1 
    """
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    theta_0 = y_bar - sum((y - y_bar) * (x - x_bar)) / sum((x - x_bar) ** 2) * x_bar
    theta_1 = sum((y - y_bar) * (x - x_bar)) / sum((x - x_bar) ** 2)
    return theta_0, theta_1

def predict_fault_type(z):
    if z <= -0.1:
        return "k: " + f'{round(z, 2)}/ ' + '\n' + "Discordant fault"
    elif 0.1 >= z >= -0.1:
        return "k: " + f'{round(z, 2)}/ ' + '\n' + "Stuck fault"
    elif 0.9 >= z >= 0.1:
        return "k: " + f'{round(z, 2)}/ ' + '\n' + "Concordant fault under estim."
    elif 1.1 >= z >= 0.9:
        return "k: " + f'{round(z, 2)}/ ' + '\n' + "ideal"
    elif z >= 1.1:
        return "k: " + f'{round(z, 2)}/ ' + '\n' + "Concordant fault over estim."
    else:
        return "k: " + f'{round(z, 2)}/ ' + '\n' + 'not determined'
