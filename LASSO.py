import numpy as np


def objective(X, Y, w, lamb):
    """ Returns our LASSO objective depending on our parameters.

    :param X: Input data
    :param Y: True output
    :param w: Parameters of affine function fit
    :param lamb: Regularization parameter
    :return: value: The objective function value
    """
    # Squared Error
    L_w = np.linalg.norm(Y - X@w, ord=2)**2

    # Regularization
    R_w = lamb*np.linalg.norm(w, ord=1)

    # Gradient of Squared Error
    Grad_L = X.T@(2*(Y - X@w))

    return L_w, R_w, Grad_L

