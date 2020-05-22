import numpy as np
import matplotlib.pyplot as plt
from LASSO import objective
from LoadData import load_data
from scipy.optimize import minimize


def ista_regression():
    """ Solves our regression problem (LASSO) by the 'Iterated Soft Thresholding Algorithm'.

    :return: params: Fitted LASSO parameters.
    """

    # Data
    data = load_data().to_numpy()
    X = data[:, :-1]
    Y = data[:, -1].reshape((-1, 1))

    # Initial guess
    w_k = np.ones((X.shape[1], 1))

    # Algorithm parameters
    eta = 0.5
    t_k = 0.9
    lamb = 0.5
    max_iter = 100

    errors = []

    # Main loop
    for k in range(0, max_iter):

        # Gradient of Lagrangian (Shape 15x1)
        g_k = objective(X, Y, w_k, lamb)[2]

        def helper(w):
            return t_k*objective(X, Y, w, lamb)[0] + 0.5*np.linalg.norm(w - w_k + (t_k*g_k), ord=2)**2

        # Calculate next iterate by proximal operator (Shape 15x1)
        w_next = minimize(helper, x0=w_k).x.reshape(-1, 1)

        # Lasso objective function value of current iterate
        def function_value():
            val = objective(X, Y, w_next, lamb)
            return val[0] + val[1]

        # Value of piecewise linear approximation of current iterate
        def approx_value():
            val = objective(X, Y, w_next, lamb)[0]
            val += objective(X, Y, w_k, lamb)[1]
            val += objective(X, Y, w_k, lamb)[2].T@(w_next-w_k)
            val += (1/(2*t_k))*np.linalg.norm(w_next-w_k, ord=2)**2
            return val

        while function_value() > approx_value():
            t_k = eta*t_k
            w_next = minimize(helper, x0=w_next).x.reshape(-1, 1)

        w_k = w_next
        errors.append(objective(X, Y, w_k, lamb)[0])
        print(f"Iteration {k} error {errors[-1]}")
    return w_k, errors


w_star, errors = ista_regression()

# Plot error over time
plt.figure()
plt.plot(np.arange(0, 100), errors)
plt.xlabel('Time Step')
plt.ylabel('Error')
plt.show()
