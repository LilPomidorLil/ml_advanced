import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

from ml_optimization.practice_1 import optimization
from ml_optimization.practice_1 import oracles

from collections import defaultdict
from typing import List

def data_loader(m: int = 10000, n: int = 8000):
    np.random.seed(31415)
    A = np.random.rand(m, n)
    b = np.sign(np.random.rand(m))
    return A, b

def plot_relative_difference(history: defaultdict, color: str, label: str):
    f_opt = history['func'][-1]
    diff = np.log(np.array(history['func']) - f_opt + 1e-5)
    nums_iter = len(history['func'])

    plt.plot(np.arange(0, nums_iter), diff, color = color, label = label, linewidth = 3)

def plot_relative_square_gradient(history: defaultdict, color: str, label: str):
    grad_norm_0 = np.array(history['grad_norm'][0]) ** 2
    rel_norm = np.log(np.array(history['grad_norm']) ** 2 / grad_norm_0)
    nums_iter = len(history['grad_norm'])

    plt.plot(np.arange(0, nums_iter), rel_norm, color = color, label = label, linewidth = 3)

def constant(oracle: oracles.BaseSmoothOracle, x_init, c_values: List[float], color_list: List[str]):
    for i, c in enumerate(c_values):
        x_star, _, history = optimization.gradient_descent(oracle,
                                                           x_init,
                                                           line_search_options={"method": "Constant", "c": c},
                                                           trace = True)
        label = f"Constant, c = {c}"
        if isinstance(oracle, oracles.QuadraticOracle):
            plot_relative_difference(history, color_list[i], label)
        else:
            plot_relative_square_gradient(history, color_list[i], label)

def armijo(oracle: oracles.BaseSmoothOracle, x_init, c1_values: List[float], color_list: List[str]):
    for i, c1 in enumerate(c1_values):
        x_star, _, history = optimization.gradient_descent(oracle,
                                                           x_init,
                                                           line_search_options={"method": "Armijo", "c": c1},
                                                           trace = True)
        label = f"Armijo, c1 = {c1}"
        if isinstance(oracle, oracles.QuadraticOracle):
            plot_relative_difference(history, color_list[i], label)
        else:
            plot_relative_square_gradient(history, color_list[i], label)

def wolfe(oracle: oracles.BaseSmoothOracle, x_init, c2_values: List[float], color_list: List[str]):
    for i, c2 in enumerate(c2_values):
        x_star, _, history = optimization.gradient_descent(oracle,
                                                           x_init,
                                                           line_search_options={"method": "Wolfe", "c": c2},
                                                           trace = True)
        label = f"Wolfe, c2 = {c2}"
        if isinstance(oracle, oracles.QuadraticOracle):
            plot_relative_difference(history, color_list[i], label)
        else:
            plot_relative_square_gradient(history, color_list[i], label)

if __name__ == '__main__':
    figure(figsize=(9, 7))
    A = np.array([[1, 0],
                  [0, 50]])
    b = np.array([0, 0])
    x_init = [10, 4]
    c_values = [0.01, 0.001, 0.0001, 0.00001]
    color_constant = ['lime', 'green', 'darkgreen', 'red']

    oracle = oracles.QuadraticOracle(A, b)
    constant(oracle, x_init, c_values, color_constant)

    c1_values = [0.001, 0.0001, 0.00001, 0.000001]
    color_armijo = ['maroon', 'darkred', 'lightcoral', 'brown']
    armijo(oracle, x_init, c1_values, color_armijo)

    c2_values = [0.4, 0.6, 0.75, 0.9]
    color_wolfe = ['blue', 'midnightblue', 'cornflowerblue', 'darkcyan']
    wolfe(oracle, x_init, c2_values, color_wolfe)

    plt.xlabel("Iteration Number")
    plt.ylabel(r'$log(f(x_k) - f_{opt})$')
    plt.grid()
    plt.legend()
    plt.savefig('../Report/img/3.6/quadratic_oracle.eps', format = 'eps')
    plt.close()

    figure(figsize=(9, 7))
    m = 1000
    n = 800
    A, b = data_loader(m, n)
    x_init = np.zeros(n)
    oracle = oracles.create_log_reg_oracle(A, b, 1 / m, oracle_type='usual')
    constant(oracle, x_init, c_values, color_constant)
    armijo(oracle, x_init, c1_values, color_armijo)
    wolfe(oracle, x_init, c2_values, color_wolfe)

    plt.xlabel("Iteration Number")
    plt.ylabel('Log Relative Square Gradient Norm')
    plt.grid()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                ncol=3, fancybox=True)
    plt.savefig('../Report/img/3.6/log_reg_oracle.eps', format='eps')









