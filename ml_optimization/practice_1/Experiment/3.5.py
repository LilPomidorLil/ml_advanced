import numpy as np
from matplotlib import pyplot as plt

from ml_optimization.practice_1 import optimization
from ml_optimization.practice_1 import oracles

from collections import defaultdict

def data_loader(m: int, n: int):
    np.random.seed(31415)
    A = np.random.rand(m, n)
    b = np.sign(np.random.rand(m))
    return A, b

def plot_func_vs_iter(history_usual: defaultdict, history_opt: defaultdict):
    num_iter_usual = len(history_usual['func'])
    num_iter_opt = len(history_opt['func'])

    plt.title("Function VS Iteration")
    plt.plot(np.arange(0, num_iter_usual), history_usual['func'], color = 'red', linewidth = 4, label = "Usual")
    plt.plot(np.arange(0, num_iter_opt), history_opt['func'], color = 'orange', linewidth = 3, label = "Optimized")
    plt.xlabel("Iteration number")
    plt.ylabel("f(w)")
    plt.grid()
    plt.legend()
    plt.savefig('../Report/img/3.5/func_vs_iter.eps', format='eps')
    plt.close()

def plot_func_vs_time(history_usual: defaultdict, history_opt: defaultdict):
    plt.title("Function VS Time")
    plt.plot(history_usual['time'], history_usual['func'], color = 'red', linewidth = 3, label = "Usual")
    plt.plot(history_opt['time'], history_opt['func'], color='orange', linewidth=3, label="Optimized")
    plt.xlabel("Time[sec]")
    plt.ylabel("f(w)")
    plt.grid()
    plt.legend()
    plt.savefig('../Report/img/3.5/func_vs_time.eps', format='eps')
    plt.close()

def plot_relative_square_gradient(history_usual: defaultdict, history_opt: defaultdict):
    plt.title("Relative Square Gradient Norm")
    norm_grad_0 = history_usual['grad_norm'][0] ** 2
    opt_grad_0 = history_opt['grad_norm'][0] ** 2
    norm_relative_square_gradient = np.log(np.array(history_usual['grad_norm']) ** 2 / norm_grad_0)
    opt_relative_square_gradient = np.log(np.array(history_opt['grad_norm']) ** 2 / opt_grad_0)

    plt.plot(history_usual['time'], norm_relative_square_gradient, color = 'red', linewidth = 3, label = "Usual")
    plt.plot(history_opt['time'], opt_relative_square_gradient, color='orange', linewidth=3, label="Optimized")
    plt.xlabel("Time[sec]")
    plt.ylabel("Log Relative Square Norm")
    plt.grid()
    plt.legend()
    plt.savefig('../Report/img/3.5/rel_sq_grad.eps', format='eps')
    plt.close()

def main(m: int = 10000, n: int = 8000):
    X, y = data_loader(m, n)
    x_0 = np.zeros(n)

    oracle_usual = oracles.create_log_reg_oracle(X, y, 1 / len(y), oracle_type='usual')
    oracle_opt = oracles.create_log_reg_oracle(X, y, 1 / len(y), oracle_type='optimized')

    [_, _, history_usual] = optimization.gradient_descent(
        oracle_usual,
        x_0,
        line_search_options={"method": "Wolfe", "c": 1},
        trace = True
    )

    [_, _, history_opt] = optimization.gradient_descent(
        oracle_opt,
        x_0,
        line_search_options={"method": "Wolfe", "c": 1},
        trace = True
    )
    plot_func_vs_iter(history_usual, history_opt)
    plot_func_vs_time(history_usual, history_opt)
    plot_relative_square_gradient(history_usual, history_opt)

    print("Successful")

if __name__ == '__main__':
    main()