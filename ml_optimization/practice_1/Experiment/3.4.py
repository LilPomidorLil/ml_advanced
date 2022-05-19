import numpy as np
from matplotlib import pyplot as plt

from ml_optimization.practice_1 import optimization
from ml_optimization.practice_1 import oracles

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

from typing import List
from collections import defaultdict

def plot_func_vs_iteration(data_title: str, history_grad: defaultdict, history_nwtn: defaultdict):
    plt.title(data_title)
    it_num_grad = len(history_grad['func'])
    it_num_nwtn = len(history_nwtn['func'])
    plt.plot(np.arange(0, it_num_grad), history_grad['func'], color = 'black', linewidth = 3, label = "Gradient Descent")
    plt.plot(np.arange(0, it_num_nwtn), history_nwtn['func'], color = 'red', linewidth = 3, label = "Newton Method")
    plt.xlabel("Number iterations")
    plt.ylabel("f(w)")
    plt.legend()
    plt.grid()
    plt.savefig('../Report/img/3.4/{}_func_vs_iter.eps'.format(data_title), format='eps')
    plt.close()

def plot_func_vs_time(data_title: str, history_grad: defaultdict, histort_nwtn: defaultdict):
    plt.title(data_title)
    plt.plot(history_grad['time'], history_grad['func'], color = 'black', linewidth = 3, label = "Gradient Descent")
    plt.plot(histort_nwtn['time'], histort_nwtn['func'], color='red', linewidth=3, label="Newton Method")
    plt.xlabel("Time from start")
    plt.ylabel("f(w)")
    plt.legend()
    plt.grid()
    plt.savefig('../Report/img/3.4/{}_func_vs_time.eps'.format(data_title), format='eps')
    plt.close()

def plot_relative_square_gradient(data_title: str, history_grad: defaultdict, histort_nwtn: defaultdict, x_0, oracle):
    plt.title(data_title)
    grad_norm_k = np.array(history_grad['grad_norm']) ** 2
    grad_norm_0 = np.linalg.norm(oracle.grad(x_0)) ** 2
    grad_relative_norm = np.log(grad_norm_k / grad_norm_0 + 1e-8)
    plt.plot(history_grad['time'], grad_relative_norm,color = 'black', linewidth = 3, label = "Gradient Descent")

    nwtn_norm_k = np.array(histort_nwtn['grad_norm']) ** 2
    nwtn_norm_0 = np.linalg.norm(oracle.grad(x_0)) ** 2
    nwtn_relative_norm = np.log(nwtn_norm_k / nwtn_norm_0 + 1e-8)
    plt.plot(histort_nwtn['time'], nwtn_relative_norm, color='red', linewidth=3, label="Newton Method")

    plt.xlabel("Time from start")
    plt.ylabel("Log Relative Square Norm")
    plt.legend()
    plt.grid()
    plt.savefig('../Report/img/3.4/{}_rel_sq_grad.eps'.format(data_title), format='eps')
    plt.close()

def data_load(filename: str):
    X, y = load_svmlight_file("datasets/{0}".format(filename))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    return X_train, X_test, y_train, y_test

def main(filename: str):
    X_train, X_test, y_train, y_test = data_load(filename)
    oracle = oracles.create_log_reg_oracle(X_train, y_train, 1 / len(y_train))
    x_0 = np.zeros(X_train.toarray().shape[1])

    [_, _, history_grad] = optimization.gradient_descent(oracle,
                                                         x_0,
                                                         line_search_options={"method": "Wolfe"},
                                                         trace = True)

    [_, _, history_nwtn] = optimization.newton(oracle,
                                               x_0,
                                               line_search_options={"method": "Wolfe"},
                                               trace=True)

    filename = filename.replace(".txt", "").replace(".bz2", "")

    plot_func_vs_iteration(filename, history_grad, history_nwtn)
    plot_func_vs_time(filename, history_grad, history_nwtn)
    plot_relative_square_gradient(filename, history_grad, history_nwtn, x_0, oracle)


if __name__ == '__main__':
    main("w8a.txt")
