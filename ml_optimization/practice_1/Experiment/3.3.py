import numpy as np

from ml_optimization.practice_1 import optimization
from ml_optimization.practice_1 import oracles

from sklearn.datasets import load_svmlight_file
from matplotlib import pyplot as plt
from typing import List

SEPARETED_TEXT = "###############################################"
STOP_WORD      = "#             PROGRAM SHUT DOWN               #"
SUCCESS_WORD   = "#                SUCCESSFUL                   #"

def load_data(filename: str):
    try:
        X, y = load_svmlight_file("datasets/{0}".format(filename))
    except FileNotFoundError:
        print(f"{filename} filename not found in ./datasets/ - check input param."
              f"\n\n\n{SEPARETED_TEXT}\n{STOP_WORD}\n{SEPARETED_TEXT}\n")
        return None
    return X, y

def accuracy_log(y_true, y_pred):
    bool_list = np.array(y_true) == np.array(y_pred)
    sum_equal = bool_list.sum()
    all = len(y_true)
    return sum_equal / all

def plot_nwtn_graph(nwtn_acc: List[float], eps_list: List[float], filename: str):
    plt.title("Newton Method")
    plt.plot(eps_list, nwtn_acc, color = 'red', linewidth = 6)
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig('../Report/img/3.3/{}.eps'.format(filename.replace('.txt', '')), format='eps')

def plot(filename: str, method_name: str, eps_list: List[float]):
    try:
        X, y = load_data(filename)
    except:
        return

    acc_nwtn_list = []
    oracle = oracles.create_log_reg_oracle(X, y, 1 / len(y), oracle_type='usual')
    x_0 = np.zeros(X.toarray().shape[1])

    for eps in eps_list:
        x_star_nwtn, _, history_nwtn = optimization.newton(oracle,
                                                           x_0,
                                                           tolerance=eps,
                                                           line_search_options={"method": method_name},
                                                           trace = True,
                                                           display = False)

        y_pred_nwtn = np.sign(X @ x_star_nwtn)

        acc_nwtn_list.append(
            accuracy_log(y, y_pred_nwtn)
        )

    plot_nwtn_graph(acc_nwtn_list, eps_list, filename)


    print(SEPARETED_TEXT, SUCCESS_WORD, SEPARETED_TEXT, sep = '\n', end = '\n\n\n')

if __name__ == '__main__':
    eps = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    plot("heart.txt", 'Wolfe', eps)
    plot("heart_scale.txt", 'Wolfe', eps)
    plot("diabetes.txt", 'Wolfe', eps)
    plot("diabetes_scale.txt", 'Wolfe', eps)
    plot("breast-cancer.txt", 'Wolfe', eps)
    plot("breast-cancer.txt", 'Wolfe', eps)
