import numpy as np

from ml_optimization.practice_1 import optimization
from ml_optimization.practice_1 import oracles

from sklearn.datasets import load_svmlight_file
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    return X_train, X_test, y_train, y_test

def plot_nwtn_graph(nwtn_acc: List[float], nwtn_recall: List[float], nwtn_precision: List[float], nwtn_f1: List[float], eps_list: List[float], filename: str):
    plt.title("Newton Method")
    plt.plot(eps_list, nwtn_acc, color = 'red', linewidth = 2, label = '- accuracy')
    plt.plot(eps_list, nwtn_recall, color = 'black', linewidth = 2, label = "- recall")
    plt.plot(eps_list, nwtn_precision, color = 'blue', linewidth = 2, label = "- precision")
    plt.plot(eps_list, nwtn_f1, color='orange', linewidth=2, label="- f1-score")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig('../Report/img/3.3/{}.eps'.format(filename.replace('.txt', '')), format='eps')
    plt.close()

def plot(filename: str, method_name: str, eps_list: List[float]):
    try:
        X_train, X_test, y_train, y_test = load_data(filename)
    except:
        return

    y_test = np.where(y_test == -1, 0, y_test)
    acc_nwtn_list = []
    rec_nwtn_list = []
    pre_nwtn_list = []
    f1_nwth_list  = []
    oracle = oracles.create_log_reg_oracle(X_train, y_train, 1 / len(y_train), oracle_type='usual')
    x_0 = np.zeros(X_train.toarray().shape[1])

    for eps in eps_list:
        x_star_nwtn, _, history_nwtn = optimization.newton(oracle,
                                                           x_0,
                                                           tolerance=eps,
                                                           line_search_options={"method": method_name},
                                                           trace = True,
                                                           display = False)

        y_pred_nwtn = np.sign(X_test @ x_star_nwtn)
        y_pred_nwtn = np.where(y_pred_nwtn == -1, 0, y_pred_nwtn)

        acc_nwtn_list.append(
            accuracy_score(y_test, y_pred_nwtn)
        )
        rec_nwtn_list.append(
            recall_score(y_test, y_pred_nwtn)
        )
        pre_nwtn_list.append(
            precision_score(y_test, y_pred_nwtn)
        )
        f1_nwth_list.append(
            f1_score(y_test, y_pred_nwtn)
        )

    plot_nwtn_graph(acc_nwtn_list, rec_nwtn_list, pre_nwtn_list, f1_nwth_list, eps_list, filename)


    print(SEPARETED_TEXT, SUCCESS_WORD, SEPARETED_TEXT, sep = '\n', end = '\n\n\n')

if __name__ == '__main__':
    eps = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    plot("heart.txt", 'Wolfe', eps)
    plot("heart_scale.txt", 'Wolfe', eps)
    plot("diabetes.txt", 'Wolfe', eps)
    plot("diabetes_scale.txt", 'Wolfe', eps)
