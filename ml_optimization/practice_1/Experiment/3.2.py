import numpy as np
import scipy
import scipy.sparse
import matplotlib.pyplot as plt

from ml_optimization.practice_1 import optimization, oracles

def gen_matrix(num_conditional, n):
    """
    Генерирование случайно квадратной матрицы
    с числом обусловленности 'num_conditional' и размерностью 'n'

    :param num_conditional: - число обусловленности
    :param n: - размерность пространства
    :return: - scipy.sparse.diags(X)
    """
    # генерируем n - 2 числа в нужном диапазоне
    val = np.random.uniform(low = 1, high = num_conditional, size = (n - 2))

    # объединяем массивы согласно условию и получаем вектор длиной n,
    # где max(a) = num_conditional, min(a) = 1.
    val = np.concatenate([[1, num_conditional], val])
    np.random.shuffle(val)
    X = scipy.sparse.diags(val)
    return X

def gen_func(num_conditional, n, ymin = -10, ymax = 10):
    """
    Generate data for func f(w) = 1/2 <Xw, w> - <y, w>
    :param num_conditional: - обусловленность Х
    :param n: - размерность Х
    :param ymin: - минимум среди y
    :param ymax: - максимум среди y
    :return: X, y
    """
    X = gen_matrix(num_conditional, n)
    y = np.random.uniform(low = ymin, high = ymax, size = n) * num_conditional
    return X, y

def get_iterations_count(cond_num, n_nums, x_init, ymin = -10, ymax = 10, C_ = 0.001, method_name = 'Constant', debug = False):
    """
    Значение итераций по каждому числу обусловленности
    :param cond_num: обусловленность Х
    :param n_nums: размерность Х
    :param x_init:  стартовая точка
    :param ymin: минимум из у
    :param ymax: макисмум из у
    :param method_name: название метода оптимизации ('Constant','Armijo','Wolfe')
    :return: iterations - число итераций для каждого числа обусловленности
    """
    iterations = []
    for cond in cond_num:
        X, y = gen_func(cond, n_nums, ymin, ymax)
        oracle = oracles.QuadraticOracle(X, y)

        x_star, _, history = optimization.gradient_descent(oracle, x_init,
                                                      line_search_options={'method': method_name, 'c': C_},
                                                      trace = True)
        if debug:
            # print(x_star)
            print(len(history['grad_norm']))
            print("####################################################")

        iterations.append(len(history['grad_norm']))
    return iterations

def plot(cond_num, n_nums, color, ymin = -10, ymax = 10, C_ = 0.001, method_name = 'Constant', iter_count = 7):
    if method_name == 'Constant':
        plt.title(method_name + ' C = {}'.format(C_))
    else:
        plt.title(method_name)

    for i, n in enumerate(n_nums):
        x_init = np.zeros(n)

        for it in range(iter_count):
            iterations = get_iterations_count(cond_num, n, x_init,
                                              ymin= ymin,
                                              ymax = ymax,
                                              C_ = C_,
                                              method_name=method_name,
                                              debug = False)

            if it == 0:
                plt.plot(cond_num, iterations, color=color[i], label='n = {}\n'.format(n))
            else:
                plt.plot(cond_num, iterations, color=color[i])
    plt.xlabel("Conditional number")
    plt.ylabel("Iterations number")
    plt.legend()
    plt.grid()

    if method_name == 'Constant':
        plt.savefig('../Report/img/3.2/{}_{}.eps'.format(method_name, C_), format = 'eps')
    else:
        plt.savefig('../Report/img/3.2/{}.eps'.format(method_name), format='eps')
    plt.show()

if __name__ == '__main__':
    conditional_nums = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    n_nums = [10, 100, 1000, 10000]
    color = ["red", "black", "blue", "orange"]

    plot(conditional_nums, n_nums, color, method_name='Constant', C_ = 0.001)
    plot(conditional_nums, n_nums, color, method_name='Constant', C_ = 0.01)
    plot(conditional_nums, n_nums, color, method_name='Constant', C_ = 0.1)
    plot(conditional_nums, n_nums, color, method_name='Constant', C_ = 1)

    plot(conditional_nums, n_nums, color, method_name='Armijo')
    plot(conditional_nums, n_nums, color, method_name='Wolfe')

