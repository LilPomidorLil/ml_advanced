import matplotlib.pyplot as plt

from ml_optimization.practice_1.plot_trajectory_2d import plot_levels, plot_trajectory
from ml_optimization.practice_1.oracles import QuadraticOracle
from ml_optimization.practice_1 import optimization
import numpy as np

def get_save_plot(X, y, w_init, xrange = None, yrange = None, levels = None, filename = None, show = False):
    """
    Получить результаты и сохранить в файл filename.eps
    :param xrange:
    :param yrange:
    :param levels:
    :param filename:
    :return:
    """

    oracle = QuadraticOracle(X, y)

    # делаем спуск разными методами
    _, _, history_constant = optimization.gradient_descent(oracle,
                                                           x_0 = w_init,
                                                           line_search_options={'method': 'Constant', 'c': 0.01},
                                                           trace = True)

    _, _, history_armijo = optimization.gradient_descent(oracle,
                                                           x_0=w_init,
                                                           line_search_options={'method': 'Armijo'},
                                                           trace=True)

    _, _, history_wolfe = optimization.gradient_descent(oracle,
                                                         x_0=w_init,
                                                         line_search_options={'method': 'Wolfe'},
                                                         trace=True)

    plot_levels(oracle.func, xrange = xrange, yrange = yrange, levels = levels)
    plot_trajectory(oracle.func, history_constant['x'], color = 'blue')
    plot_trajectory(oracle.func, history_armijo['x'], color = 'orange')
    plot_trajectory(oracle.func, history_wolfe['x'], color = 'red')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(["Constant, count iterations = {}".format(len(history_constant['x']) - 1),
                "Armijo, count iterations = {}".format(len(history_armijo['x']) - 1),
                "Wolfe, count iterations = {}".format(len(history_wolfe['x']) - 1)
                ])

    plt.savefig('../Report/img/3.1/{}.eps'.format(filename), format = 'eps')

    if show:
        plt.show()



if __name__ == '__main__':
    X = np.array([[1, 0],
                  [0, 50]])
    y = np.array([0, 0])
    w_init = [10, 4]
    xrange = [-6, 20]
    yrange= [-5, 5]
    levels = [0, 16, 64, 128, 256, 512]
    get_save_plot(X, y, w_init, xrange=xrange, yrange=yrange, levels=levels, filename='fig_1')

    X = np.array([[1, 0],
                  [0, 5]])
    y = np.array([0, 0])
    w_init = [10, 4]
    xrange = [-6, 20]
    yrange= [-5, 5]
    levels = [0, 16, 64, 128, 256, 512]
    get_save_plot(X, y, w_init, xrange=xrange, yrange=yrange, levels=levels, filename='fig_2')

    X = np.array([[0.5, 0],
                  [0, -0.5]])
    y = np.array([0, 0])
    w_init = [3, -3]
    xrange = [-50, 50]
    yrange= [-5, 5]
    levels = [0, 1, 2, 3, 4]
    get_save_plot(X, y, w_init, xrange=xrange, yrange=yrange, levels=levels, filename='fig_3')

    X = np.array([[0.5, 0],
                  [0, -0.5]])
    y = np.array([0, 0])
    w_init = [1, 0]
    xrange = [-5, 5]
    yrange= [-5, 5]
    levels = [0, 1, 2, 3, 4, 16, 64, 128]
    get_save_plot(X, y, w_init, xrange=xrange, yrange=yrange, levels=levels, filename='fig_4', show = True)


