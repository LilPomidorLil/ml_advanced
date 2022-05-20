import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))

    def hess_vec(self, x, v):
        """
        Computes matrix-vector product with Hessian matrix f''(x) v
        """
        return self.hess(x).dot(v)


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.
    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()
    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATy : function of y
            Computes matrix-vector product A^Ty, where y is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b                                 # label
        self.regcoef = regcoef

    def func(self, w):
        """
        Подсчет функционала ошибки
        :param w: вектор весов
        :return: значение функционала
        """
        m = len(self.b)
        in_log = - self.b * self.matvec_Ax(w)
        loss = np.logaddexp(0, in_log)
        return (np.ones(m) @ loss) / m + (self.regcoef / 2) * np.linalg.norm(w) ** 2

    def grad(self, w):
        m = len(self.b)
        sigmoid_and_label = scipy.special.expit(self.matvec_Ax(w)) - (self.b + 1) / 2
        res = self.matvec_ATx(sigmoid_and_label) / m
        return res + self.regcoef * w

    def hess(self, w):
        m = len(self.b)
        n = len(w)
        sigmoid_arg = self.matvec_Ax(w)
        sigmoid_der = scipy.special.expit(sigmoid_arg) * (1 - scipy.special.expit(sigmoid_arg))
        res = self.matmat_ATsA(sigmoid_der)
        return res / m + self.regcoef * np.eye(n)


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).
    For explanation see LogRegL2Oracle.
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
        self.x = None
        self.d = None
        self.xhat = None                # x_hat = x + alpha * d
        self.A_xhat = None              # A_xhat = Ax + alpha * Ad
        self.Ad = None
        self.Ax = None
        self.ATx = None


    def update_Ax(self, x):
        if np.all(x == self.x):
            return

        self.x = x
        self.Ax = self.matvec_Ax(x)

    def update_Ad(self, d):
        if np.all(d == self.d):
            return

        self.d = d
        self.Ad = self.matvec_Ax(d)

    def update_xhat(self, x, alpha, d):
        if np.all(self.xhat == x + alpha * d):
            return

        self.xhat = x + alpha * d
        self.A_xhat = self.Ax + alpha * self.Ad

    def func(self, x):
        m = len(self.b)

        # last point in task
        if np.all(self.xhat == x):
            in_log = - self.b * self.A_xhat
            loss = np.logaddexp(0, in_log)
            return (np.ones(m) @ loss) / m + (self.regcoef / 2) * np.linalg.norm(x) ** 2

        self.update_Ax(x)
        in_log = - self.b * self.Ax
        loss = np.logaddexp(0, in_log)
        return (np.ones(m) @ loss) / m + (self.regcoef / 2) * np.linalg.norm(x) ** 2

    def grad(self, x):
        m = len(self.b)

        if np.all(self.xhat == x):
            sigmoid_and_label = scipy.special.expit(self.A_xhat) - (self.b + 1) / 2
            res = self.matvec_ATx(sigmoid_and_label) / m
            return res + self.regcoef * x

        self.update_Ax(x)
        sigmoid_and_label = scipy.special.expit(self.Ax) - (self.b + 1) / 2
        res = self.matvec_ATx(sigmoid_and_label) / m
        return res + self.regcoef * x

    def hess(self, x):
        m = len(self.b)
        n = len(x)

        if np.all(self.xhat == x):
            sigmoid_der = scipy.special.expit(self.A_xhat) * (1 - scipy.special.expit(self.A_xhat))
            res = self.matmat_ATsA(sigmoid_der)
            return res / m + self.regcoef * np.eye(n)

        self.update_Ax(x)
        sigmoid_der = scipy.special.expit(self.Ax) * (1 - scipy.special.expit(self.Ax))
        res = self.matmat_ATsA(sigmoid_der)
        return res / m + self.regcoef * np.eye(n)


    def func_directional(self, x, d, alpha):
        # TODO: Implement optimized version with pre-computation of Ax and Ad
        return None

    def grad_directional(self, x, d, alpha):
        # TODO: Implement optimized version with pre-computation of Ax and Ad
        return None


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    def matvec_Ax(x):
        return A @ x

    def matvec_ATx(x):
        return A.T @ x

    def matmat_ATsA(s):
        return A.T @ scipy.sparse.diags(s) @ A

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    # TODO: Implement numerical estimation of the gradient
    n = len(x)

    res = np.zeros(n)

    for i in range(n):
        e = np.zeros(n)
        e[i] = 1
        res[i] = (func(x + eps * e) - func(x)) / eps
    return res


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i)
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    n = len(x)
    res = np.zeros((n, n))

    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1
        for j in range(n):
            ej = np.zeros(n)
            ej[j] = 1
            res[i][j] = (func(x + eps * ei + eps * ej) - func(x + eps * ei) - func(x + eps * ej) + func(x)) / eps**2
    return res