"""
时空高斯过程 — 创新模块
在原空间GP基础上增加时间维度
核函数: K(x,x',t,t') = K_space(x,x') * K_time(t,t')
  K_space = Matérn 1/2 (指数核)
  K_time = 指数衰减 (近时刻权重大)
"""
import numpy as np
from scipy.linalg import cho_solve, cho_factor


def spatiotemporal_kernel(X1, X2, space_ls=0.1, time_ls=5.0, sigma_f=1.0):
    """
    时空核函数
    X1, X2: (n, 4) [x, y, z, t]
    """
    if X1.ndim == 1:
        X1 = X1.reshape(1, -1)
    if X2.ndim == 1:
        X2 = X2.reshape(1, -1)

    # 空间距离
    space_diff = X1[:, np.newaxis, :3] - X2[np.newaxis, :, :3]
    space_dist = np.sqrt(np.sum(space_diff ** 2, axis=-1))

    # 时间距离
    time_diff = X1[:, np.newaxis, 3] - X2[np.newaxis, :, 3]
    time_dist = np.abs(time_diff)

    # 时空核 = 空间核 × 时间核
    k_space = np.exp(-space_dist / space_ls)
    k_time = np.exp(-time_dist / time_ls)

    return sigma_f ** 2 * k_space * k_time


class SpatiotemporalGP:
    """时空高斯过程"""

    def __init__(self, space_ls=0.1, time_ls=5.0, sigma_f=1.0, sigma_n=0.01, max_train_points=500):
        self.space_ls = space_ls
        self.time_ls = time_ls
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.max_train_points = max_train_points

        self.X_train = np.zeros((0, 4))  # [x, y, z, t]
        self.y_train = np.zeros(0)
        self._K_inv = None
        self._alpha = None

    def _kernel(self, X1, X2):
        return spatiotemporal_kernel(X1, X2, self.space_ls, self.time_ls, self.sigma_f)

    def set_data(self, X, y):
        """X: (n, 4) [x,y,z,t], y: (n,)"""
        if len(X) == 0:
            self.X_train = np.zeros((0, 4))
            self.y_train = np.zeros(0)
        else:
            X = np.array(X)
            if X.shape[1] == 3:
                # 如果只给了空间坐标, 补时间=0
                X = np.column_stack([X, np.zeros(len(X))])
            self.X_train = X.copy()
            self.y_train = np.atleast_1d(y).copy()

            if len(self.X_train) > self.max_train_points:
                # 保留最近的点 (时间靠后的更重要)
                self.X_train = self.X_train[-self.max_train_points:]
                self.y_train = self.y_train[-self.max_train_points:]

        self._K_inv = None
        self._alpha = None

    def _precompute(self):
        n = len(self.X_train)
        if n == 0:
            return
        K = self._kernel(self.X_train, self.X_train) + self.sigma_n ** 2 * np.eye(n)
        try:
            chol = cho_factor(K + 1e-6 * np.eye(n), lower=True)
            self._K_inv = chol
            self._alpha = cho_solve(chol, self.y_train)
        except np.linalg.LinAlgError:
            K += 1e-4 * np.eye(n)
            chol = cho_factor(K, lower=True)
            self._K_inv = chol
            self._alpha = cho_solve(chol, self.y_train)

    def predict(self, X_test, current_time=0):
        """
        X_test: (n, 3) 或 (n, 4) 候选位置
        如果只给3维, 自动补当前时间
        """
        X_test = np.array(X_test)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)

        if X_test.shape[1] == 3:
            X_test = np.column_stack([X_test, np.full(len(X_test), current_time)])

        n_test = len(X_test)
        if len(self.X_train) == 0:
            return np.zeros(n_test), np.ones(n_test) * self.sigma_f ** 2

        if self._alpha is None:
            self._precompute()

        K_star = self._kernel(X_test, self.X_train)
        K_ss = self._kernel(X_test, X_test)
        mean = K_star @ self._alpha
        v = cho_solve(self._K_inv, K_star.T)
        var = np.diag(K_ss) - np.sum(K_star * v.T, axis=1)
        var = np.maximum(var, 0)
        return mean, var

    def get_trace(self, X_test, current_time=0):
        _, var = self.predict(X_test, current_time)
        return np.sum(var)

    def reset(self):
        self.X_train = np.zeros((0, 4))
        self.y_train = np.zeros(0)
        self._K_inv = None
        self._alpha = None
