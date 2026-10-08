"""
高斯过程 — 论文 Section III-B, IV-A
Matérn 1/2 核函数
两个GP: 效用GP (建模目标分布) + 通信GP (建模其他机器人探索区域)
"""
import numpy as np
from scipy.linalg import cho_solve, cho_factor


def matern12_kernel(X1, X2, length_scale=0.1, sigma_f=1.0):
    """
    Matérn 1/2 核函数 (即指数核)
    k(x, x') = σ_f² * exp(-||x - x'|| / l)

    Matérn 1/2 对应 ν=1/2, 核函数为指数衰减
    """
    if X1.ndim == 1:
        X1 = X1.reshape(1, -1)
    if X2.ndim == 1:
        X2 = X2.reshape(1, -1)

    # 计算 pairwise 距离
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))

    return sigma_f ** 2 * np.exp(-dists / length_scale)


class GaussianProcess:
    """高斯过程回归"""

    def __init__(self, kernel_type="matern12", length_scale=0.1,
                 sigma_f=1.0, sigma_n=0.01, max_train_points=500):
        self.kernel_type = kernel_type
        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.max_train_points = max_train_points

        # 训练数据
        self.X_train = np.zeros((0, 3))
        self.y_train = np.zeros(0)

        # 预计算的核矩阵
        self._K_inv = None
        self._K_train = None
        self._alpha = None

    def _kernel(self, X1, X2):
        return matern12_kernel(X1, X2, self.length_scale, self.sigma_f)

    def add_data(self, X_new, y_new):
        """添加训练数据"""
        if isinstance(X_new, list):
            X_new = np.array(X_new)
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)

        self.X_train = np.vstack([self.X_train, X_new[:, :3]]) if len(self.X_train) > 0 else X_new[:, :3].copy()
        self.y_train = np.concatenate([self.y_train, np.atleast_1d(y_new)])

        # 限制训练点数量 (FIFO)
        if len(self.X_train) > self.max_train_points:
            self.X_train = self.X_train[-self.max_train_points:]
            self.y_train = self.y_train[-self.max_train_points:]

        self._invalidate_cache()

    def set_data(self, X, y):
        """直接设置训练数据"""
        if len(X) == 0:
            self.X_train = np.zeros((0, 3))
            self.y_train = np.zeros(0)
        else:
            if isinstance(X, list):
                X = np.array(X)
            self.X_train = X[:, :3].copy()
            self.y_train = np.atleast_1d(y).copy()
        self._invalidate_cache()

    def _invalidate_cache(self):
        self._K_inv = None
        self._alpha = None

    def _precompute(self):
        """预计算核矩阵的逆 (用于加速预测)"""
        n = len(self.X_train)
        if n == 0:
            return

        K = self._kernel(self.X_train, self.X_train) + self.sigma_n ** 2 * np.eye(n)

        try:
            chol = cho_factor(K, lower=True)
            self._K_inv = chol
            self._alpha = cho_solve(chol, self.y_train - 0)  # 均值假设为0
        except np.linalg.LinAlgError:
            # 如果矩阵不可逆, 加大噪声
            K += 1e-4 * np.eye(n)
            chol = cho_factor(K, lower=True)
            self._K_inv = chol
            self._alpha = cho_solve(chol, self.y_train)

    def predict(self, X_test):
        """
        预测均值和方差

        返回:
            mean: (n_test,) 预测均值
            var: (n_test,) 预测方差 (对角线)
        """
        if isinstance(X_test, list):
            X_test = np.array(X_test)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)

        X_test = X_test[:, :3]
        n_test = len(X_test)

        if len(self.X_train) == 0:
            # 无训练数据, 返回先验
            return np.zeros(n_test), np.ones(n_test) * self.sigma_f ** 2

        if self._alpha is None:
            self._precompute()

        # 计算核矩阵
        K_star = self._kernel(X_test, self.X_train)  # (n_test, n_train)
        K_star_star = self._kernel(X_test, X_test)   # (n_test, n_test)

        # 预测均值
        mean = K_star @ self._alpha

        # 预测方差
        v = cho_solve(self._K_inv, K_star.T)  # (n_train, n_test)
        var = np.diag(K_star_star) - np.sum(K_star * v.T, axis=1)
        var = np.maximum(var, 0)  # 确保非负

        return mean, var

    def get_prior_variance(self, X_test):
        """获取先验方差 (用于计算奖励中的探索项)"""
        if isinstance(X_test, list):
            X_test = np.array(X_test)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        X_test = X_test[:, :3]
        K_ss = self._kernel(X_test, X_test)
        return np.diag(K_ss)

    def get_trace(self, X_test):
        """获取预测协方差矩阵的迹"""
        _, var = self.predict(X_test)
        return np.sum(var)

    def reset(self):
        """重置GP"""
        self.X_train = np.zeros((0, 3))
        self.y_train = np.zeros(0)
        self._invalidate_cache()
