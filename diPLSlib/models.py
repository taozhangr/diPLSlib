# -*- coding: utf-8 -*-
'''
diPLSlib 模型类

- DIPLS 基类
- GCTPLS 类
- EDPLS 类
- KDAPLS 类
'''

# 模块
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_X_y, validate_data
from sklearn.utils import check_random_state
from sklearn.exceptions import NotFittedError
from scipy.sparse import issparse, sparray
import numpy as np
import matplotlib.pyplot as plt
from diPLSlib import functions as algo
from diPLSlib.utils import misc as helpers
import scipy.stats
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel

# 创建 KDAPLS 类

class KDAPLS(RegressorMixin, BaseEstimator):
    """
    用于域自适应的核域自适应偏最小二乘 (KDAPLS) 算法。

    该类通过调用 `functions.py` 中的 `kdapls` 函数实现 KDAPLS。
    KDAPLS 将源域和目标域数据投影到再生核希尔伯特空间 (RKHS) 中，
    并在该空间中对齐各域，同时在有标签数据上拟合回归模型。

    参数
    ----------
    A : int, 默认=2
        模型中使用的潜变量数量。

    l : float 或元组, 默认=0
        正则化参数。如果提供单个值，则所有潜变量应用相同的正则化。

    kernel_params : dict, 可选
        指定核类型和参数的字典。接受的键包括：
        - "type" : str, 默认="rbf"
            核类型，可以是 "rbf"、"linear" 或 "primal"。
        - "gamma" : float, 默认=0.0001
            RBF 核的核系数。

    target_domain : int, 默认=0
        指定使用哪个域的系数向量进行预测。

    属性
    ----------
    n_ : int
        `X` 中的样本数量。

    n_features_in_ : int
        `X` 中的特征数量。

    ns_ : int
        `xs` 中的样本数量。

    nt_ : int 或列表
        `xt` 中的样本数量。如果提供了多个目标域，则这是一个包含每个域样本数的列表。

    coef_ : 形状为 (n_features, 1) 的 ndarray
        用于预测的回归系数向量。

    X_ : 形状为 (n_, n_features_in_) 的 ndarray
        用于拟合模型的训练数据。

    xs_ : 形状为 (ns_, n_features_in_) 的 ndarray
        用于拟合模型的（无标签）源域数据。

    xt_ : 形状为 (nt_, n_features_in_) 的 ndarray
        用于拟合模型的（无标签）目标域数据。

    y_mean_ : float
        训练响应变量的均值。

    centering_ : dict
        存储核操作所需的中心化信息的字典。

    is_fitted_ : bool
        模型是否已拟合。

    示例
    --------
    >>> import numpy as np
    >>> from diPLSlib.models import KDAPLS
    >>> x = np.random.rand(100, 10)
    >>> y = np.random.rand(100)
    >>> xs = np.random.rand(80, 10)
    >>> xt = np.random.rand(50, 10)
    >>> model = KDAPLS(A=2, l=0.5, kernel_params={"type": "rbf", "gamma": 0.001})
    >>> model.fit(x, y, xs, xt)
    KDAPLS(kernel_params={'gamma': 0.001, 'type': 'rbf'}, l=0.5)
    >>> xtest = np.random.rand(5, 10)
    >>> yhat = model.predict(xtest)

    参考文献
    ----------
    1. Huang, G., Chen, X., Li, L., Chen, X., Yuan, L., & Shi, W. (2020). Domain adaptive partial least squares regression. 
       Chemometrics and Intelligent Laboratory Systems, 201, 103986.
    2. B. Schölkopf, A. Smola, and K. Müller. Nonlinear component analysis as a kernel eigenvalue problem. 
       Neural computation, 10(5):1299-1319, 1998.
    """

    def __init__(self, A=2, l=0, kernel_params=None, target_domain=0):
        self.A = A
        self.l = l
        self.kernel_params = kernel_params
        self.target_domain = target_domain

    def fit(self, X, y, xs=None, xt=None, **kwargs):
        """
        拟合 KDAPLS 模型。

        参数
        ----------
        X : np.ndarray
            有标签的源域数据（通常与 xs 相同）。
        y : np.ndarray
            与 X 对应的标签。
        xs : np.ndarray
            源域数据。
        xt : np.ndarray
            目标域数据。
        **kwargs : dict, 可选
            传递给模型的其他关键字参数（例如，用于模型选择目的）。

        返回
        -------
        self : object
            已拟合的评估器。
        """

        # 设置核参数
        if self.kernel_params is None:
            
            kernel_params = {"type": "primal"}

        else:

            kernel_params = self.kernel_params.copy()

        
        # 检查稀疏 (sparse) 输入
        if issparse(X):

            raise ValueError("不支持稀疏 (sparse) 输入。请将数据转换为密集格式。")
 
        # 验证输入数组
        X, y = validate_data(self, X, y, ensure_2d=True, allow_nd=False, accept_large_sparse=False, accept_sparse=False, ensure_all_finite=True)
        

        # 检查是否提供了源域和目标域数据
        if xs is None:

            xs = X

        if xt is None:

            xt = X

        # 验证源域和目标域数组
        xs = check_array(xs, ensure_2d=True, allow_nd=False, accept_large_sparse=False, accept_sparse=False, ensure_all_finite=True)
        xs = np.atleast_2d(xs) if xs is not None else X
        if isinstance(xt, list):
            xt = [check_array(x, ensure_2d=True, allow_nd=False, accept_large_sparse=False, accept_sparse=False, ensure_all_finite=True) for x in xt]
        else:
            xt = check_array(xt, ensure_2d=True, allow_nd=False, accept_large_sparse=False, accept_sparse=False, ensure_all_finite=True)
        xt = [np.atleast_2d(x) for x in xt] if isinstance(xt, list) else np.atleast_2d(xt) if xt is not None else X

        # 检查是否提供了至少两个样本和特征
        if X.shape[0] < 2:
            raise ValueError("拟合模型至少需要两个样本（当前 n_samples = {}）。".format(X.shape[0]))
        
        if X.shape[1] < 2:
            raise ValueError("KDAPLS 拟合模型至少需要 2 个特征（当前 n_features = {}）。".format(X.shape[1]))


        # 确保 y 是二维的
        if y.ndim == 1:
            y = y.reshape(-1, 1)


        # 检查复数数据
        if np.iscomplexobj(X) or np.iscomplexobj(y) or np.iscomplexobj(xs) or np.iscomplexobj(xt):
            
            raise ValueError("不支持复数数据")

        # 准备工作
        self.n_, self.n_features_in_ = X.shape
        self.ns_, _ = xs.shape
        if isinstance(xt, list):
            self.nt_ = [x.shape[0] for x in xt]
        else:
            self.nt_, _ = xt.shape
        
        self.y_ = y
        self.xs_ = xs
        self.xt_ = xt


        b, bst, T, Tst, W, P, Pst, E, Est, Ey, C, centering = algo.kdapls(
            X, y, xs, xt,
            A=self.A,
            l=self.l,
            kernel_params=kernel_params
        )

        # 根据 target_domain 选择系数向量
        if self.target_domain == 0:
            self.coef_ = b
        else:
            self.coef_ = bst

        self.centering_ = centering[self.target_domain]
        self.X_ = X
        self.y_mean_ = centering[0]["y_mean_"] if 0 in centering else 0.0
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        使用 KDAPLS 模型进行预测。

        参数
        ----------

        X : 形状为 (n_samples, n_features) 的 ndarray
            执行预测的测试数据矩阵。

        返回
        -------

        yhat : 形状为 (n_samples_test,) 的 ndarray
            测试数据的预测响应值。

        """
        # 检查模型是否已拟合
        check_is_fitted = getattr(self, "is_fitted_", False)
        if not check_is_fitted:
            raise NotFittedError("KDAPLS 对象尚未拟合。")
        
        # 检查稀疏输入
        if issparse(X):
            raise ValueError("不支持稀疏输入。请将数据转换为密集格式。")

        # 验证输入数组
        X = validate_data(self, X, reset=False, ensure_2d=True, allow_nd=False, ensure_all_finite=True)

        Kt_c = self._x_centering(X)
        yhat = Kt_c @ self.coef_ + self.centering_["y_mean_"]

        # 确保 yhat 的形状与 y 匹配
        yhat = np.ravel(yhat)

        return yhat 

    def _x_centering(self, X):
        """
        使用存储的 centering_ 对新数据 X 进行中心化。

        参数
        ----------

        X : 形状为 (n_samples, n_features) 的 ndarray
            执行预测的测试数据矩阵。

        返回
        -------

        Kt : ndarray 
            中心化后的测试数据矩阵。Kt 的形状取决于核类型：
            - 对于 'rbf' 和 'linear'，Kt 是 X 和 X_ 之间的核矩阵。
            - 对于 'primal'，Kt 是中心化后的测试数据矩阵。

        """
    
        n = self.X_.shape[0]
        Kt = None

        # 检查 X 的特征数量是否与 X_ 相同
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError(
                f"测试数据中的特征数量 ({X.shape[1]}) 与 "
                f"训练数据中的特征数量 ({self.X_.shape[1]}) 不匹配。"
            )

        if self.kernel_params is not None:

            if self.kernel_params["type"] == "rbf":
                gamma_ = self.kernel_params["gamma"]
                Kt = rbf_kernel(X, self.X_, gamma=gamma_)

            elif self.kernel_params["type"] == "linear":
                Kt = linear_kernel(X, self.X_)

            elif self.kernel_params["type"] == "primal":
                Kt = X.copy()

            else:
                raise ValueError("核类型无效。支持的类型有 'rbf'、'linear' 和 'primal'。")

            if self.kernel_params["type"] == "primal":
                return Kt - self.centering_["K"].mean(axis=0)
        
            else:

                J = (1 / n) * np.ones((n, n))
                Jt = (1 / self.centering_["n"]) * (np.ones((X.shape[0], 1)) @ np.ones((1, self.centering_["n"])))
                return Kt - Kt @ J - Jt @ self.centering_["K"] + Jt @ self.centering_["K"] @ J
        
        else: # 使用原始 (primal) da-PLS

            Kt = X.copy()
            mean_vec = self.centering_["K"].mean(axis=0)
        
            return Kt - mean_vec


class DIPLS(RegressorMixin, BaseEstimator):
    """
    用于域自适应的域不变偏最小二乘 (DIPLS) 算法。

    该类实现了 DIPLS 算法，该算法旨在预测目标变量 `y` 的同时对齐不同域之间的特征分布。
    它通过域特定的特征转换支持多个源域和目标域。

    参数
    ----------
    A : int, 默认=2
        模型中使用的潜变量数量。

    l : float 或长度为 A 的元组, 默认=0
        正则化参数。如果提供单个值，则所有潜变量应用相同的正则化。

    centering : bool, 默认=True
        如果为 True，则源域和目标域数据将进行均值中心化。

    heuristic : bool, 默认=False
        如果为 True，则正则化参数设置为启发式值，以平衡拟合输出变量 y 和最小化域差异。

    target_domain : int, 默认=0
        如果传递了多个目标域，target_domain 指定模型应应用于哪个目标域。
        如果 target_domain=0，模型应用于源域；
        如果 target_domain=1，模型应用于第一个目标域，依此类推。

    rescale : str 或 ndarray, 默认='Target'
        确定测试数据的重缩放方式。如果是 'Target' 或 'Source'，测试数据将分别按 xt 或 xs 的均值进行重缩放。
        如果提供 ndarray，则测试数据将按提供数组的均值进行重缩放。

    属性
    ----------
    n_ : int
        `X` 中的样本数量。

    ns_ : int
        `xs` 中的样本数量。

    nt_ : int
        `xt` 中的样本数量。

    n_features_in_ : int
        `X` 中的特征数量。

    mu_ : 形状为 (n_features,) 的 ndarray
        `X` 中各列的均值。

    mu_s_ : 形状为 (n_features,) 的 ndarray
        `xs` 中各列的均值。

    mu_t_ : 形状为 (n_features,) 的 ndarray 或 ndarray 列表
        `xt` 中各列的均值。如果存在多个域，则按目标域计算均值。

    b_ : 形状为 (n_features, 1) 的 ndarray
        回归系数向量。

    b0_ : float
        回归模型的截距。

    T_ : 形状为 (n_samples, A) 的 ndarray
        训练数据投影（得分）。

    Ts_ : 形状为 (n_source_samples, A) 的 ndarray
        源域投影（得分）。

    Tt_ : 形状为 (n_target_samples, A) 的 ndarray 或 ndarray 列表
        目标域投影（得分）。

    W_ : 形状为 (n_features, A) 的 ndarray
        权重矩阵。

    P_ : 形状为 (n_features, A) 的 ndarray
        与 X 对应的载荷矩阵。

    Ps_ : 形状为 (n_features, A) 的 ndarray
        与 xs 对应的载荷矩阵。

    Pt_ : 形状为 (n_features, A) 的 ndarray 或 ndarray 列表
        与 xt 对应的载荷矩阵。

    E_ : ndarray
        训练数据的残差。

    Es_ : ndarray
        源域残差矩阵。

    Et_ : ndarray 或 ndarray 列表
        目标域残差矩阵。

    Ey_ : ndarray
        源域中响应变量的残差。

    C_ : 形状为 (A, 1) 的 ndarray
        将源域投影与响应变量关联起来的回归向量。

    opt_l_ : 形状为 (A,) 的 ndarray
        为每个潜变量启发式确定的正则化参数。

    discrepancy_ : 形状为 (A,) 的 ndarray
        源域和目标域投影之间的方差差异。

    is_fitted_ : bool
        模型是否已拟合。

    参考文献
    ----------
    1. Ramin Nikzad-Langerodi et al., "Domain-Invariant Partial Least Squares Regression", Analytical Chemistry, 2018.
    2. Ramin Nikzad-Langerodi et al., "Domain-Invariant Regression under Beer-Lambert's Law", Proc. ICMLA, 2019.
    3. Ramin Nikzad-Langerodi et al., "Domain adaptation for regression under Beer–Lambert’s law", Knowledge-Based Systems, 2020.
    4. B. Mikulasek et al., "Partial least squares regression with multiple domains", Journal of Chemometrics, 2023.

    示例
    --------
    >>> import numpy as np
    >>> from diPLSlib.models import DIPLS
    >>> x = np.random.rand(100, 10)
    >>> y = np.random.rand(100)
    >>> xs = np.random.rand(100, 10)
    >>> xt = np.random.rand(50, 10)
    >>> model = DIPLS(A=5, l=10)
    >>> model.fit(x, y, xs, xt)
    DIPLS(A=5, l=10)
    >>> xtest = np.array([5, 7, 4, 3, 2, 1, 6, 8, 9, 10]).reshape(1, -1)
    >>> yhat = model.predict(xtest)
    """

    def __init__(self, A=2, l=0, centering=True, heuristic=False, target_domain=0, rescale='Target'):
        # 模型参数
        self.A = A
        self.l = l
        self.centering = centering
        self.heuristic = heuristic
        self.target_domain = target_domain
        self.rescale = rescale
        



    def fit(self, X, y, xs=None, xt=None, **kwargs):
        """
        拟合 DIPLS 模型。

        该方法使用提供的源域和目标域数据拟合域不变偏最小二乘 (di-PLS) 模型。
        它可以处理单个和多个目标域。

        参数
        ----------
        X : 形状为 (n_samples, n_features) 的 ndarray
            来自源域的有标签输入数据。

        y : 形状为 (n_samples, 1) 的 ndarray
            与输入数据 `x` 对应的响应变量。

        xs : 形状为 (n_samples_source, n_features) 的 ndarray
            源域 X 数据。如果未提供，则默认使用 `X`。

        xt : Union[形状为 (n_samples_target, n_features) 的 ndarray, List[ndarray]]
            目标域 X 数据。可以是单个目标域，也可以是代表多个目标域的数组列表。
            如果未提供，则默认使用 `X`。

        **kwargs : dict, 可选
            传递给模型的其他关键字参数（例如，用于模型选择目的）。


        返回
        -------
        self : object
            已拟合的模型实例。
        """
        
        # 检查稀疏 (sparse) 输入
        if issparse(X):

            raise ValueError("不支持稀疏 (sparse) 输入。请将数据转换为密集格式。")
 
        # 验证输入数组
        X, y = validate_data(self, X, y, ensure_2d=True, allow_nd=False, accept_large_sparse=False, accept_sparse=False, ensure_all_finite=True)
        

        # 检查是否提供了源域和目标域数据
        if xs is None:

            xs = X

        if xt is None:

            xt = X

        # 验证源域和目标域数组
        xs = check_array(xs, ensure_2d=True, allow_nd=False, accept_large_sparse=False, accept_sparse=False, ensure_all_finite=True)
        xs = np.atleast_2d(xs) if xs is not None else X
        if isinstance(xt, list):
            xt = [check_array(x, ensure_2d=True, allow_nd=False, accept_large_sparse=False, accept_sparse=False, ensure_all_finite=True) for x in xt]
        else:
            xt = check_array(xt, ensure_2d=True, allow_nd=False, accept_large_sparse=False, accept_sparse=False, ensure_all_finite=True)
        xt = [np.atleast_2d(x) for x in xt] if isinstance(xt, list) else np.atleast_2d(xt) if xt is not None else X

        # 将 y 展平为一维数组
        y = np.ravel(y)

        # 检查是否提供了至少两个样本
        if X.shape[0] < 2:
            raise ValueError("拟合模型至少需要两个样本（当前 n_samples = {}）。".format(X.shape[0]))

        # 检查复数数据
        if np.iscomplexobj(X) or np.iscomplexobj(y) or np.iscomplexobj(xs) or np.iscomplexobj(xt):
            
            raise ValueError("不支持复数数据")
        
        
        # 再次检查是否提供了源域和目标域数据（为了逻辑严密性）
        if xs is None:

            xs = X

        if xt is None:

            xt = X
        
        
        # 准备工作
        self.n_, self.n_features_in_ = X.shape
        self.ns_, _ = xs.shape        
        
        self.x_ = X
        self.y_ = y
        self.xs_ = xs
        self.xt_ = xt
        self.b0_ = np.mean(self.y_)

        # 均值中心化
        if self.centering:

            self.mu_ = np.mean(self.x_, axis=0)
            self.mu_s_ = np.mean(self.xs_, axis=0)
            self.x_ = self.x_ - self.mu_
            self.xs_ = self.xs_ - self.mu_s_
            y = self.y_ - self.b0_

            # 多个目标域
            if isinstance(self.xt_, list):
                
                self.nt_, _ = xt[0].shape
                self.mu_t_ = [np.mean(x, axis=0) for x in self.xt_]
                self.xt_ = [x - mu for x, mu in zip(self.xt_, self.mu_t_)]
            
            else:

                self.nt_, _ = xt.shape
                self.mu_t_ = np.mean(self.xt_, axis=0)
                self.xt_ = self.xt_ - self.mu_t_

        else:

            y = self.y_
        

        x = self.x_ 
        xs = self.xs_
        xt = self.xt_

    
        # 拟合模型
        results = algo.dipals(x, y.reshape(-1,1), xs, xt, self.A, self.l, heuristic=self.heuristic, target_domain=self.target_domain)
        self.b_, self.T_, self.Ts_, self.Tt_, self.W_, self.P_, self.Ps_, self.Pt_, self.E_, self.Es_, self.Et_, self.Ey_, self.C_, self.opt_l_, self.discrepancy_ = results
        
        self.is_fitted_ = True        
        return self

            
    def predict(self, X):
        """
        使用已拟合的 DIPLS 模型预测 y。

        该方法使用拟合好的域不变偏最小二乘 (di-PLS) 模型预测测试数据的响应变量。

        参数
        ----------

        X : 形状为 (n_samples, n_features) 的 ndarray
            执行预测的测试数据矩阵。

        返回
        -------

        yhat : 形状为 (n_samples_test,) 的 ndarray
            测试数据的预测响应值。

        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise NotFittedError("此 DIPLS 实例尚未拟合。在使用此评估器之前，请使用适当的参数调用 'fit'。")
        
        
        # 检查稀疏输入
        if issparse(X):
            raise ValueError("不支持稀疏输入。请将数据转换为密集格式。")

        # 验证输入数组
        X = validate_data(self, X, reset=False, ensure_2d=True, allow_nd=False, ensure_all_finite=True)
        
        # 重缩放测试数据
        if(type(self.rescale) is str):

            if(self.rescale == 'Target'):

                if(type(self.xt_) is list):

                    if(self.target_domain==0):

                        Xtest = X[...,:] - self.mu_s_

                    else:

                        Xtest = X[...,:] - self.mu_t_[self.target_domain-1]

                else:

                    Xtest = X[...,:] - self.mu_t_

            elif(self.rescale == 'Source'):

                Xtest = X[...,:] - self.mu_

            elif(self.rescale == 'none'):

                Xtest = X

        elif(type(self.rescale) is np.ndarray):

             Xtest = X[...,:] - np.mean(self.rescale,0)

        else: 

            raise Exception('rescale 必须是 Source、Target 或数据集')
            
        
        yhat = Xtest@self.b_ + self.b0_

        # 确保 yhat 的形状与 y 匹配
        yhat = np.ravel(yhat)

        return yhat



# 为 GCT-PLS 模型创建独立类，继承自基类模型
class GCTPLS(DIPLS):
    """
    基于图的校正转移偏最小二乘 (Graph-based Calibration Transfer Partial Least Squares, GCT-PLS)。

    该方法在拟合响应的同时，最小化潜变量空间中源域 (xs) 和目标域 (xt) 数据对之间的距离。

    参数
    ----------
    A : int, 默认=2
        模型中使用的潜变量数量。

    l : float 或长度为 A 的元组, 默认=0
        正则化参数。如果提供单个值，则所有潜变量应用相同的正则化。

    centering : bool, 默认=True
        如果为 True，源域和目标域数据在拟合前将进行均值中心化。

    heuristic : bool, 默认=False
        如果为 True，正则化参数设置为启发式值，旨在平衡响应变量 y 的模型拟合质量并最小化域表示之间的差异。

    rescale : str 或 ndarray, 默认='Target'
        确定测试数据的重缩放方式。如果是 'Target' 或 'Source'，测试数据将分别按 xt 或 xs 的均值进行重缩放。
        如果提供 ndarray，则测试数据将按提供数组的均值进行重缩放。

    属性
    ----------
    n_ : int
        `X` 中的样本数量。

    ns_ : int
        `xs` 中的样本数量。

    nt_ : int
        `xt` 中的样本数量。

    n_features_in_ : int
        `X` 中的特征数量。

    mu_ : 形状为 (n_features,) 的 ndarray
        `X` 中各列的均值。

    mu_s_ : 形状为 (n_features,) 的 ndarray
        `xs` 中各列的均值。

    mu_t_ : 形状为 (n_features,) 的 ndarray
        `xt` 中各列的均值。

    b_ : 形状为 (n_features, 1) 的 ndarray
        回归系数向量。

    b0_ : float
        回归模型的截距。

    T_ : 形状为 (n_samples, A) 的 ndarray
        训练数据投影（得分）。

    Ts_ : 形状为 (n_source_samples, A) 的 ndarray
        源域投影（得分）。

    Tt_ : 形状为 (n_target_samples, A) 的 ndarray
        目标域投影（得分）。

    W_ : 形状为 (n_features, A) 的 ndarray
        权重矩阵。

    P_ : 形状为 (n_features, A) 的 ndarray
        与 X 对应的载荷矩阵。

    Ps_ : 形状为 (n_features, A) 的 ndarray
        与 xs 对应的载荷矩阵。

    Pt_ : 形状为 (n_features, A) 的 ndarray
        与 xt 对应的载荷矩阵。

    E_ : 形状为 (n_source_samples, n_features) 的 ndarray
        源域数据的残差。

    Es_ : 形状为 (n_source_samples, n_features) 的 ndarray
        源域残差矩阵。

    Et_ : 形状为 (n_target_samples, n_features) 的 ndarray
        目标域残差矩阵。

    Ey_ : 形状为 (n_source_samples, 1) 的 ndarray
        源域中响应变量的残差。

    C_ : 形状为 (A, 1) 的 ndarray
        将源域投影与响应变量关联起来的回归向量。

    opt_l_ : 形状为 (A,) 的 ndarray
        为每个潜变量启发式确定的正则化参数。

    discrepancy_ : ndarray
        源域和目标域投影之间的方差差异。

    is_fitted_ : bool
        模型是否已拟合。

    参考文献
    ----------
    Nikzad‐Langerodi, R., & Sobieczky, F. (2021). Graph‐based calibration transfer. 
    Journal of Chemometrics, 35(4), e3319.

    示例
    --------
    >>> import numpy as np
    >>> from diPLSlib.models import GCTPLS
    >>> x = np.random.rand(100, 10)
    >>> y = np.random.rand(100)
    >>> xs = np.random.rand(80, 10)
    >>> xt = np.random.rand(80, 10)
    >>> model = GCTPLS(A=3, l=(2, 5, 7))
    >>> model.fit(x, y, xs, xt)
    GCTPLS(A=3, l=(2, 5, 7))
    >>> xtest = np.array([5, 7, 4, 3, 2, 1, 6, 8, 9, 10]).reshape(1, -1)
    >>> yhat = model.predict(xtest)
    """

    def __init__(self, A=2, l=0, centering=True, heuristic=False, rescale='Target'):
        # 模型参数
        self.A = A
        self.l = l
        self.centering = centering
        self.heuristic = heuristic
        self.rescale = rescale

        
    def fit(self, X, y, xs=None, xt=None, **kwargs):
        """
        拟合 GCT-PLS 模型。

        参数
        ----------

        x : 形状为 (n_samples, n_features) 的 ndarray
            来自源域的有标签输入数据。

        y : 形状为 (n_samples, 1) 的 ndarray
            与输入数据 `x` 对应的响应变量。

        xs : 形状为 (n_sample_pairs, n_features) 的 ndarray
            源域 X 数据。如果未提供，则默认使用 `X`。

        xt : 形状为 (n_sample_pairs, n_features) 的 ndarray
            目标域 X 数据。如果未提供，则默认使用 `X`。

        **kwargs : dict, 可选
            传递给模型的其他关键字参数（例如，用于模型选择目的）。
 

        返回
        -------

        self : object
            已拟合的模型实例。
        """
        # 检查稀疏 (sparse) 输入
        if issparse(X):

            raise ValueError("不支持稀疏 (sparse) 输入。请将数据转换为密集格式。")

        # 验证输入数组
        X, y = check_X_y(X, y, ensure_2d=True, allow_nd=False, accept_large_sparse=False, accept_sparse=False, ensure_all_finite=True)
        
        # 检查是否提供了源域和目标域数据
        if xs is None:

            xs = X

        if xt is None:

            xt = X

        # 验证源域和目标域数组
        xs = check_array(xs, ensure_2d=True, allow_nd=False, accept_large_sparse=False, accept_sparse=False, ensure_all_finite=True)
        xs = np.atleast_2d(xs) if xs is not None else X
        if isinstance(xt, list):
            xt = [check_array(x, ensure_2d=True, allow_nd=False, accept_large_sparse=False, accept_sparse=False, ensure_all_finite=True) for x in xt]
        else:
            xt = check_array(xt, ensure_2d=True, allow_nd=False, accept_large_sparse=False, accept_sparse=False, ensure_all_finite=True)
        xt = [np.atleast_2d(x) for x in xt] if isinstance(xt, list) else np.atleast_2d(xt) if xt is not None else X

        # 将 y 展平为一维数组
        y = np.ravel(y)

        # 检查是否提供了至少两个样本
        if X.shape[0] < 2:
            raise ValueError("拟合模型至少需要两个样本（当前 n_samples = {}）。".format(X.shape[0]))

        # 检查复数数据
        if np.iscomplexobj(X) or np.iscomplexobj(y) or np.iscomplexobj(xs) or np.iscomplexobj(xt):
            
            raise ValueError("不支持复数数据")
        
        
        # 再次检查是否提供了源域和目标域数据
        if xs is None:

            xs = X

        if xt is None:

            xt = X
        

        # 准备工作
        self.n_, self.n_features_in_ = X.shape
        self.ns_, _ = xs.shape        
        self.nt_, _ = xt.shape

        if self.ns_ != self.nt_:
            raise ValueError("源域样本数 (ns) 必须等于目标域样本数 (nt)。")
        
        self.x_ = X
        self.y_ = y
        self.xs_ = xs
        self.xt_ = xt
        self.b0_ = np.mean(self.y_)
        self.mu_ = np.mean(self.x_, axis=0)
        self.mu_s_ = np.mean(self.xs_, axis=0)
        self.mu_t_ = np.mean(self.xt_, axis=0)

        # 均值中心化
        if self.centering is True:
            
            x = self.x_[...,:] - self.mu_
            y = self.y_ - self.b0_

        else: 
            
            x = self.x_
            y = self.y_

        xs = self.xs_
        xt = self.xt_
            
        # 拟合模型并存储矩阵
        results = algo.dipals(x, y.reshape(-1,1), xs, xt, self.A, self.l, heuristic=self.heuristic, laplacian=True)
        self.b_, self.T_, self.Ts_, self.Tt_, self.W_, self.P_, self.Ps_, self.Pt_, self.E_, self.Es_, self.Et_, self.Ey_, self.C_, self.opt_l_, self.discrepancy_ = results

        self.is_fitted_ = True  # 将 is_fitted 属性设置为 True
        return self


class EDPLS(DIPLS):
    r'''
    (\epsilon, \delta)-差分隐私偏最小二乘回归。

    该类实现了由 Nikzad-Langerodi 等人（2024, 未发表）提出的 (\epsilon, \delta)-差分隐私偏最小二乘 (PLS) 回归方法。

    参数
    ----------
    A : int, 默认=2
        潜变量数量。

    epsilon : float, 默认=1.0
        隐私损失参数。

    delta : float, 默认=0.05
        失败概率。

    centering : bool, 默认=True
        如果为 True，在拟合模型前将对数据进行中心化。

    random_state : int, RandomState 实例或 None, 默认=None
        控制为差分隐私添加的噪声的随机性。

    属性
    ----------
    n_ : int
        训练数据中的样本数量。

    n_features_in_ : int
        训练数据中的特征数量。

    x_mean_ : 形状为 (n_features,) 的 ndarray
        每个特征的估计均值。

    coef_ : 形状为 (n_features, 1) 的 ndarray
        估计的回归系数。

    y_mean_ : float
        估计的截距。

    x_scores_ : 形状为 (n_samples, A) 的 ndarray
        X 得分。

    x_loadings_ : 形状为 (n_features, A) 的 ndarray
        X 载荷。

    x_weights_ : 形状为 (n_features, A) 的 ndarray
        X 权重。

    y_loadings_ : 形状为 (A, 1) 的 ndarray
        Y 载荷。

    x_residuals_ : 形状为 (n_samples, n_features) 的 ndarray
        X 残差。

    y_residuals_ : 形状为 (n_samples, 1) 的 ndarray
        Y 残差。

    is_fitted_ : bool
        如果模型已拟合，则为 True。

    参考文献
    ----------
    - R. Nikzad-Langerodi, et al. (2024). (epsilon,delta)-Differentially private partial least squares regression (unpublished).
    - Balle, B., & Wang, Y. X. (2018, July). Improving the Gaussian mechanism for differential privacy: Analytical calibration and optimal denoising. In International Conference on Machine Learning (pp. 394-403). PMLR.

    示例
    --------
    >>> from diPLSlib.models import EDPLS
    >>> import numpy as np
    >>> x = np.random.rand(100, 10)
    >>> y = np.random.rand(100)
    >>> model = EDPLS(A=5, epsilon=0.1, delta=0.01)
    >>> model.fit(x, y)
    EDPLS(A=5, delta=0.01, epsilon=0.1)
    >>> xtest = np.array([5, 7, 4, 3, 2, 1, 6, 8, 9, 10]).reshape(1, -1)
    >>> yhat = model.predict(xtest)
    '''

    def __init__(self, A:int=2, epsilon:float=1.0, delta:float=0.05, centering:bool=True, random_state=None):
        # 模型参数
        self.A = A
        self.epsilon = epsilon
        self.delta = delta
        self.centering = centering
        self.random_state = random_state


    def fit(self, X:np.ndarray, y:np.ndarray, **kwargs):
        '''
        拟合 EDPLS 模型。

        参数
        ----------
        X : 数组，形状为 (n_samples, n_features)
            训练数据。

        y : 数组，形状为 (n_samples,)
            目标值。

        **kwargs : dict, 可选
            传递给模型的其他关键字参数（例如，用于模型选择目的）。

        返回
        -------

        self : object
           已拟合的模型实例。

        '''

        ### 验证输入数据
        # 检查稀疏 (sparse) 输入
        if issparse(X):

            raise ValueError("不支持稀疏 (sparse) 输入。请将数据转换为密集格式。")
 
        # 验证输入数组
        X, y = validate_data(self, X, y, ensure_2d=True, allow_nd=False, accept_large_sparse=False, accept_sparse=False, ensure_all_finite=True)
         
        # 将 y 展平为一维数组
        y = np.ravel(y)

        # 检查是否提供了至少两个样本
        if X.shape[0] < 2:
            raise ValueError("拟合模型至少需要两个样本（当前 n_samples = {}）。".format(X.shape[0]))

        # 检查复数项
        if np.iscomplexobj(X) or np.iscomplexobj(y):
            
            raise ValueError("不支持复数数据")
        
        
        ### 准备工作
        self.n_, self.n_features_in_ = X.shape
        self.x_ = X
        self.y_ = y
        self.y_mean_= np.mean(self.y_)

        # 均值中心化
        if self.centering:

            self.x_mean_ = np.mean(self.x_, axis=0)
            self.x_ = self.x_ - self.x_mean_
            y = self.y_ - self.y_mean_

        else:

            y = self.y_


        x = self.x_ 

        ### 拟合模型
        rng = check_random_state(self.random_state)
        results = algo.edpls(x, y.reshape(-1,1), self.A, epsilon=self.epsilon, delta=self.delta, rng=rng)
        self.coef_, self.x_weights_, self.x_loadings_, self.y_loadings_, self.x_scores_, self.x_residuals_, self.y_residuals_  = results

        self.is_fitted_ = True 

        return self
    
    
    def predict(self, x:np.ndarray):
        """
        使用已拟合的 EDPLS 模型预测 y。

        参数
        ----------

        x: 形状为 (n_samples_test, n_features) 的 numpy 数组
            执行预测的测试数据矩阵。

        返回
        -------

        yhat: 形状为 (n_samples_test, ) 的 numpy 数组
            测试数据的预测响应值。


        """
        
        # 检查模型是否已拟合
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise NotFittedError("此 DIPLS 实例尚未拟合。在使用此评估器之前，请使用适当的参数调用 'fit'。")
        
        
        # 检查稀疏 (sparse) 输入
        if issparse(x):
            raise ValueError("不支持稀疏 (sparse) 输入。请将数据转换为密集格式。")

        # 验证输入数组
        x = validate_data(self, x, reset=False, ensure_2d=True, allow_nd=False, ensure_all_finite=True)


        # 中心化并缩放 x
        if self.centering is True:
            x = x[...,:] - self.x_mean_

        # 预测 y
        yhat = x@self.coef_ + self.y_mean_

        # 确保 yhat 的形状与 y 匹配
        yhat = np.ravel(yhat)


        return yhat
    
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.regressor_tags.poor_score = True
        return tags

    def _more_tags(self):
        '''
        返回评估器的标签。
        '''
        return {"poor_score": True}
