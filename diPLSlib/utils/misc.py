'''
diPLSlib 的一些辅助函数
'''

import numpy as np
from scipy.stats import norm 
from scipy.stats import f
from math import exp, sqrt
from scipy.special import erf

def gengaus(length, mu, sigma, mag, noise=0):
    """
    生成一个具有可选随机噪声的类似高斯光谱的信号。

    参数
    ----------

    length : int
        生成的信号长度。

    mu : float
        高斯函数的均值。

    sigma : float
        高斯函数的标准差。

    mag : float
        高斯信号的幅度。

    noise : float, 可选 (默认=0)
        要添加到信号中的高斯噪声的标准差。

    返回
    -------

    signal : 形状为 (length,) 的 ndarray
        生成的带有噪声的高斯信号。

    示例
    --------

    >>> from diPLSlib.utils.misc import gengaus
    >>> import numpy as np
    >>> import scipy.stats
    >>> signal = gengaus(100, 50, 10, 5, noise=0.1)
    """

    s = mag*norm.pdf(np.arange(length),mu,sigma)
    n = noise*np.random.rand(length)
    signal = s + n

    return signal


def hellipse(X, alpha=0.05): 
    """
    计算 2D 散点图的 95% 置信区间椭圆。

    参数
    ----------

    X : 形状为 (n_samples, 2) 的 ndarray
        数据点矩阵。

    alpha : float, 可选 (默认=0.05)
        置信区间的显著性水平。

    返回
    -------

    el : 形状为 (2, 100) 的 ndarray
        椭圆点的坐标。要绘制，请使用 `plt.plot(el[0, :], el[1, :])`。

    示例
    --------

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from diPLSlib.utils.misc import hellipse
    >>> X = np.random.random((100, 2))
    >>> el = hellipse(X)
    >>> plt.scatter(X[:,0], X[:,1], label='Data points')            # doctest: +ELLIPSIS
    <matplotlib.collections.PathCollection object at ...>
    >>> plt.plot(el[0,:], el[1,:], label='95% Confidence Ellipse')  # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.legend()                                                # doctest: +ELLIPSIS
    <matplotlib.legend.Legend object at ...>
    """
    
    # 均值
    mean_all = np.zeros((2,1))   
    mean_all[0] = np.mean(X[:,0])
    mean_all[1] = np.mean(X[:,1])

    # 协方差矩阵
    X = X[:,:2]
    comat_all = np.cov(np.transpose(X))

    # SVD 分解
    U,S,V = np.linalg.svd(comat_all)

    # 置信限计算为 F 分布的 95% 分位数
    N = np.shape(X)[0]
    quant = 1 - alpha
    Conf = (2*(N-2))/(N-2)*f.ppf(quant,2,(N-2))
    
    # 在 (0, 2pi) 上评估置信区间 (CI)
    el = np.zeros((2,100))
    t = np.linspace(0,2*np.pi,100)
    for j in np.arange(100):
        sT = np.matmul(U,np.diag(np.sqrt(S*Conf)))
        el[:,j] = np.transpose(mean_all)+np.matmul(sT,np.array([np.cos(t[j]),np.sin(t[j])]))   

    return el


def rmse(y, yhat):
    """
    计算两个数组之间的均方根误差 (RMSE)。

    参数
    ----------

    y : 形状为 (n_samples,) 的 ndarray
        真实值。

    yhat : 形状为 (n_samples,) 的 ndarray
        预测值。

    返回
    -------

    error : 形状为 (n_samples,) 的 ndarray
        `y` 和 `yhat` 之间的 RMSE。

    示例
    --------

    >>> import numpy as np
    >>> from diPLSlib.utils.misc import rmse
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([2, 3, 4])
    >>> error = rmse(x, y)
    >>> print(error)
    1.0
    """

    return np.sqrt(((y.ravel()-yhat.ravel())**2).mean())


def calibrateAnalyticGaussianMechanism(epsilon, delta, GS, tol = 1.e-12):
    """ 
    使用 [Balle and Wang, ICML'18] 的解析高斯机制校准用于差分隐私的高斯扰动。

    参数
    ----------
    epsilon : float
        隐私参数 epsilon

    delta : float
        期望的隐私失败概率

    GS : float
        应用该机制的函数的 L2 敏感度上界

    tol : float
        二分查找的误差容限


    返回
    -------
    sigma : float
        在全局敏感度 GS 下实现 (epsilon, delta)-DP 所需的高斯噪声标准差

    参考文献
    ----------
    - Balle, B., & Wang, Y. X. (2018, July). Improving the gaussian mechanism for differential privacy: Analytical calibration and optimal denoising. In International Conference on Machine Learning (pp. 394-403). PMLR.

    示例
    --------
    >>> from diPLSlib.utils.misc import calibrateAnalyticGaussianMechanism
    >>> calibrateAnalyticGaussianMechanism(1.0, 1e-5, 1.0)
    3.730631634944469
    """

    def Phi(t):
        return 0.5*(1.0 + erf(float(t)/sqrt(2.0)))

    def caseA(epsilon,s):
        return Phi(sqrt(epsilon*s)) - exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def caseB(epsilon,s):
        return Phi(-sqrt(epsilon*s)) - exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def doubling_trick(predicate_stop, s_inf, s_sup):
        while(not predicate_stop(s_sup)):
            s_inf = s_sup
            s_sup = 2.0*s_inf
        return s_inf, s_sup

    def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
        s_mid = s_inf + (s_sup-s_inf)/2.0
        while(not predicate_stop(s_mid)):
            if (predicate_left(s_mid)):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup-s_inf)/2.0
        return s_mid

    delta_thr = caseA(epsilon, 0.0)

    if (delta == delta_thr):
        alpha = 1.0

    else:
        if (delta > delta_thr):
            predicate_stop_DT = lambda s : caseA(epsilon, s) >= delta
            function_s_to_delta = lambda s : caseA(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) > delta
            function_s_to_alpha = lambda s : sqrt(1.0 + s/2.0) - sqrt(s/2.0)

        else:
            predicate_stop_DT = lambda s : caseB(epsilon, s) <= delta
            function_s_to_delta = lambda s : caseB(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) < delta
            function_s_to_alpha = lambda s : sqrt(1.0 + s/2.0) + sqrt(s/2.0)

        predicate_stop_BS = lambda s : abs(function_s_to_delta(s) - delta) <= tol

        s_inf, s_sup = doubling_trick(predicate_stop_DT, 0.0, 1.0)
        s_final = binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
        alpha = function_s_to_alpha(s_final)
        
    sigma = alpha*GS/sqrt(2.0*epsilon)

    return sigma
