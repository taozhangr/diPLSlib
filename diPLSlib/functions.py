# -*- coding: utf-8 -*-

# 模块
import numpy as np
import scipy.linalg
import scipy.stats
from scipy.linalg import eigh
import scipy.spatial.distance as scd
from scipy.spatial import distance_matrix
from sklearn.metrics import pairwise as kernels
import warnings
warnings.filterwarnings("ignore")
from sklearn.utils.validation import check_array
from diPLSlib.utils.misc import calibrateAnalyticGaussianMechanism


def kdapls(x: np.ndarray, y: np.ndarray, xs: np.ndarray, xt, 
           A: int, 
           l, 
           kernel_params: dict = {"type": "rbf", "gamma": 10}):
    r'''
    执行核域自适应偏最小二乘 (Kernel Domain Adaptive Partial Least Squares, kda-PLS) 回归。

    该方法利用有标签的源域数据和潜在的无标签目标域数据拟合核偏最小二乘回归模型。
    与 di-PLS 不同，kda-PLS 以非参数方式对齐再生核希尔伯特空间 (RKHS) 中的源分布和目标分布，
    因此不对底层数据分布做任何假设。

    数学上，对于每个潜变量 (LV)，kda-PLS 寻找一个权重向量 :math:`\mathbf{w}`（满足 :math:`\mathbf{w}^T\mathbf{w} = 1`），
    使得以下目标函数最大化：

    .. math::
        \max_{\mathbf{w} : \mathbf{w}^T\mathbf{w} = 1} \Biggl(
        \mathbf{w}^T K(X_s, X_s)^T Y Y^T K(X_s, X_s) \mathbf{w}
        - \gamma \mathbf{w}^T K(X_{st}, X_s)^T H L H K(X_{st}, X_s) \mathbf{w}
        \Biggr),

    其中：

    - :math:`K(X_s, X_s)` 是根据源域数据计算的核矩阵，
    - :math:`K(X_{st}, X_s)` 是合并的源/目标数据 :math:`X_{st} = [X_s; X_t]` 与源域数据之间计算的核矩阵，
    - :math:`Y` 是响应变量，
    - :math:`H` 表示中心化矩阵，
    - :math:`L` 是拉普拉斯矩阵，定义为：如果 :math:`X_{st}` 中的第 i 个和第 j 个样本属于同一域，则 :math:`L_{ij}=1`，否则为 0，
    - :math:`\gamma` 是正则化参数，用于平衡最大化 :math:`K(X_s, X_s)` 与 :math:`Y` 之间的协方差以及最小化域差异。

    参数
    ----------
    x : 形状为 (n_samples, n_features) 的 ndarray
        有标签的源域数据。

    y : 形状为 (n_samples, 1) 的 ndarray
        与源域相关的响应变量。

    xs : 形状为 (n_source_samples, n_features) 的 ndarray
        源域特征数据。

    xt : 形状为 (n_target_samples, n_features) 的 ndarray 或 ndarray 列表
        目标域特征数据。多个域可以作为列表提供。

    A : int
        模型中使用的潜变量数量。

    l : float 或长度为 A 的元组
        正则化参数。如果提供单个值，则所有潜变量应用相同的正则化。

    kernel_params : dict, 默认={"type": "rbf", "gamma": 10}
        核参数。字典必须包含以下键：
        - "type": str, 默认="rbf"
            使用的核类型。支持的类型有 "rbf"、"linear" 和 "primal"。
        - "gamma": float, 默认=10
            RBF 核的核系数。

    返回
    -------
    b : 形状为 (n_features, 1) 的 ndarray
        回归系数向量。

    bst : 形状为 (n_features, 1) 的 ndarray
        目标域的回归系数向量。

    T : 形状为 (n_samples, A) 的 ndarray
        训练数据投影（得分）。

    Tst : 形状为 (n_source_samples + n_target_samples, A) 的 ndarray
        源域和目标域投影（得分）。

    W : 形状为 (m, A) 的 ndarray
        权重矩阵。

    P : 形状为 (m, A) 的 ndarray
        源域的载荷矩阵。

    Pst : 形状为 (m, A) 的 ndarray
        源域和目标域的载荷矩阵。

    E : ndarray
        源域的残差。

    Est : ndarray
        源域和目标域的残差。

    Ey : ndarray
        响应变量的残差。

    C : 形状为 (A, q) 的 ndarray
        将投影与响应变量关联起来的回归向量。

    centering : dict
        包含中心化信息的字典。

    参考文献
    ----------
    1. Huang, G., Chen, X., Li, L., Chen, X., Yuan, L., & Shi, W. (2020). Domain adaptive partial least squares regression. Chemometrics and Intelligent Laboratory Systems, 201, 103986.

    示例
    --------
    >>> import numpy as np
    >>> from diPLSlib.functions import kdapls
    >>> x = np.random.random((100, 10))
    >>> y = np.random.random((100, 1))
    >>> xs = np.random.random((50, 10))
    >>> xt = np.random.random((50, 10))
    >>> b, bst, T, Tst, W, P, Pst, E, Est, Ey, C, centering = kdapls(x, y, xs, xt, 2, 0.5)
    '''
    # 输入验证
    x = check_array(x, dtype=np.float64)
    xs = check_array(xs, dtype=np.float64)
    if isinstance(xt, list):
        xt = [check_array(xti, dtype=np.float64) for xti in xt]
    else:
        xt = check_array(xt, dtype=np.float64)
    y = check_array(y, dtype=np.float64)
    
    # 获取数组维度并初始化矩阵
    (ns, k) = np.shape(xs)
    (n, k) = np.shape(x)
    #(nt, k) = np.shape(xt)
    #xst = np.vstack((xs,xt)) 


    if isinstance(xt, list):
        nt_list = [np.shape(xti)[0] for xti in xt]
        nt = sum(nt_list)
        xst = np.vstack([xs] + xt)
    else:
        (nt, k) = np.shape(xt)
        xst = np.vstack((xs, xt))
    
    Y = y.copy()
    if Y.ndim == 1:        
        Y = Y.reshape(-1,1).copy()   
        
    q = Y.shape[1]
    
    if kernel_params["type"] == "primal":
        m = k
    else:
        m = n

    W = np.zeros([m, A])
    T = np.zeros([n, A])
    Tst = np.zeros([ns+nt, A])
    P = np.zeros([m, A])
    Pst = np.zeros([m, A])
    C = np.zeros([A, q])    
    
    # 拉普拉斯矩阵       
    J = (1/n)*np.ones((n,n))
    H = np.eye(n) - J
    Jst = (1/(ns+nt))*np.ones((ns+nt,ns+nt))
    Hst = np.eye(ns+nt) - Jst
    #L1 = np.ones((ns+nt,1))
    #L1[ns:,0] = -1
    #L = L1@L1.T
    L = np.zeros((ns + nt, ns + nt))
    L[:ns, :ns] = 1
    if isinstance(xt, list):
        start_idx = ns
        for nti in nt_list:
            L[start_idx:start_idx+nti, start_idx:start_idx+nti] = 1
            start_idx += nti
    else:
        L[ns:, ns:] = 1

    # 计算核矩阵
    if kernel_params["type"] == "rbf":
    
        gamma = kernel_params["gamma"]
        K = kernels.rbf_kernel(x, x, gamma = gamma)
        Kst = kernels.rbf_kernel(xst, x, gamma = gamma)
        
    elif kernel_params["type"] == "linear":
        
        K = x@x.T
        Kst = xst@x.T
        
    elif kernel_params["type"] == "primal":
        
        K = x.copy()
        Kst = xst.copy()

    # 存储中心化元素
    centering = {}
    y_mean_ = Y.mean(axis=0)
    # 源域
    centering[0] = {}
    centering[0]["n"] = n
    centering[0]["K"] = K
    centering[0]["y_mean_"] = y_mean_
    
    # 源-目标域
    centering[1] = {}
    centering[1]["n"] = ns+nt    
    centering[1]["K"] = Kst   
    centering[1]["y_mean_"] = y_mean_

    
    # 中心化
    if kernel_params["type"] == "primal":
        K = H@K
        Kst = Hst@Kst
    else:
        K = H@K@H
        Kst = Kst - Kst@J - Jst@Kst + Jst@Kst@J 

    Y = H@Y   
        
    # 计算潜变量 (LVs)
    for i in range(A):
        
        
        if isinstance(l, tuple) and len(l) == A:       # 为每个潜变量设置独立的正则化参数

            lA = l[i]

        elif isinstance(l, (float, int, np.int64)):    # 所有潜变量使用相同的正则化参数

            lA = l

        else:

            raise ValueError("正则化参数必须是单个值或 A 元组。")
        
        
        # 计算域不变权重向量
        wM = (K.T@Y@Y.T@K) - lA*(Kst.T@L@Kst)
        wd , wm = eigh(wM)         
        w = wm[:,-1]              
        w.shape = (w.shape[0],1)
        
        # 计算得分并归一化
        t = K@w           
        tst = Kst@w
        t = t / np.linalg.norm(t)
        tst = tst / np.linalg.norm(tst)
        
        # 计算载荷        
        p = K.T@t
        pst = Kst.T@tst
        
        # 对 t 进行 y 的回归
        c = t.T@Y

        # 存储 w, t, p, c
        W[:, i] = w.reshape(m)        
        T[:, i] = t.reshape(n)        
        Tst[:, i] = tst.reshape(ns+nt)
        P[:, i] = p.reshape(m)        
        Pst[:, i] = pst.reshape(m)
        C[i] = c.reshape(q)        

        # 矩阵紧缩 (Deflation)
        K = K - t@p.T
        Kst = Kst - tst@pst.T 
        
        Y = Y - (t@c)


    # 计算回归向量
    b = W@(np.linalg.inv(P.T@W))@C
    bst = W@(np.linalg.inv(Pst.T@W))@C

    # 残差    
    E = K    
    Est = Kst
    Ey = Y
    

    return b, bst, T, Tst, W, P, Pst, E, Est, Ey, C, centering


def dipals(x, y, xs, xt, A, l, heuristic: bool = False, target_domain=0, laplacian: bool = False):
    """
    执行（多）域不变偏最小二乘 (Domain-Invariant Partial Least Squares, di-PLS) 回归。

    该方法利用有标签的源域数据和潜在的跨多个域的无标签目标域数据拟合 PLS 回归模型，
    旨在构建一个在不同域之间具有良好泛化能力的模型。

    参数
    ----------
    x : 形状为 (n_samples, n_features) 的 ndarray
        有标签的源域数据。

    y : 形状为 (n_samples, 1) 的 ndarray
        与源域相关的响应变量。

    xs : 形状为 (n_source_samples, n_features) 的 ndarray
        源域特征数据。

    xt : 形状为 (n_target_samples, n_features) 的 ndarray 或 ndarray 列表
        目标域特征数据。多个域可以作为列表提供。

    A : int
        模型中使用的潜变量数量。

    l : float 或长度为 A 的元组
        正则化参数。如果提供单个值，则所有潜变量应用相同的正则化。

    heuristic : bool, 默认=False
        如果为 True，则自动确定正则化参数，以平衡拟合 Y 和最小化域差异。

    target_domain : int, 默认=0
        指定模型应应用于哪个目标域，其中 0 表示源域。

    laplacian : bool, 默认=False
        如果为 True，则使用拉普拉斯矩阵来正则化潜变量空间中匹配的校正转移样本之间的距离。

    返回
    -------
    b : 形状为 (n_features, 1) 的 ndarray
        回归系数向量。

    T : 形状为 (n_samples, A) 的 ndarray
        训练数据投影（得分）。

    Ts : 形状为 (n_source_samples, A) 的 ndarray
        源域投影（得分）。

    Tt : 形状为 (n_target_samples, A) 的 ndarray 或 ndarray 列表
        目标域投影（得分）。

    W : 形状为 (n_features, A) 的 ndarray
        权重矩阵。

    P : 形状为 (n_features, A) 的 ndarray
        与 x 对应的载荷矩阵。

    Ps : 形状为 (n_features, A) 的 ndarray
        与 xs 对应的载荷矩阵。

    Pt : 形状为 (n_features, A) 的 ndarray 或 ndarray 列表
        与 xt 对应的载荷矩阵。

    E : ndarray
        训练数据的残差。

    Es : ndarray
        源域残差矩阵。

    Et : ndarray 或 ndarray 列表
        目标域残差矩阵。

    Ey : ndarray
        源域中响应变量的残差。

    C : 形状为 (A, 1) 的 ndarray
        将源域投影与响应变量关联起来的回归向量。

    opt_l : 形状为 (A,) 的 ndarray
        为每个潜变量启发式确定的正则化参数。

    discrepancy : 形状为 (A,) 的 ndarray
        源域和目标域投影之间的方差差异。

    参考文献
    ----------
    1. Ramin Nikzad-Langerodi et al., "Domain-Invariant Partial Least Squares Regression", Analytical Chemistry, 2018.
    2. Ramin Nikzad-Langerodi et al., "Domain-Invariant Regression under Beer-Lambert's Law", Proc. ICMLA, 2019.
    3. Ramin Nikzad-Langerodi et al., "Domain adaptation for regression under Beer–Lambert’s law", Knowledge-Based Systems, 2020.
    4. B. Mikulasek et al., "Partial least squares regression with multiple domains", Journal of Chemometrics, 2023.

    示例
    --------
    >>> import numpy as np
    >>> from diPLSlib.functions import dipals
    >>> x = np.random.random((100, 10))
    >>> y = np.random.random((100, 1))
    >>> xs = np.random.random((50, 10))
    >>> xt = np.random.random((50, 10))
    >>> b, T, Ts, Tt, W, P, Ps, Pt, E, Es, Et, Ey, C, opt_l, discrepancy = dipals(x, y, xs, xt, 2, 0.1)
    """
    # 获取数组维度
    (n, k) = np.shape(x)
    (ns, k) = np.shape(xs)

    
    # 初始化矩阵
    Xt = xt

    if(type(xt) is list):
        Pt = []
        Tt = []


        for z in range(len(xt)):

                Tti = np.zeros([np.shape(xt[z])[0], A])
                Pti = np.zeros([k, A])

                Pt.append(Pti)
                Tt.append(Tti)


    else:

        (nt, k) = np.shape(xt)
        Tt = np.zeros([nt, A])
        Pt = np.zeros([k, A])


    T = np.zeros([n, A])
    P = np.zeros([k, A])
    Ts = np.zeros([ns, A])
    Ps = np.zeros([k, A])
    W = np.zeros([k, A])
    C = np.zeros([A, 1])
    opt_l = np.zeros(A)
    discrepancy = np.zeros(A)
    I = np.eye(k)

    # 计算潜变量
    for i in range(A):

        if isinstance(l, tuple) and len(l) == A:       # 为每个潜变量设置独立的正则化参数

            lA = l[i]

        elif isinstance(l, (float, int, np.int64)):              # 所有潜变量使用相同的正则化参数

            lA = l

        else:

            raise ValueError("正则化参数必须是单个值或 A 元组。")


        # 计算域不变权重向量
        w_pls = ((y.T@x)/(y.T@y))  # 普通 PLS 解
       


        if(lA != 0 or heuristic is True):  # 如果使用了正则化

                if(type(xt) is not list):

                    # 协方差差异矩阵的凸松弛
                    D = convex_relaxation(xs, xt)

                # 多个目标域
                elif(type(xt) is list):
                    
                    #print('正在松弛域 ... ')
                    ndoms = len(xt)
                    D = np.zeros([k, k])

                    for z in range(ndoms):

                        d = convex_relaxation(xs, xt[z])
                        D = D + d

                elif(laplacian is True):
                
                    J = np.vstack([xs, xt])
                    L = transfer_laplacian(xs, xt)
                    D = J.T@L@J


                else:

                    print('xt 必须是矩阵或（适当维度的）矩阵列表')

                if(heuristic is True): # 正则化参数启发式确定
                    w_pls = w_pls/np.linalg.norm(w_pls)
                    gamma = (np.linalg.norm((x-y@w_pls))**2)/(w_pls@D@w_pls.T)
                    opt_l[i] = gamma
                    lA = gamma


                reg = I+lA/((y.T@y))*D
                w = scipy.linalg.solve(reg.T, w_pls.T, assume_a='sym').T  # 比之前的 reg 计算快 10 倍

                # 归一化 w
                w = w/np.linalg.norm(w)

                # 源域和目标域投影方差之间的绝对差异
                discrepancy[i] = (w @ D @ w.T).item()


        else:        

            if(type(xt) is list):

                D = convex_relaxation(xs, xt[0])

            else:

                D = convex_relaxation(xs, xt)

            
            w = w_pls/np.linalg.norm(w_pls)
            discrepancy[i] = (w @ D @ w.T).item()

    
        # 计算得分
        t = x@w.T
        ts = xs@w.T
        
        if(type(xt) is list):

            tt = []

            for z in range(len(xt)):

                tti = xt[z]@w.T
                tt.append(tti)

        else:

            tt = xt@w.T


        # 对 t 进行 y 的回归
        c = (y.T@t)/(t.T@t)

        # 计算载荷
        p = (t.T@x)/(t.T@t)
        ps = (ts.T@xs)/(ts.T@ts)
        if(type(xt) is list):

            pt = []

            for z in range(len(xt)):

                pti = (tt[z].T@xt[z])/(tt[z].T@tt[z])
                pt.append(pti)

        else:

            pt = (tt.T@xt)/(tt.T@tt)


        # 紧缩 X 和 y (施密特正交化)
        x = x - t@p

        if laplacian is False:                       # 仪器校正 (Calibration transfer) 情况
            xs = xs - ts@ps
        
        if(type(xt) is list):

            for z in range(len(xt)):

                xt[z] = xt[z] - tt[z]@pt[z]

        else:

            if(np.sum(xt) != 0):  # 仅当目标矩阵不为零时进行紧缩

                if laplacian is False:                       # 仪器校正 (Calibration transfer) 情况
                    xt = xt - tt@pt


        y = y - t*c

        # 存储 w, t, p, c
        W[:, i] = w
        T[:, i] = t.reshape(n)
        Ts[:, i] = ts.reshape(ns)
        P[:, i] = p.reshape(k)
        Ps[:, i] = ps.reshape(k)
        C[i] = c       

        if(type(xt) is list):

            for z in range(len(xt)):

                Pt[z][:, i] = pt[z].reshape(k)
                Tt[z][:, i] = tt[z].reshape(np.shape(xt[z])[0])

        else:
            
            Pt[:, i] = pt.reshape(k)
            Tt[:, i] = tt.reshape(nt)         


    # 计算回归向量
    if laplacian is True:                       # 仪器校正 (Calibration transfer) 情况

        b = W@(np.linalg.inv(P.T@W))@C

    else:

        if isinstance(l, tuple):                # 检查是否传递了多个正则化参数（每个 LV 一个）

            if target_domain==0:                # 多个目标域（域未知）

                b = W@(np.linalg.inv(P.T@W))@C

            elif type(xt) is np.ndarray:        # 单个目标域

                b = W@(np.linalg.inv(Pt.T@W))@C

            elif type(xt) is list:              # 多个目标域（域已知）

                b = W@(np.linalg.inv(Pt[target_domain-1].T@W))@C

        else:

                b = W@(np.linalg.inv(P.T@W))@C   


    # 存储残差
    E = x
    Es = xs
    Et = xt
    Ey = y

    return b, T, Ts, Tt, W, P, Ps, Pt, E, Es, Et, Ey, C, opt_l, discrepancy


def convex_relaxation(xs, xt):
    """
    执行协方差差异矩阵的凸松弛。

    该松弛涉及计算对称协方差差异矩阵的特征值分解，反转负特征值的符号，并重建矩阵。
    这对应于源域和目标域之间协方差差异的上界。

    参数
    ----------
    xs : 形状为 (n_source_samples, n_features) 的 ndarray
        来自源域的特征数据。

    xt : 形状为 (n_target_samples, n_features) 的 ndarray
        来自目标域的特征数据。

    返回
    -------
    D : 形状为 (n_features, n_features) 的 ndarray
        松弛后的协方差差异矩阵。

    参考文献
    ----------
    Ramin Nikzad-Langerodi et al., "Domain-Invariant Regression under Beer-Lambert's Law", Proc. ICMLA, 2019.

    示例
    --------
    >>> import numpy as np
    >>> from diPLSlib.functions import convex_relaxation
    >>> xs = np.random.random((100, 10))
    >>> xt = np.random.random((100, 10))
    >>> D = convex_relaxation(xs, xt)
    """
    # 确保输入数组是数值型的
    xs = np.asarray(xs, dtype=np.float64)
    xt = np.asarray(xt, dtype=np.float64)
    
    # 检查 NaN 或无穷大值
    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(xt)):
        raise ValueError("输入数组不得包含 NaN 或无穷大值。")

    # 检查复数数据
    if np.iscomplexobj(xs) or np.iscomplexobj(xt):
        raise ValueError("不支持复数数据。")
    
    # 准备工作
    ns = np.shape(xs)[0]
    nt = np.shape(xt)[0]
    x = np.vstack([xs, xt])
    x = x[..., :] - np.mean(x, 0)
    
    # 计算源域和目标域协方差矩阵之间的差异   
    rot = (1/ns*xs.T@xs- 1/nt*xt.T@xt) 

    # 凸松弛
    w,v = eigh(rot)
    eigs = np.abs(w)
    eigs = np.diag(eigs)
    D = v@eigs@v.T 

    return D
                    

def transfer_laplacian(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    为仪器校正 (Calibration transfer) 问题构建拉普拉斯矩阵。

    参数
    ----------
    x : 形状为 (n_samples, n_features) 的 ndarray
        来自设备 1 的数据样本。

    y : 形状为 (n_samples, n_features) 的 ndarray
        来自设备 2 的数据样本。

    返回
    -------
    L : 形状为 (2 * n_samples, 2 * n_samples) 的 ndarray
        仪器校正问题的拉普拉斯矩阵。

    参考文献
    ----------
    Nikzad‐Langerodi, R., & Sobieczky, F. (2021). Graph‐based calibration transfer. 
    Journal of Chemometrics, 35(4), e3319.

    示例
    --------
    >>> import numpy as np
    >>> from diPLSlib.functions import transfer_laplacian
    >>> x = np.array([[1, 2], [3, 4]])
    >>> y = np.array([[2, 3], [4, 5]])
    >>> L = transfer_laplacian(x, y)
    >>> print(L)
    [[ 1.  0. -1. -0.]
     [ 0.  1. -0. -1.]
     [-1. -0.  1.  0.]
     [-0. -1.  0.  1.]]
    """
    (n, p) = np.shape(x)
    I = np.eye(n)
    L = np.vstack([np.hstack([I,-I]),np.hstack([-I,I])])

    return L


def edpls(x: np.ndarray, y: np.ndarray, n_components: int, epsilon: float, delta: float = 0.05, rng=None):
    r'''
    (\epsilon, \delta)-差分隐私偏最小二乘回归。

    根据 Balle & Wang (2018) 的高斯机制，用于从 PLS1 算法中私密发布权重 :math:`\mathbf{W}`、得分 :math:`\mathbf{T}` 
    以及 :math:`X/Y` 载荷 :math:`\mathbf{P}`/:math:`\mathbf{c}`。对于每个潜变量，添加来自 :math:`\mathcal{N}(0,\sigma^2)` 
    的独立同分布噪声，其方差满足：

    .. math::
        \Phi\left( \frac{\Delta}{2\sigma} - \frac{\epsilon\sigma}{\Delta} \right) - e^{\epsilon} \Phi\left( -\frac{\Delta}{2\sigma} - \frac{\epsilon\sigma}{\Delta} \right)\leq \delta,

    其中 :math:`\Phi(t) = \mathrm{P}[\mathcal{N}(0,1)\leq t]`（即标准单变量高斯分布的 CDF），噪声被添加到权重、得分和载荷中，
    而发布相应量的函数的敏感度 :math:`\Delta(\cdot)` 计算如下：

    .. math::
        \Delta(w) = \sup_{(\mathbf{x}, y)} |y| \|\mathbf{x}\|_2

    .. math::
        \Delta(t) \leq \sup_{\mathbf{x}}  \|\mathbf{x}\|_2

    .. math::
        \Delta(p) \leq \sup_{\mathbf{x}}  \|\mathbf{x}\|_2

    .. math::
        \Delta(c) \leq \sup_{y}  |y|.

    请注意，与 Dwork 等人 (2006) 和 Dwork 等人 (2014) 提出的高斯机制相比，Balle & Wang (2018) 的机制保证了对于任何 
    :math:`\epsilon > 0` 的值（而不仅仅是 :math:`\epsilon \leq 1`）都具有 :math:`(\epsilon, \delta)`-差分隐私。

    参数
    ----------
    x : 形状为 (n_samples, n_features) 的 ndarray
        输入数据。

    y : 形状为 (n_samples, n_targets) 的 ndarray
        目标值。

    n_components : int
        潜变量的数量。

    epsilon : float
        隐私损失参数。

    delta : float, 默认=0.05
        失败概率。

    rng : numpy.random.Generator, 可选
        随机数生成器。

    返回
    -------
    coef_ : 形状为 (n_features, n_targets) 的 ndarray
        回归系数。

    x_weights_ : 形状为 (n_features, n_components) 的 ndarray
        X 权重。

    x_loadings_ : 形状为 (n_features, n_components) 的 ndarray
        X 载荷。

    y_loadings_ : 形状为 (n_components, n_targets) 的 ndarray
        Y 载荷。

    x_scores_ : 形状为 (n_samples, n_components) 的 ndarray
        X 得分。

    x_residuals_ : 形状为 (n_samples, n_features) 的 ndarray
        X 残差。

    y_residuals_ : 形状为 (n_samples, n_targets) 的 ndarray
        Y 残差。

    参考文献
    ----------
    - R. Nikzad-Langerodi, et al. (2024). (epsilon,delta)-Differentially private partial least squares regression (unpublished).
    - Balle, B., & Wang, Y. X. (2018, July). Improving the Gaussian mechanism for differential privacy: Analytical calibration and optimal denoising. In International Conference on Machine Learning (pp. 394-403). PMLR.

    示例
    --------
    >>> from diPLSlib.functions import edpls
    >>> import numpy as np
    >>> x = np.random.rand(100, 10)
    >>> y = np.random.rand(100, 1)
    >>> coef_, x_weights_, x_loadings_, y_loadings_, x_scores_, x_residuals_, y_residuals_ = edpls(x, y, 2, epsilon=0.1, delta=0.05)
    '''
    # 输入验证
    x = check_array(x, dtype=np.float64)
    y = check_array(y, dtype=np.float64)

    # 获取数组维度
    (n_, n_features_) = np.shape(x)
    I = np.eye(n_features_)

    # 权重
    x_weights_ = np.zeros([n_features_, n_components])

    # X 得分
    x_scores_ = np.zeros([n_, n_components])

    # X 载荷
    x_loadings_ = np.zeros([n_features_, n_components])

    # Y 载荷
    y_loadings_ = np.zeros([n_components, 1])

    # 迭代组件数量
    for i in range(n_components):

        # 计算权重 w
        w_pls = ((y.T@x)/(y.T@y))  

        # 归一化 w（无噪声）
        wo = w_pls / np.linalg.norm(w_pls)

        # 计算 x 得分并归一化（无噪声）
        to = x @ wo.T
        to = to / np.linalg.norm(to)

        # 计算 x 得分并归一化（添加噪声前）
        t = x @ wo.T
        t = to / np.linalg.norm(to)

        # 向 w 添加噪声
        x_min = x.min(axis=0)
        x_max = x.max(axis=0)
        y_min = y.min(axis=0)
        y_max = y.max(axis=0)
        x_norm = np.linalg.norm(x_max - x_min)
        y_norm = y_max - y_min
        x_max_norm = np.linalg.norm(x, axis=1).max()
        
        sensitivity = x_max_norm*y_max
        R = calibrateAnalyticGaussianMechanism(epsilon, delta, sensitivity)**2
        
        if rng is None:
            v = np.random.normal(0, R, n_features_)

        else:
            v = rng.normal(0, R, n_features_)

        w = wo + v

        # 归一化 w（添加噪声后）
        w = w / np.linalg.norm(w)

        # 向 t 添加噪声
        sensitivity = x_max_norm
        R = calibrateAnalyticGaussianMechanism(epsilon, delta, sensitivity)**2

        if rng is None:
            v = np.random.normal(0, R, n_)
        
        else:
            v = rng.normal(0, R, n_)

        t = t + v.reshape(n_,1)

        # 归一化 t（添加噪声后）
        t = t / np.linalg.norm(t)

        # 计算 x 载荷（无噪声）
        po = (to.T@x)/(to.T@to)

        # 计算 x 载荷（添加噪声前）
        p = (to.T@x)/(to.T@to)

        # 添加噪声
        #sensitivity = 2*x_max_norm
        sensitivity = x_max_norm
        R = calibrateAnalyticGaussianMechanism(epsilon, delta, sensitivity)**2

        if rng is None:
            v = np.random.normal(0, R, n_features_)
        
        else:
            v = rng.normal(0, R, n_features_)

        
        p = p + v

        # 计算 y 载荷（无噪声）
        co = (y.T@to)/(to.T@to)

        # 计算 y 载荷（添加噪声前）
        c = (y.T@to)/(to.T@to)

        # 添加噪声
        sensitivity = y_max
        R = calibrateAnalyticGaussianMechanism(epsilon, delta, sensitivity)**2

        if rng is None:
            v = np.random.normal(0, R, 1)

        else:
            v = rng.normal(0, R, 1)

        c = c + v

        # 存储权重、得分和载荷
        x_weights_[:, i] = w
        x_scores_[:, i] = t.reshape(n_)
        x_loadings_[:, i] = p.reshape(n_features_)
        y_loadings_[i] = c

        # 紧缩 x 和 y
        x = x - to @ po
        y = y - to * co

    # 计算回归系数
    coef_ = x_weights_@(np.linalg.inv(x_loadings_.T@x_weights_))@y_loadings_

    # 计算残差
    x_residuals_ = x
    y_residuals_ = y

    return (coef_, x_weights_, x_loadings_, y_loadings_, x_scores_, x_residuals_, y_residuals_ )
