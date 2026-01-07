# diPLSlib: 用于多元校正中域自适应的 Python 库

![](https://img.shields.io/badge/python-3.13-blue.svg)
![](https://static.pepy.tech/badge/diplslib)

![](https://user-images.githubusercontent.com/77445667/104728864-d5fede80-5737-11eb-8aad-59f9901a0cf4.png)

diPLSlib 旨在构建保护隐私的回归模型，并通过域自适应执行校正模型维护。它具有与 scikit-learn 兼容的 API，并包含以下方法：

- （多）域不变偏最小二乘回归 (di-PLS/mdi-PLS)
- 基于图的校正转移 (GCT-PLS)
- $(\epsilon, \delta)$-差分隐私偏最小二乘回归 (edPLS)

**引用**: 如果您在研究中使用此库，请引用以下参考资料：

```
Nikzad-Langerodi R.(2024). diPLSlib : A Python library for domain adaptation in multivariate calibration (version 2.4.1). URL: https://di-pls.readthedocs.io/
```
或使用 Bibtex 格式：
```bibtex
@misc{nikzad2024diPLSlib,
author = {Nikzad-Langerodi, Ramin},
month = {12},
title = {diPLSlib: A Python library for domain adaptation in multivariate calibration},
url = {https://di-pls.readthedocs.io/},
year = {2024}
}
```

# 安装
```bash
pip install diPLSlib
```

# 快速开始
## 如何应用 di-PLS
训练回归模型
```python
from diPLSlib.models import DIPLS
from diPLSlib.utils import misc

l = 100000                    # 或 l = (10000, 100) 正则化
m = DIPLS(A=2, l=l)
m.fit(X, y, X_source, X_target)

# 通常 X=X_source 且 y 是相应的响应值
```
应用模型
```python
yhat_dipls = m.predict(X_test)
err = misc.rmse(y_test, yhat_dipls)
```

## 如何应用 mdi-PLS
```python
from diPLSlib.models import DIPLS

l = 100000                    # 或 l = (5, 100, 1000) 正则化
m = DIPLS(A=3, l=l, target_domain=2)
m.fit(X, y, X_source, X_target)

# X_target = [X1, X2, ... , Xk] 是目标域数据的列表
# 参数 target_domain 指定应为哪个域训练模型（这里是 X2）。
```

## 如何应用 GCT-PLS
```python
from diPLSlib.models import GCTPLS

# 训练
l = 10                         # 或 l = (10, 10) 正则化
m = GCTPLS(A=2, l=l)
m.fit(X, y, X_source, X_target)

# X_source 和 X_target 分别保存了在源域和目标域中测量的相同样本。
```

## 如何应用 EDPLS
```python
from diPLSlib.models import EDPLS

# 训练
epsilon = 1                      # 隐私损失
delta = 0.05                     # 隐私保证失败概率
m = EDPLS(A=2, epsilon=epsilon, delta=delta)
m.fit(X, y)
```

## 如何应用 KDAPLS
```python
from diPLSlib.models import KDAPLS

# 训练
model = KDAPLS(A=2, l=0.5, kernel_params={"type": "rbf", "gamma": 0.001})
model.fit(x, y, xs, xt)
```

## 示例
更多示例，请参考 [Notebooks](notebooks)：

- [使用 di-PLS 进行域自适应](https://github.com/B-Analytics/diPLSlib/blob/main/notebooks/demo_diPLS.ipynb)
- [包含多个域 (mdi-PLS)](https://github.com/B-Analytics/diPLSlib/blob/main/notebooks/demo_mdiPLS.ipynb)
- [使用 GCT-PLS 进行隐式校正转移](https://github.com/B-Analytics/diPLSlib/blob/main/notebooks/demo_gctPLS.ipynb)
- [模型选择 (配合 `scikit-learn`)](https://github.com/B-Analytics/diPLSlib/blob/main/notebooks/demo_ModelSelection.ipynb)
- [使用 EDPLS 进行保护隐私的回归](https://github.com/B-Analytics/diPLSlib/blob/main/notebooks/demo_edPLS.ipynb)
- [使用 KDAPLS 进行非参数域自适应](https://github.com/B-Analytics/diPLSlib/blob/main/notebooks/demo_daPLS.ipynb)

# 文档
文档可以在 [这里](https://di-pls.readthedocs.io/en/latest/diPLSlib.html) 找到。

# 致谢
di-PLS 的第一个版本由 Ramin Nikzad-Langerodi, Werner Zellinger, Edwin Lughofer, Bernhard Moser 和 Susanne Saminger-Platz 开发
并发表于：

- *Ramin Nikzad-Langerodi, Werner Zellinger, Edwin Lughofer, and Susanne Saminger-Platz
Analytical Chemistry 2018 90 (11), 6693-6701 https://doi.org/10.1021/acs.analchem.8b00498*

对初始算法的进一步改进发表于：

- *R. Nikzad-Langerodi, W. Zellinger, S. Saminger-Platz and B. Moser, "Domain-Invariant Regression Under Beer-Lambert's Law," 2019 18th IEEE International Conference On Machine Learning And Applications (ICMLA), Boca Raton, FL, USA, 2019, pp. 581-586, https://doi.org/10.1109/ICMLA.2019.00108.*

- *Ramin Nikzad-Langerodi, Werner Zellinger, Susanne Saminger-Platz, Bernhard A. Moser,
Domain adaptation for regression under Beer–Lambert’s law,
Knowledge-Based Systems, Volume 210, 2020, https://doi.org/10.1016/j.knosys.2020.106447.*

- *Bianca Mikulasek, Valeria Fonseca Diaz, David Gabauer, Christoph Herwig, Ramin Nikzad-Langerodi,
"Partial least squares regression with multiple domains" Journal of Chemometrics 2023 37 (5), e3477, https://doi.org/10.13140/RG.2.2.23750.75845*

- *Ramin Nikzad-Langerodi & Florian Sobieczky (2021). Graph‐based calibration transfer. Journal of Chemometrics, 35(4), e3319. https://doi.org/10.1002/cem.3319*

- *Ramin Nikzad-Langerodi,  Mohit Kumar, Du Nguyen Duy, and Mahtab Alghasi (2024), "(epsilon, delta)-Differentially Private Partial Least Squares Regression", unpublished.*

# 联系我们
Bottleneck Analytics GmbH  
info@bottleneck-analytics.com
