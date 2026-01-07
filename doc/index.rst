.. diPLSlib documentation master file, created by
   sphinx-quickstart on Sun Nov  3 00:13:47 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

diPLSlib 文档
======================

简介
------------

**diPLSlib** 是一个专为多元校正中的域自适应设计的 Python 库，重点关注保护隐私的回归和校正模型维护。它提供了一个与 scikit-learn 兼容的 API，并实现了对齐不同域之间数据分布的高级方法，从而构建健壮且可转移的回归模型。

该库具有多种先进算法，包括：

- **域不变偏最小二乘 (di-PLS/mdi-PLS):** 对齐源域和目标域之间的特征分布，以提高模型的泛化能力。
- **基于图的校正转移 (GCT-PLS):** 在潜变量空间中最小化来自不同域的配对样本之间的差异。
- **核域自适应 PLS (KDAPLS):** 将数据投影到再生核希尔伯特空间中，进行非参数域自适应。
- **差分隐私 PLS (EDPLS):** 使用 :math:`(\epsilon, \delta)`-差分隐私框架确保敏感数据的隐私保证。

diPLSlib 适用于化学计量学、分析化学以及其他需要健壮校正转移和保护隐私建模的领域。有关更多详细信息、使用示例和 API 文档，请参阅以下章节。

.. toctree::
   :maxdepth: 2
   :caption: 内容:

   diPLSlib
