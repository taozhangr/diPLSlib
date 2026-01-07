# 更新日志

本项目的所有显著更改都将记录在本文件中。


## [2.5.0] - 2024-02-14
域自适应偏最小二乘回归

### 新增
- 新增模型类 `DAPLS`，用于（核）域自适应偏最小二乘回归。
- 新增 `demo_daPLS.ipynb` 笔记本，用于演示新模型类。
- 包含新模型类的文档。
- 在 README 中添加了新模型类的使用模式。
- `DIPLS` 类的启发式正则化参数选择演示。

### 变更
- 修改了 `demo_ModelSelection.ipynb` 笔记本，包含 `DIPLS` 类的启发式正则化参数选择演示。

### 修复
- 修复了 `EDPLS` 类中 `check_estimator()` 失败的问题。

### 移除
无

### 破坏性变更
无

## [2.4.2] - 2024-12-13

### 新增
- 为 `fit` 函数添加了 `**kwargs`

### 变更
- 修改了 `DIPLS` 和 `GCTPLS` 类中 `fit` 方法的签名，以接受 `**kwargs`。

### 修复
- 无

### 移除
- 无

### 破坏性变更
- 无

## [未发布] - 2024-11-05

### 新增
- 在文档测试中添加了 'check_estimator()' 以验证模型类。

### 变更
- 在 diPLSlib.models 和 diPLSlib.functions 中添加了输入/输出验证。
- 将 'fit()' 方法中添加的属性从公有更改为私有。
- 适配了笔记本。
- 添加了 'demo_ModelSelection_SciKitLearn.ipynb'。
- 正确执行了测试。

### 修复
无

### 移除
- 无

### 破坏性变更
- 在 'DIPLS' 和 'GCTPLS' 类中将参数 'l' 的类型从 Union[int, List[int]] 更改为 Union[float, tuple]。

## [2.3.0] - 2024-11-06

### 新增
- 新增使用交叉验证进行模型选择的功能。
- 为新功能添加了额外的单元测试。
- 新模型选择功能的文档。

### 变更
- 重构代码以获得更好的可读性和可维护性。
- 将依赖项更新到最新版本。

### 修复
- 修复了 `DIPLS` 类中 `predict()` 方法的一个错误。

### 移除
- 从 `diPLSlib.utils` 中移除了弃用的方法。

### 破坏性变更
- 重构了 `DIPLS` 和 `GCTPLS` 类中 `fit()` 方法的签名。

[2.3.0]: https://github.com/B-Analytics/di-PLS/releases/tag/v2.3.0
[2.2.1]: https://github.com/B-Analytics/di-PLS/releases/tag/v2.2.1

## [2.2.1] - 2024-11-04

### 新增
无

### 变更
无

### 修复
- 纠正了 fit 方法中提取样本数 nt 的错误。
- 在笔记本中测试了正确行为。

### 移除
无

[2.2.1]: https://github.com/B-Analytics/di-PLS/releases/tag/v2.2.1

## [2.2.0] - 2024-11-02

### 新增
- 模型、函数和工具的单元测试

### 变更
- DIPLS 和 GCTPLS 类现在与 sklearn 兼容。
- 文档已更新。

### 修复
- 无

### 移除
- 无

[2.2.0]: https://github.com/B-Analytics/di-PLS/releases/tag/v2.2.0

## [2.1.0] - 2024-11-02
### 新增
- 添加了 utils 子模块以将实用函数外包。
- 添加了文档

### 变更
- 无

### 修复
- 无

### 移除
- 无

[2.1.0]: https://github.com/B-Analytics/di-PLS/releases/tag/v2.1.0

## [2.0.0] - 2024-10-30
### 新增
- 项目架构的重大调整。
- 用于校正转移的新 'GCTPLS' 类。
- GCT-PLS 的演示笔记本。
- 演示笔记本的数据仓库。
- 添加了更新日志。

### 变更
- 将类名从 'model' 更改为 'DIPLS' 和 'GCTPLS'。

### 修复
- 与预测函数相关的次要错误修复。

### 移除
- 无

[2.0.0]: https://github.com/B-Analytics/di-PLS/releases/tag/v2.0.0

## [1.0.2] - 2024-10-30
### 新增
- 无

### 变更
- 无

### 修复
- 文档中的安装和使用章节。

### 移除
- 无

[1.0.2]: https://github.com/B-Analytics/di-PLS/releases/tag/v1.0.2

## [1.0.0] - 2024-10-30
### 新增
- 项目的初始发布。
- 带有 'fit' 和 'predict' 方法的 'Model' 类。
- 支持具有多个域的域自适应场景。

### 变更
- 无

### 修复
- 无

### 移除
- 无

[1.0.0]: https://github.com/B-Analytics/di-PLS/releases/tag/v1.0.0
