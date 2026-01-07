import doctest
import unittest
import diPLSlib.models
import diPLSlib.functions
import diPLSlib.utils.misc
from sklearn.utils.estimator_checks import check_estimator
from diPLSlib.models import DIPLS, GCTPLS, EDPLS, KDAPLS
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import sys

# 修复 Windows 上 zmq 的事件循环警告
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class TestDocstrings(unittest.TestCase):

    # 测试所有文档字符串示例是否能无错运行
    def test_docstrings(self):

        # 在 diPLSlib 的所有模块中运行 doctest
        doctest.testmod(diPLSlib.models)
        doctest.testmod(diPLSlib.functions)
        doctest.testmod(diPLSlib.utils.misc)

    # 测试 diPLSlib.model 类是否通过 check_estimator
    def test_check_estimator(self):

        models = [
        DIPLS(),
        GCTPLS(),
        EDPLS(A=2, epsilon=1.0, delta=0.05),  # 为 EDPLS 添加必需的参数
        KDAPLS()
        ]
    
        for model in models:
            check_estimator(model)
        

    # 测试所有 notebook 是否能无错运行
    def test_notebooks(self):
        # 待测试的 notebook 列表
        notebooks = [
            './notebooks/demo_diPLS.ipynb',
            './notebooks/demo_mdiPLS.ipynb',
            './notebooks/demo_gctPLS.ipynb',
            './notebooks/demo_edPLS.ipynb',
            './notebooks/demo_daPLS.ipynb',
        ]
        
        for notebook in notebooks:
            with open(notebook) as f:
                nb = nbformat.read(f, as_version=4)
                ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

                # 将工作目录设置为项目根目录
                root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                os.chdir(root_dir)

                ep.preprocess(nb, {'metadata': {'path': './notebooks/'}})


if __name__ == '__main__':
    unittest.main()
