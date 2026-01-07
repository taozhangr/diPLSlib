# Sphinx 文档生成器的配置文件。
#
# 有关内置配置值的完整列表，请参阅文档：
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../'))

# -- 项目信息 -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'diPLSlib'
copyright = '2025, Ramin Nikzad-Langerodi'
author = 'Ramin Nikzad-Langerodi'
release = '2.5.0'

# -- 通用配置 ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'numpydoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax']
    #'sphinx.ext.napoleon']

templates_path = ['_templates']
exclude_patterns = []



# -- HTML 输出选项 -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
