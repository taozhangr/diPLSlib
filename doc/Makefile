# Sphinx 文档的简易 Makefile
#

# 您可以从命令行设置这些变量，也可以从前两个变量的环境中设置。
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# 将其放在首位，这样不带参数执行 "make" 就等同于 "make help"。
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# 总括目标 (Catch-all target)：使用新的 "make mode" 选项将所有未知目标路由到 Sphinx。
# $(O) 被用作 $(SPHINXOPTS) 的缩写。
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
