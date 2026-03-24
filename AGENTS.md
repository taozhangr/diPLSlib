# AGENTS Guide for diPLSlib

## Mission and Scope
- `diPLSlib` is a small, algorithm-centric library for domain-adaptive PLS regression with sklearn-style estimators.
- Core public models are in `diPLSlib/models.py`: `DIPLS`, `GCTPLS`, `EDPLS`, `KDAPLS`.
- Numerical kernels live in `diPLSlib/functions.py`; estimator classes are thin wrappers over those functions.
- Utility math/helpers (RMSE, DP noise calibration, plotting setup) are in `diPLSlib/utils/misc.py`.
- Treat `build/lib/diPLSlib/*` as build artifacts, not source-of-truth for edits.

## Architecture and Data Flow
- Typical flow is `model.fit(...) -> diPLSlib.functions.* -> matrices stored as estimator attributes -> model.predict(...)`.
- Example: `DIPLS.fit` centers data, calls `algo.dipals(...)`, then stores `b_`, `T_`, `W_`, residuals, etc. (`diPLSlib/models.py`).
- `GCTPLS` reuses `dipals` with `laplacian=True` for paired-source/target calibration transfer.
- `KDAPLS` uses `algo.kdapls(...)` with kernel-specific centering and keeps `centering_` for prediction-time kernel centering.
- `EDPLS` calls `algo.edpls(...)`, which injects Gaussian noise calibrated by `calibrateAnalyticGaussianMechanism`.

## Project-Specific Conventions
- Validation is strict and repeated across estimators: reject sparse input (`issparse`), require finite numeric arrays (`validate_data`/`check_array`), reject complex values.
- Many APIs support either a single target matrix or a list of target domains (`xt` can be `ndarray` or `list[ndarray]`).
- Regularization parameter `l` is scalar or tuple of length `A`; tuple behavior is enforced in `dipals`/`kdapls`.
- `DIPLS.predict` applies explicit domain rescaling via `rescale in {'Target','Source','none'}` before `X @ b_ + b0_`.
- sklearn compatibility matters: estimators inherit `BaseEstimator`/`RegressorMixin`; `EDPLS` defines `__sklearn_tags__` and `_more_tags` for estimator checks.
- Existing docs/comments are Chinese-heavy; preserve current style when modifying nearby code.

## Developer Workflows (Discovered)
- Install (from README): `pip install diPLSlib`.
- Local test entrypoint is `tests/test_doctests.py`, covering:
  - doctests in `diPLSlib.models`, `diPLSlib.functions`, `diPLSlib.utils.misc`
  - sklearn `check_estimator` on all model classes
  - execution of demo notebooks in `notebooks/`
- Run tests from repo root with pytest/unittest (project ships `pytest` in `dependency-groups.dev`).
- Notebook execution on Windows includes an event-loop workaround in tests (`asyncio.WindowsSelectorEventLoopPolicy`).
- Build docs from `doc/` via Sphinx (`make html` or `make.bat html`); docs are autodoc-driven (`doc/diPLSlib.rst`, `doc/conf.py`).

## Integration Points and Dependencies
- External runtime dependencies are NumPy/SciPy/scikit-learn; notebook/docs tooling is included in `pyproject.toml`.
- Model-selection and usage examples live in notebooks, especially `notebooks/demo_ModelSelection.ipynb` and method-specific demos.
- Public usage examples in `README.md` are aligned with estimator APIs; keep signatures backward-compatible when possible.
- Changelog (`changelog.md`) is actively maintained and records behavioral/API shifts (e.g., `fit(**kwargs)`, tuple `l`).

## Where to Look First for Changes
- New algorithm math: `diPLSlib/functions.py` first, then wire through `diPLSlib/models.py`.
- API/validation behavior: estimator `fit`/`predict` methods in `diPLSlib/models.py`.
- Differential privacy logic: `edpls` in `diPLSlib/functions.py` and `calibrateAnalyticGaussianMechanism` in `diPLSlib/utils/misc.py`.
- Repro/verification assets: `tests/test_doctests.py` and `notebooks/*.ipynb`.
