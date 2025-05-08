.. diPLSlib documentation master file, created by
   sphinx-quickstart on Sun Nov  3 00:13:47 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

diPLSlib documentation
======================

Introduction
------------

**diPLSlib** is a Python library designed for domain adaptation in multivariate calibration, with a focus on privacy-preserving regression and calibration model maintenance. It provides a scikit-learn compatible API and implements advanced methods for aligning data distributions across different domains, enabling robust and transferable regression models.

The library features several state-of-the-art algorithms, including:

- **Domain-Invariant Partial Least Squares (di-PLS/mdi-PLS):** Aligns feature distributions between source and target domains to improve model generalization.
- **Graph-based Calibration Transfer (GCT-PLS):** Minimizes discrepancies between paired samples from different domains in the latent variable space.
- **Kernel Domain Adaptive PLS (KDAPLS):** Projects data into a reproducing kernel Hilbert space for non-parametric domain adaptation.
- **Differentially Private PLS (EDPLS):** Ensures privacy guarantees for sensitive data using the :math:`(\epsilon, \delta)`-differential privacy framework.

diPLSlib is suitable for chemometrics, analytical chemistry, and other fields where robust calibration transfer and privacy-preserving modeling are required. For more details, usage examples, and API documentation, please refer to the sections below.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   diPLSlib
