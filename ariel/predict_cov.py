"""
predict_cov.py — backwards-compatibility shim.

All functionality has moved to predict.py.
"""
from predict import (
    ystar_pp_cov_normal_vectorized,
    ystar_pp_cov_tophat_vectorized,
    plot_cov,
)

__all__ = [
    "ystar_pp_cov_normal_vectorized",
    "ystar_pp_cov_tophat_vectorized",
    "plot_cov",
]
