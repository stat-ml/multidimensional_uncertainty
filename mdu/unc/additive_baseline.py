from __future__ import annotations

import numpy as np


class AdditiveUncertaintyOrdering:
    """Naive baseline that sums all component uncertainty scores."""

    def __init__(self):
        self.n_features_: int | None = None

    def fit(self, scores_cal: np.ndarray) -> "AdditiveUncertaintyOrdering":
        X = self._as_2d(scores_cal, name="scores_cal")
        if not np.all(np.isfinite(X)):
            raise ValueError("scores_cal must contain only finite values.")

        self.n_features_ = X.shape[1]
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = self._as_2d(scores, name="scores")
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"scores must have {self.n_features_} features, got {X.shape[1]}."
            )
        if not np.all(np.isfinite(X)):
            raise ValueError("scores must contain only finite values.")

        return np.sum(X, axis=1)

    @staticmethod
    def _as_2d(values: np.ndarray, *, name: str) -> np.ndarray:
        X = np.asarray(values, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"{name} must be a 1D or 2D array.")
        if X.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one sample.")
        if X.shape[1] == 0:
            raise ValueError(f"{name} must contain at least one feature.")
        return X

    def _check_is_fitted(self) -> None:
        if self.n_features_ is None:
            raise RuntimeError("Call fit(scores_cal) before predict(scores).")
