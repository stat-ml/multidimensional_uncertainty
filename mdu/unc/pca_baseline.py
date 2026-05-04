from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCAUncertaintyOrdering:
    """Project multidimensional uncertainty scores to one PCA component."""

    def __init__(self, variance_tol: float = 1e-12):
        self.variance_tol = float(variance_tol)
        self.scaler_: StandardScaler | None = None
        self.pca_: PCA | None = None
        self.sign_: float = 1.0
        self.n_features_: int | None = None
        self.is_constant_: bool = False

    def fit(self, scores_cal: np.ndarray) -> "PCAUncertaintyOrdering":
        X = self._as_2d(scores_cal, name="scores_cal")
        if not np.all(np.isfinite(X)):
            raise ValueError("scores_cal must contain only finite values.")

        self.n_features_ = X.shape[1]
        variances = np.var(X, axis=0)
        self.is_constant_ = bool(np.all(variances <= self.variance_tol))

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        if self.is_constant_:
            self.pca_ = None
            self.sign_ = 1.0
            return self

        self.pca_ = PCA(n_components=1)
        self.pca_.fit(X_scaled)

        loadings_sum = float(np.sum(self.pca_.components_[0]))
        self.sign_ = -1.0 if loadings_sum < 0.0 else 1.0
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

        if self.is_constant_:
            return np.zeros(X.shape[0], dtype=np.float64)

        X_scaled = self.scaler_.transform(X)
        return self.sign_ * self.pca_.transform(X_scaled).ravel()

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
        if self.scaler_ is None or self.n_features_ is None:
            raise RuntimeError("Call fit(scores_cal) before predict(scores).")
