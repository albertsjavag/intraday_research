from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from data.base import MarketData
from strategies.base import Strategy


class FeatureBuilder(ABC):
    """Abstract base class for feature builders.

    A feature builder transforms a MarketData object into a feature matrix
    (MultiIndex: symbol × timestamp, or flat with symbol as a column).
    """

    @abstractmethod
    def build(self, data: MarketData) -> pd.DataFrame:
        """Build and return a feature DataFrame.

        Returns
        -------
        pd.DataFrame
            Index: pd.MultiIndex (symbol, timestamp)
            Columns: feature names
        """
        ...


class MLStrategy(Strategy, ABC):
    """Abstract base class for ML-based strategies.

    Extends Strategy with an explicit model fitting interface.
    Concrete subclasses must implement generate_signals() and optionally
    override fit() to train a model.
    """

    def __init__(self, feature_builder: FeatureBuilder) -> None:
        self._feature_builder = feature_builder
        self._model = None
        self._is_fitted = False

    @abstractmethod
    def fit(self, data: MarketData, params: dict | None = None) -> None:
        """Train the model on the provided MarketData."""
        ...

    @abstractmethod
    def generate_signals(self, data: MarketData) -> pd.DataFrame:
        """Generate target weights using the trained model."""
        ...
