"""Strategy Protocol для scoring components (GoF Strategy)."""
from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy.typing as npt
import numpy as np

from app.shared.features import TrackFeatures
from app.domain.transition.scoring.bulk.arrays import FeatureArrays


@runtime_checkable
class ScoringComponent(Protocol):
    """Strategy interface для scoring component."""
    name: str

    def score(self, from_t: TrackFeatures, to_t: TrackFeatures) -> float: ...
    def score_pairs(
        self, fa: FeatureArrays, ia: npt.NDArray[np.int64], ib: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]: ...
