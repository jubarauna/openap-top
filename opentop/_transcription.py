"""Internal representation of one aircraft in a CasADi Opti problem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AircraftTranscription:
    """Symbolic variables and expressions for one discretized trajectory.

    The object deliberately owns the build-specific expressions that used to
    be available only through mutable ``Base`` attributes.  A single-aircraft
    solve and a shared multi-aircraft solve can therefore use the same
    transcription code.
    """

    optimizer: Any
    opti: Any
    X: list[Any]
    Xc: list[list[Any]]
    U: list[Any]
    ts_final: Any
    interval_dts: list[Any]
    objective_raw: Any
    objective_scaled: Any
    objective_scale: float
    objective_kwargs: dict[str, Any]
    collocation_roots: tuple[float, ...]
    projection_center: tuple[float, float] | None = None

    def control_at(self, interval: int, tau: float) -> Any:
        """Continuous, piecewise-linear control at local interval time tau."""
        return (1 - tau) * self.U[interval] + tau * self.U[interval + 1]

    def path_points(self):
        """State/control pairs at mesh boundaries and collocation points."""
        yield from zip(self.X, self.U)
        for k, states in enumerate(self.Xc):
            for tau, state in zip(self.collocation_roots, states):
                yield state, self.control_at(k, tau)
