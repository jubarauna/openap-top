"""Tests for Cruise.follow_track — the lateral ground-track constraint.

follow_track(lat, lon) stores a reference path as
``self.track_ref = (x_ref, y_ref, s_max)`` where x_ref/y_ref are CasADi
arc-length interpolants in projected (Cartesian) space and s_max is the total
path length. trajectory() then pins each interior node to that path and drops
the default smooth-heading constraint.
"""

import openap.casadi as oc
import pytest

import numpy as np
import opentop as top

# ---- helpers ----------------------------------------------------------------


def _track_between(opt, n=15, bow_deg=0.0):
    """Build a (lat, lon) track from opt's origin to destination.

    Points are linearly interpolated in lat/lon so the endpoints match the
    boundary conditions exactly. A non-zero ``bow_deg`` adds a half-sine
    lateral offset in latitude (zero at both ends) to make the path a
    meaningful detour rather than the great-circle default.
    """
    frac = np.linspace(0.0, 1.0, n)
    lat = opt.lat1 + (opt.lat2 - opt.lat1) * frac
    lon = opt.lon1 + (opt.lon2 - opt.lon1) * frac
    lat = lat + bow_deg * np.sin(np.pi * frac)
    return lat, lon


# ---- fixtures ---------------------------------------------------------------


@pytest.fixture()
def opt(aircraft_type, short_flight):
    return top.Cruise(
        aircraft_type,
        short_flight["origin"],
        short_flight["destination"],
        short_flight["m0"],
    )


# ---- unit tests: track_ref construction -------------------------------------


class TestTrackRefConstruction:
    def test_track_ref_none_by_default(self, opt):
        assert opt.track_ref is None

    def test_follow_track_sets_three_tuple(self, opt):
        lat, lon = _track_between(opt)
        opt.follow_track(lat, lon)

        assert opt.track_ref is not None
        assert len(opt.track_ref) == 3
        x_ref, y_ref, s_max = opt.track_ref
        assert callable(x_ref)
        assert callable(y_ref)
        assert isinstance(s_max, float)
        assert s_max > 0

    def test_accepts_python_lists(self, opt):
        """lat/lon are passed through np.asarray, so plain lists must work."""
        lat, lon = _track_between(opt)
        opt.follow_track(lat.tolist(), lon.tolist())
        assert opt.track_ref is not None

    def test_s_max_equals_total_arc_length(self, opt):
        lat, lon = _track_between(opt)
        opt.follow_track(lat, lon)
        _, _, s_max = opt.track_ref

        x, y = opt.proj(lon, lat)
        expected = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
        assert s_max == pytest.approx(expected, rel=1e-9)

    def test_accepts_non_uniform_spacing(self, opt):
        """The path is parametrized by cumulative arc length, not by index, so
        unevenly spaced points must build and still reproduce both endpoints."""
        frac = np.array([0.0, 0.02, 0.05, 0.1, 0.5, 0.85, 0.95, 1.0])
        lat = opt.lat1 + (opt.lat2 - opt.lat1) * frac
        lon = opt.lon1 + (opt.lon2 - opt.lon1) * frac

        opt.follow_track(lat, lon)
        x_ref, y_ref, s_max = opt.track_ref

        x0, y0 = opt.proj(opt.lon1, opt.lat1)
        xf, yf = opt.proj(opt.lon2, opt.lat2)
        assert float(x_ref(0.0)) == pytest.approx(x0, abs=1.0)
        assert float(y_ref(0.0)) == pytest.approx(y0, abs=1.0)
        assert float(x_ref(s_max)) == pytest.approx(xf, abs=1.0)
        assert float(y_ref(s_max)) == pytest.approx(yf, abs=1.0)

    def test_rejects_duplicate_consecutive_points(self, opt):
        """A zero-length segment makes the arc-length grid non-increasing, which
        CasADi's bspline interpolant rejects."""
        lat, lon = _track_between(opt)
        lat = np.insert(lat, 1, lat[0])  # duplicate the first point
        lon = np.insert(lon, 1, lon[0])
        with pytest.raises(RuntimeError, match="is_increasing"):
            opt.follow_track(lat, lon)

    def test_interpolants_reproduce_projected_endpoints(self, opt):
        lat, lon = _track_between(opt)
        opt.follow_track(lat, lon)
        x_ref, y_ref, s_max = opt.track_ref

        x0, y0 = opt.proj(opt.lon1, opt.lat1)
        xf, yf = opt.proj(opt.lon2, opt.lat2)

        # A bspline interpolant passes through the data at the knots (s=0, s=s_max).
        assert float(x_ref(0.0)) == pytest.approx(x0, abs=1.0)
        assert float(y_ref(0.0)) == pytest.approx(y0, abs=1.0)
        assert float(x_ref(s_max)) == pytest.approx(xf, abs=1.0)
        assert float(y_ref(s_max)) == pytest.approx(yf, abs=1.0)


# ---- unit tests: endpoint validation ----------------------------------------


class TestEndpointValidation:
    def test_rejects_start_far_from_origin(self, opt):
        lat, lon = _track_between(opt)
        lat[0] += 1.0  # ~111 km off — well over the 1 km tolerance
        with pytest.raises(ValueError, match="track endpoints do not match"):
            opt.follow_track(lat, lon)

    def test_rejects_end_far_from_destination(self, opt):
        lat, lon = _track_between(opt)
        lon[-1] += 1.0
        with pytest.raises(ValueError, match="track endpoints do not match"):
            opt.follow_track(lat, lon)

    def test_accepts_small_endpoint_offset(self, opt):
        """An offset under the 1 km tolerance must be accepted."""
        lat, lon = _track_between(opt)
        # ~50 m north of the origin — inside tolerance.
        lat[0] += 50.0 / oc.geo.distance(opt.lat1, opt.lon1, opt.lat1 + 1, opt.lon1)
        opt.follow_track(lat, lon)
        assert opt.track_ref is not None


# ---- integration test: constraint is actually enforced ----------------------


class TestFollowTrackTrajectory:
    def test_trajectory_follows_bowed_track(self, opt):
        """Solve with a bowed reference path and confirm the solution's lateral
        positions stay on that path rather than on the great circle."""
        lat, lon = _track_between(opt, n=25, bow_deg=0.15)
        opt.follow_track(lat, lon)

        df = opt.trajectory(objective="fuel")
        assert df is not None and len(df) > 0

        x_ref, y_ref, s_max = opt.track_ref
        ref_s = np.linspace(0.0, s_max, 400)
        ref_x = np.array([float(x_ref(s)) for s in ref_s])
        ref_y = np.array([float(y_ref(s)) for s in ref_s])

        sol_x, sol_y = opt.proj(df.longitude.to_numpy(), df.latitude.to_numpy())

        # Every solved node must sit on (very near) the reference polyline.
        for xk, yk in zip(np.atleast_1d(sol_x), np.atleast_1d(sol_y)):
            dist = np.min(np.hypot(ref_x - xk, ref_y - yk))
            assert dist < 5_000, f"node {dist / 1000:.1f} km off the reference track"

        # The bow must actually show up: the great circle would keep heading
        # nearly constant, so a followed bow means a non-trivial heading swing.
        assert df.heading.max() - df.heading.min() > 2.0
