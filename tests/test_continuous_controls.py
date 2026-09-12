"""Analytical integration catches stepped controls in dynamics or quadrature."""

import casadi as ca
import pytest

import numpy as np
from opentop.base import Base


class Integrator(Base):
    def __init__(self):
        self.nodes = 2
        self.polydeg = 3
        self.performance_model = "openap"
        self.solver_options = {}
        self.x_lb = [-100.0] * 5
        self.x_ub = [100.0] * 5
        self.x_0_lb = self.x_0_ub = [0.0] * 5
        self.x_f_lb, self.x_f_ub = self.x_lb, self.x_ub
        self.u_lb = self.u_0_lb = self.u_f_lb = [-10.0] * 3
        self.u_ub = self.u_0_ub = self.u_f_ub = [10.0] * 3
        self.x_guess = np.zeros((3, 5))
        self.u_guess = [0.0] * 3

    def init_model(self, objective, *, function_name="f", **kwargs):
        self.x = ca.MX.sym("x", 5)
        self.u = ca.MX.sym("u", 3)
        dt = ca.MX.sym("dt")
        self.func_dynamics = ca.Function(
            function_name,
            [self.x, self.u, dt],
            [ca.vertcat(self.u[0], 0, 0, 0, 1), dt * self.u[0] ** 2],
        )


@pytest.mark.parametrize("variable", [False, True])
def test_linear_controls_integrate_dynamics_and_cost(variable):
    opt = Integrator()
    problem = ca.Opti()
    durations = [2.0, 3.0] if variable else [2.5, 2.5]
    t = opt._add_transcription(
        problem, "fuel", 5.0, variable_timestep=variable, dt_min=1, dt_max=4
    )
    assert len(t.U) == len(t.X) == 3
    values = [0.0, 2.0, 1.0]
    for control, value in zip(t.U, values):
        problem.set_initial(control, [value, 0, 0])
    elapsed, position, cost = 0.0, 0.0, 0.0
    for k, dt in enumerate(durations):
        if variable:
            problem.set_initial(t.interval_dts[k], dt)
        problem.set_initial(t.X[k], [position, 0, 0, 0, elapsed])
        u0, u1 = values[k : k + 2]
        for tau, x in zip(t.collocation_roots, t.Xc[k]):
            xp = position + dt * (u0 * tau + (u1 - u0) * tau**2 / 2)
            problem.set_initial(x, [xp, 0, 0, 0, elapsed + tau * dt])
        position += dt * (u0 + u1) / 2
        cost += dt * (u0 * u0 + u0 * u1 + u1 * u1) / 3
        elapsed += dt
        np.testing.assert_allclose(
            problem.debug.value(t.control_at(k, 1), problem.initial()),
            problem.debug.value(t.U[k + 1], problem.initial()),
        )
    problem.set_initial(t.X[-1], [position, 0, 0, 0, elapsed])

    def evaluate(x):
        return np.asarray(problem.debug.value(x, problem.initial())).ravel()

    residual, lower, upper = map(evaluate, [problem.g, problem.lbg, problem.ubg])
    assert np.max(np.maximum(lower - residual, residual - upper)) < 1e-10
    np.testing.assert_allclose(evaluate(t.objective_raw), cost, atol=1e-12)
    assert len(list(t.path_points())) == 9
    # The terminal control participates in the final interval's physics and cost.
    problem.set_initial(t.U[-1], [3, 0, 0])
    changed = evaluate(problem.g)
    assert np.max(np.maximum(lower - changed, changed - upper)) > 0.1
    assert abs(evaluate(t.objective_raw)[0] - cost) > 1
