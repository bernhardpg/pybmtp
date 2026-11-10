"""Tests for the fixed-duration mode: a given total time and the derivative cost
``int ||d^k r / dt^k||^2 dt`` in place of minimum time.

The oracles: the derivative cost formula agrees with numerical quadrature, the
trajectory update returns exactly that cost at the given duration, continuity and
rest orders above the limits' order are allowed, and the full planner keeps the
duration, stays collision-free, and never increases the cost between feasible
iterates.
"""

import numpy as np
import pybezier as pb
import pydrake.all as pd
from scipy.integrate import simpson

from pybmtp import Limits, MinimumTimePlanner
from pybmtp.collisions import intersect_with_hpolyhedra
from pybmtp.updaters import TrajectoryUpdater
from pybmtp.updaters.trajectory import derivative_cost


def box(lb, ub):
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    d = lb.size
    A = np.vstack([np.eye(d), -np.eye(d)])
    return pd.HPolyhedron(A, np.concatenate([ub, -lb]))


def limits(dim, vmax=2.0, amax=4.0):
    return Limits(box([-vmax] * dim, [vmax] * dim), box([-amax] * dim, [amax] * dim))


def quadrature_cost(curve: pb.BezierCurve, segment_time: float, k: int, n: int = 4001) -> float:
    """``int_0^T ||d^k r / dt^k||^2 dt`` by Simpson's rule, for ``curve`` on [0, 1]."""
    derivative = curve
    for _ in range(k):
        derivative = derivative.derivative()
    s = np.linspace(0.0, 1.0, n)
    values = np.array([np.sum(np.asarray(derivative(si)) ** 2) for si in s])
    return segment_time ** (1 - 2 * k) * simpson(values, x=s)


def test_derivative_cost_matches_quadrature():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(8, 2))
    curve = pb.BezierCurve(points, 0.0, 1.0)
    for k in (1, 2, 3):
        expected = quadrature_cost(curve, 1.7, k)
        actual = derivative_cost(pb.CompositeBezierCurve([curve]), 1.7, k)
        np.testing.assert_allclose(actual, expected, rtol=1e-9)


def test_updater_keeps_the_duration_and_returns_the_derivative_cost():
    updater = TrajectoryUpdater(
        np.array([0.0, 0.0]),
        np.array([2.0, 1.0]),
        box([-5, -5], [5, 5]),
        3,
        limits(2),
        degree=7,
        continuity_order=3,
        terminal_order=3,
        fixed_segment_time=1.5,
        cost_derivative_order=3,
    )
    trajectory, segment_time, cost, _ = updater.solve({})
    np.testing.assert_allclose(segment_time, 1.5, rtol=1e-6)
    np.testing.assert_allclose(cost, derivative_cost(trajectory, 1.5, 3), rtol=1e-5)
    # Rest to order 3: the first and last four control points coincide.
    first, last = trajectory.curves[0].points, trajectory.curves[-1].points
    np.testing.assert_allclose(first[:4], np.tile(first[0], (4, 1)), atol=1e-6)
    np.testing.assert_allclose(last[-4:], np.tile(last[-1], (4, 1)), atol=1e-6)


def test_planner_with_fixed_duration_detours_without_raising_the_cost():
    obstacle = box([-0.5, -0.6], [0.5, 0.6])
    path = np.array([[-2.0, 0.0], [0.0, 1.2], [2.0, 0.0]])
    result = MinimumTimePlanner(
        trajectory_degree=7,
        continuity_order=3,
        terminal_order=3,
        max_iter=50,
        rel_term=1e-3,
        fixed_duration=8.0,
        cost_derivative_order=3,
    ).solve(path, [obstacle], box([-5, -5], [5, 5]), limits(2))

    np.testing.assert_allclose(result.total_time, 8.0, rtol=1e-6)
    np.testing.assert_allclose(result.trajectory(result.trajectory.initial_time), path[0], atol=1e-5)
    np.testing.assert_allclose(result.trajectory(result.trajectory.final_time), path[-1], atol=1e-5)
    assert intersect_with_hpolyhedra(result.trajectory, [obstacle], tol=1e-3) == {}
    assert len(result.feasible_costs) >= 2
    assert all(b <= a * (1 + 1e-6) for a, b in zip(result.feasible_costs, result.feasible_costs[1:]))
    assert result.feasible_costs[-1] < result.feasible_costs[0]
