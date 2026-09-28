import jax
import jax.numpy as jnp
import numpy as np

from cannon.game import analyse_shot
from cannon.physics import launch_to_raw, simulate
from cannon.solver import (DEFAULT_WALL, INITIAL_GUESSES, LEARNING_RATE, N_ITERS, adam_init, adam_update,
                           initial_raw_batch, loss_fn, miss_loss, optimise_batch, point_segment_sq_dist, solve,
                           wall_penalty)

N = len(INITIAL_GUESSES)


def test_point_segment_sq_dist():
    a, b = jnp.array([0.0, 0.0]), jnp.array([10.0, 0.0])
    assert jnp.allclose(point_segment_sq_dist(a, b, jnp.array([5.0, 3.0])), 9.0)     # above the middle
    assert jnp.allclose(point_segment_sq_dist(a, b, jnp.array([13.0, 4.0])), 25.0)   # past the end
    p = jnp.array([1.0, 1.0])
    assert jnp.isfinite(point_segment_sq_dist(p, p, jnp.array([4.0, 5.0])))         # zero-length segment


def test_miss_loss_between_timesteps_is_zero():
    traj = simulate(launch_to_raw(jnp.pi / 4, 20.0))
    assert miss_loss(traj, 0.5 * (traj[123] + traj[124])) < 1e-4


def test_wall_penalty():
    target = jnp.array([60.0, 5.0])
    traj = simulate(launch_to_raw(jnp.deg2rad(20.0), 30.0))    # low shot through a tall wall
    wall_on = jnp.array([30.0, 2.0, 15.0, 1.0])
    assert wall_penalty(traj, target, wall_on) > 0
    assert wall_penalty(traj, target, wall_on.at[3].set(0.0)) == 0
    assert wall_penalty(traj, target, jnp.array([80.0, 2.0, 15.0, 1.0])) == 0     # behind the target


def test_wall_gradient_raises_the_angle():
    raw = launch_to_raw(jnp.deg2rad(20.0), 30.0)
    g = jax.grad(loss_fn)(raw, jnp.array([60.0, 5.0]), jnp.array([30.0, 2.0, 15.0, 1.0]))
    assert jnp.all(jnp.isfinite(g)) and g[0] < 0


def test_adam_first_step_is_lr_for_every_param():
    raw = jnp.array([1.0, -2.0])
    new_raw, (_, _, t) = adam_update(raw, jnp.array([100.0, -0.01]), adam_init(raw))
    assert t == 1
    assert jnp.allclose(jnp.abs(new_raw - raw), LEARNING_RATE, rtol=1e-3)


def test_optimise_batch_converges():
    target = jnp.array([40.0, 5.0])
    final, hist, losses = optimise_batch(initial_raw_batch(), target, DEFAULT_WALL)
    assert final.shape == (N, 2) and hist.shape == (N, N_ITERS, 2) and losses.shape == (N, N_ITERS)
    assert min(float(miss_loss(simulate(r), target)) for r in final) < 1e-3


def test_solve_ignores_a_wall_that_is_not_in_the_way():
    target = jnp.array([84.0, 15.5])
    wall_off = jnp.array([41.0, 2.0, 12.0, 0.0])
    f_off, hist, losses = solve(initial_raw_batch(), target, wall_off)
    assert hist.shape == (N, 2 * N_ITERS, 2) and losses.shape == (N, 2 * N_ITERS)
    f_low, _, _ = solve(initial_raw_batch(), target, wall_off.at[3].set(1.0))   # under both final arcs
    assert jnp.allclose(f_off, f_low, atol=1e-4)


def test_solve_clears_a_blocking_wall():
    target, wall = jnp.array([60.0, 5.0]), jnp.array([30.0, 2.0, 15.0, 1.0])
    final, _, _ = solve(initial_raw_batch(), target, wall)
    assert all(analyse_shot(simulate(r), np.asarray(target), np.asarray(wall))["hit"] for r in final)
