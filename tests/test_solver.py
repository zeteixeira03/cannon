import jax
import jax.numpy as jnp
import numpy as np

from cannon.game import analyse_shot, choose_arc
from cannon.physics import DT, N_STEPS, launch_to_raw, simulate, to_launch
from cannon.solver import (LEARNING_RATE, N_ARCS, N_ITERS, adam_init, adam_update, closest_approach,
                           initial_guesses, loss_fn, make_scene, optimise_batch, point_segment_sq_dist, solve,
                           target_path, wall_penalty)

N = N_ARCS
WALL = jnp.array([30.0, 2.0, 15.0, 1.0])
TIMES = DT * jnp.arange(1, N_STEPS + 1)


def low_shot():
    """20°, 30 m/s: passes x = 30 at about 2 m high, so it goes through ``WALL``."""
    return simulate(launch_to_raw(jnp.deg2rad(20.0), 30.0))


def moving_target_through(point, i, vel):
    """Target path with velocity ``vel`` that is at ``point`` at timestep ``i``."""
    return point + (TIMES - TIMES[i])[:, None] * jnp.asarray(vel)


def hits(final, scene):
    tpath = np.asarray(target_path(scene))
    return [analyse_shot(simulate(r, scene.drag), tpath, np.asarray(scene.wall), np.asarray(scene.target))["hit"]
            for r in final]


# ----- loss pieces --------------------------------------------------------

def test_point_segment_sq_dist():
    a, b = jnp.array([0.0, 0.0]), jnp.array([10.0, 0.0])
    assert jnp.allclose(point_segment_sq_dist(a, b, jnp.array([5.0, 3.0])), 9.0)     # above the middle
    assert jnp.allclose(point_segment_sq_dist(a, b, jnp.array([13.0, 4.0])), 25.0)   # past the end
    p = jnp.array([1.0, 1.0])
    assert jnp.isfinite(point_segment_sq_dist(p, p, jnp.array([4.0, 5.0])))         # zero-length segment


def test_closest_approach_static_target_between_timesteps():
    traj = low_shot()
    target = 0.5 * (traj[123] + traj[124])
    d2, i = closest_approach(traj, jnp.broadcast_to(target, traj.shape))
    assert d2 < 1e-4 and i == 123


def test_closest_approach_static_target_off_the_path():
    traj = low_shot()
    target = traj[100] + jnp.array([0.0, 3.0])
    d2, _ = closest_approach(traj, jnp.broadcast_to(target, traj.shape))
    geometric = jnp.min(jax.vmap(point_segment_sq_dist, in_axes=(0, 0, None))(traj[:-1], traj[1:], target))
    assert jnp.allclose(d2, geometric, rtol=1e-4)                # a static target: plain distance to the path


def test_closest_approach_moving_target_on_time():
    traj = low_shot()
    d2, i = closest_approach(traj, moving_target_through(traj[150], 150, (-6.0, 2.0)))
    assert d2 < 1e-4 and i in (149, 150)


def test_closest_approach_moving_target_late_is_a_miss():
    # The target crosses the ball's path, but 1 s after the ball has passed.
    traj = low_shot()
    d2, _ = closest_approach(traj, moving_target_through(traj[100], 150, (-6.0, 0.0)))
    assert d2 > 1.0


def test_wall_penalty_counts_only_the_flight_before_impact():
    traj = low_shot()
    assert wall_penalty(traj, WALL, N_STEPS - 1) > 0
    assert wall_penalty(traj, WALL.at[3].set(0.0), N_STEPS - 1) == 0
    before_wall = int(jnp.argmax(traj[:, 0] > 20.0))            # impact at x ≈ 20, left of the wall
    assert wall_penalty(traj, WALL, before_wall) < 1e-9          # ~0: the smooth mask only has tails


def test_wall_counts_for_a_target_that_starts_before_it():
    # The target starts left of the wall but is hit to the right of it: the ball goes through
    # the wall, so the penalty must count.
    traj = low_shot()
    i = int(jnp.argmax(traj[:, 0] > 45.0))
    scene = make_scene(target=(20.0, 0.0), target_vel=((traj[i, 0] - 20.0) / TIMES[i], traj[i, 1] / TIMES[i]),
                       wall=WALL)
    d2, impact = closest_approach(traj, target_path(scene))
    assert d2 < 1e-3 and wall_penalty(traj, scene.wall, impact) > 0


def test_wall_gradient_raises_the_angle():
    raw = launch_to_raw(jnp.deg2rad(20.0), 30.0)
    g = jax.grad(loss_fn)(raw, make_scene((60.0, 5.0), wall=WALL))
    assert jnp.all(jnp.isfinite(g)) and g[0] < 0


# ----- optimiser ----------------------------------------------------------

def test_adam_first_step_is_lr_for_every_param():
    raw = jnp.array([1.0, -2.0])
    new_raw, (_, _, t) = adam_update(raw, jnp.array([100.0, -0.01]), adam_init(raw))
    assert t == 1
    assert jnp.allclose(jnp.abs(new_raw - raw), LEARNING_RATE, rtol=1e-3)


def test_initial_guesses_are_the_two_drag_free_arcs():
    scene = make_scene((70.0, 10.0))
    raw = initial_guesses(scene)
    assert raw.shape == (N_ARCS, 2)
    angles = [float(to_launch(r)[0]) for r in raw]
    assert angles[0] < np.pi / 4 < angles[1]                     # one flat, one lobbed
    assert all(loss_fn(r, scene) < 1e-3 for r in raw)            # both already hit without drag


def test_initial_guesses_land_inside_the_simulated_window():
    for x in range(5, 126, 10):                                  # the whole view
        for y in range(0, 51, 10):
            scene = make_scene((float(x), float(y)))
            assert all(loss_fn(r, scene) < 1e-2 for r in initial_guesses(scene)), (x, y)


def test_optimise_batch_converges():
    scene = make_scene((40.0, 5.0))
    final, hist, losses = optimise_batch(initial_guesses(scene), scene)
    assert final.shape == (N, 2) and hist.shape == (N, N_ITERS, 2) and losses.shape == (N, N_ITERS)
    assert min(float(loss_fn(r, scene)) for r in final) < 1e-3


# ----- solve --------------------------------------------------------------

def test_solve_ignores_a_wall_that_is_not_in_the_way():
    wall_off = (41.0, 2.0, 12.0, 0.0)
    f_off, hist, losses = solve(make_scene((84.0, 15.5), wall=wall_off))
    assert hist.shape == (N, 2 * N_ITERS, 2) and losses.shape == (N, 2 * N_ITERS)
    f_low, _, _ = solve(make_scene((84.0, 15.5), wall=(41.0, 2.0, 12.0, 1.0)))
    assert jnp.allclose(f_off, f_low, atol=1e-4)


def test_solve_clears_a_blocking_wall():
    scene = make_scene((60.0, 5.0), wall=WALL)
    assert all(hits(solve(scene)[0], scene))


def test_solve_hits_a_moving_target():
    scene = make_scene((50.0, 5.0), target_vel=(-5.0, 2.0))
    assert any(hits(solve(scene)[0], scene))


def test_solve_lobs_a_wall_in_front_of_a_receding_target():
    scene = make_scene((70.0, 5.0), target_vel=(8.0, 0.0), wall=WALL)
    assert any(hits(solve(scene)[0], scene))


def test_solve_hits_with_drag():
    scene = make_scene((50.0, 5.0), drag=0.015)
    assert any(hits(solve(scene)[0], scene))


def test_changing_the_scene_does_not_recompile():
    solve(make_scene((40.0, 5.0)))
    before = solve._cache_size()
    solve(make_scene((70.0, 9.0), (4.0, -1.0), (50.0, 2.0, 8.0, 1.0), 0.015))
    assert solve._cache_size() == before


def test_choose_arc_prefers_the_cheapest_hit():
    hit, miss = dict(hit=True, blocked=False, miss=0.1), dict(hit=False, blocked=False, miss=0.5)
    assert choose_arc([hit, hit], [30.0, 20.0]) == 1
    assert choose_arc([miss, hit], [10.0, 40.0]) == 1
