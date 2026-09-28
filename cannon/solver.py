"""Aiming as optimisation: a differentiable loss, Adam, and a batched, compiled solve.

Everything the player can change lives in a :class:`Scene`, a pytree of arrays. The
solver is compiled once; changing the target, the wall or the drag changes data, not
the program, so it never recompiles.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from cannon.physics import DT, MAX_SPEED, N_STEPS, launch_to_raw, simulate

N_ITERS = 300           # Adam steps per phase
LEARNING_RATE = 0.1
WALL_WEIGHT = 10.0      # wall penalty vs miss distance
WALL_SMOOTH = 0.3       # m; softness of the wall's sides, so the penalty has useful gradients
BLOCKED_TOL = 1e-6      # wall penalty above which an arc counts as going through the wall

DEFAULT_WALL = (30.0, 2.0, 15.0, 0.0)

# Starting guesses as (angle in degrees, fraction of MAX_SPEED): one shallow, one steep,
# so the batch tends to find both arcs through the target.
INITIAL_GUESSES = [(25.0, 0.6), (70.0, 0.6)]


class Scene(NamedTuple):
    """Everything the solver aims against. Build it with :func:`make_scene`."""
    target: jax.Array       # (2,)  m, position at launch
    target_vel: jax.Array   # (2,)  m/s, constant
    wall: jax.Array         # (4,)  [x_center, thickness, height, enabled]; enabled is 0.0 or 1.0
    drag: jax.Array         # ()    1/m, see physics.acceleration


def make_scene(target, target_vel=(0.0, 0.0), wall=DEFAULT_WALL, drag=0.0):
    """A :class:`Scene` with every field as float32.

    Fixed dtypes and shapes matter: ``jit`` recompiles when they change, e.g. a Python
    float in one call and a float32 array in the next.
    """
    f32 = lambda x: jnp.asarray(x, jnp.float32)
    return Scene(f32(target), f32(target_vel), f32(wall), f32(drag))


def target_path(scene):
    """Target position at each simulation time, aligned row by row with :func:`simulate`."""
    t = DT * jnp.arange(1, N_STEPS + 1)
    return scene.target + t[:, None] * scene.target_vel


def initial_raw_batch():
    """``INITIAL_GUESSES`` as raw params, shape ``(n_guesses, 2)``."""
    return jnp.stack([launch_to_raw(jnp.deg2rad(a), f * MAX_SPEED) for a, f in INITIAL_GUESSES])


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def point_segment_sq_dist(a, b, p):
    """Squared distance from point ``p`` to segment ``a -> b`` (all shape (2,))."""
    ab, ap = b - a, p - a
    t = jnp.clip(jnp.dot(ap, ab) / (jnp.dot(ab, ab) + 1e-9), 0.0, 1.0)  # eps: zero-length segments
    return jnp.sum((p - (a + t * ab)) ** 2)


def closest_approach(traj, target_traj):
    """Closest the ball gets to the target in space AND time: ``(squared_distance, index)``.

    ``index`` is the timestep where the closest segment starts.
    """
    # TODO(you): work in the target's frame. Within one timestep both the ball and the target move
    # in a straight line, so the ball's position RELATIVE to the target does too: the relative
    # path is a polyline, and a hit means it passes through the origin. point_segment_sq_dist
    # already measures that. A static target must give the same answer as the old miss loss.
    # Tests: tests/test_solver.py -k closest_approach
    raise NotImplementedError


def wall_penalty(traj, wall, until):
    """How deep the flight up to timestep ``until`` goes through the wall: ``Σ inside·depth²``.

    ``inside`` is a smooth 0..1 mask over the wall's width, so the gradient points out of the wall
    (a hard mask would have zero gradient). Points after ``until`` (the impact) don't count, and
    neither does a disabled wall.
    """
    x_center, thickness, height, enabled = wall
    x, y = traj[:, 0], traj[:, 1]
    left, right = x_center - thickness / 2, x_center + thickness / 2
    inside = jax.nn.sigmoid((x - left) / WALL_SMOOTH) * jax.nn.sigmoid((right - x) / WALL_SMOOTH)
    depth = jax.nn.relu(height - y)
    # TODO(you): only timesteps 0..until count. `until` is a traced integer, so no slicing
    # (traj[:until] needs a concrete length): build a 0/1 mask instead.
    # Tests: tests/test_solver.py -k wall
    raise NotImplementedError


def shot_terms(raw, scene):
    """``(miss, wall)`` loss terms for launch params ``raw``."""
    traj = simulate(raw, scene.drag)
    miss, impact = closest_approach(traj, target_path(scene))
    return miss, wall_penalty(traj, scene.wall, impact)


def loss_fn(raw, scene):
    miss, wall = shot_terms(raw, scene)
    return miss + WALL_WEIGHT * wall


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------
# Angle and speed have very different gradient scales; Adam normalises each
# parameter's step by its own running gradient size, so both converge together.

def adam_init(raw):
    """Optimiser state ``(m, v, t)``: first and second moment estimates and step count."""
    return jnp.zeros_like(raw), jnp.zeros_like(raw), 0


def adam_update(raw, grads, opt_state, lr=LEARNING_RATE, b1=0.9, b2=0.999, eps=1e-8):
    """One Adam step. Returns ``(new_raw, new_opt_state)``."""
    m, v, t = opt_state
    t = t + 1
    m = b1 * m + (1 - b1) * grads
    v = b2 * v + (1 - b2) * grads**2
    m_hat = m / (1 - b1**t)            # bias correction: m and v start at 0
    v_hat = v / (1 - b2**t)
    return raw - lr * m_hat / (jnp.sqrt(v_hat) + eps), (m, v, t)


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

def optimise(raw0, scene):
    """``N_ITERS`` Adam steps on ``loss_fn`` from ``raw0``, as a single ``lax.scan``.

    Returns ``(final_raw, raw_history, loss_history)`` with shapes (2,), (N_ITERS, 2), (N_ITERS,).
    """
    loss_and_grad = jax.value_and_grad(loss_fn)

    def body(carry, _):
        raw, opt_state = carry
        loss, grads = loss_and_grad(raw, scene)
        raw, opt_state = adam_update(raw, grads, opt_state)
        return (raw, opt_state), (raw, loss)

    (final_raw, _), (raw_history, loss_history) = jax.lax.scan(
        body, (raw0, adam_init(raw0)), xs=None, length=N_ITERS)
    return final_raw, raw_history, loss_history


@jax.jit
def optimise_batch(raw0_batch, scene):
    """:func:`optimise` from each row of ``raw0_batch``; every output gains a leading batch axis."""
    return jax.vmap(optimise, in_axes=(0, None))(raw0_batch, scene)


@jax.jit
def solve(raw0_batch, scene):
    """Find launch params that hit the target from each starting guess, avoiding the wall.

    Two phases, so the wall only changes arcs it actually blocks:
      1. optimise with the wall off -> the arcs the target alone would give;
      2. for arcs that go through the wall, keep optimising with it on, warm-started
         from phase 1 (the arc already hits; it only needs lifting over).
    A single phase with the wall on is worse on both counts: starting guesses that fly
    through the wall get bent even when the final arc clears it, and guesses that start
    deep inside it often never escape.

    Returns ``(final_raw, raw_history, loss_history)`` with shapes
    (n, 2), (n, 2 * N_ITERS, 2), (n, 2 * N_ITERS). Unblocked arcs repeat their phase-1
    result through phase 2.
    """
    free_final, free_hist, free_loss = optimise_batch(raw0_batch, scene._replace(wall=scene.wall.at[3].set(0.0)))

    _, penalty = jax.vmap(shot_terms, in_axes=(0, None))(free_final, scene)
    blocked = penalty > BLOCKED_TOL

    # Under jit and vmap there is no per-arc branching: phase 2 runs for every arc and
    # jnp.where selects the result.
    wall_final, wall_hist, wall_loss = optimise_batch(free_final, scene)
    final = jnp.where(blocked[:, None], wall_final, free_final)
    phase2_hist = jnp.where(blocked[:, None, None], wall_hist, free_final[:, None, :])
    phase2_loss = jnp.where(blocked[:, None], wall_loss, free_loss[:, -1:])

    return (final,
            jnp.concatenate([free_hist, phase2_hist], axis=1),
            jnp.concatenate([free_loss, phase2_loss], axis=1))
