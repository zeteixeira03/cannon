"""Projectile physics: launch parametrisation and trajectory simulation.

Everything here is a pure JAX function of its inputs, so it can be jit-compiled,
vmapped and differentiated end to end.
"""

import jax
import jax.numpy as jnp

GRAVITY = 9.81          # m/s²
DT = 0.02               # s, integration timestep
N_STEPS = 400           # timesteps per flight (8 s)
CANNON_POS = (0.0, 0.0)

MAX_SPEED = 45.0        # m/s
MAX_ANGLE = jnp.pi / 2  # rad


def to_launch(raw):
    """Map unconstrained params ``raw = [raw_angle, raw_speed]`` to physical ``(angle, speed)``.

    A sigmoid squashes each value into ``(0, MAX_ANGLE)`` and ``(0, MAX_SPEED)``, so the
    optimiser can move ``raw`` freely without ever producing an invalid launch.
    """
    raw_angle, raw_speed = raw
    return MAX_ANGLE * jax.nn.sigmoid(raw_angle), MAX_SPEED * jax.nn.sigmoid(raw_speed)


def launch_to_raw(angle, speed):
    """Inverse of :func:`to_launch`: physical ``(angle, speed)`` -> ``raw`` of shape (2,)."""
    def logit(p):
        return jnp.log(p / (1 - p))

    return jnp.array([logit(angle / MAX_ANGLE), logit(speed / MAX_SPEED)])


def initial_state(raw):
    """``(pos, vel)`` at the muzzle for launch params ``raw``."""
    angle, speed = to_launch(raw)
    pos = jnp.array(CANNON_POS)
    vel = speed * jnp.array([jnp.cos(angle), jnp.sin(angle)])
    return pos, vel


def acceleration(vel, drag):
    """Gravity plus quadratic air drag: ``a = g - drag * |v| * v``.

    ``drag`` is ``k = ρ·C_d·A / (2m)`` in 1/m; terminal speed is ``sqrt(GRAVITY / k)``.
    """
    # TODO(you): tests/test_physics.py::test_acceleration
    raise NotImplementedError


def step(state, drag):
    """Advance ``(pos, vel)`` by one timestep of ``DT`` under :func:`acceleration`."""
    # TODO(you): with drag the acceleration depends on velocity, so the old constant-acceleration
    # update is no longer exact. Forward Euler is off by 0.2-0.6 m after 4 s; the test wants < 1 cm.
    # Pick a higher-order scheme (midpoint/RK2 or RK4). With drag = 0 it must still give the
    # analytic range. Tests: tests/test_physics.py -k "range or drag"
    raise NotImplementedError


def simulate(raw, drag=0.0):
    """Ball positions after each timestep, shape ``(N_STEPS, 2)``; row ``i`` is time ``(i + 1) * DT``.

    The flight is not cut at the ground or the wall: a fixed-length output keeps the
    function jit-friendly, and the losses decide what counts.
    """
    def body(state, _):
        state = step(state, drag)
        return state, state[0]

    _, traj = jax.lax.scan(body, initial_state(raw), xs=None, length=N_STEPS)
    return traj
