"""Projectile physics: launch parametrisation and trajectory simulation.

Everything here is a pure JAX function of its inputs, so it can be jit-compiled,
vmapped and differentiated end to end.
"""

import jax
import jax.numpy as jnp

GRAVITY = 9.81          # m/s²
DT = 0.02               # s, integration timestep
N_STEPS = 400           # timesteps per flight (400 x 0.02s = 8 s)
CANNON_POS = (0.0, 0.0)

MAX_SPEED = 100.0       # m/s; reaches the edge of the view under heavy drag
MAX_ANGLE = jnp.pi / 2  # rad


def to_launch(raw):
    """Map unconstrained params ``raw = [raw_angle, raw_speed]`` to physical ``(angle, speed)``.

    A sigmoid squashes each value into ``(0, MAX_ANGLE)`` and ``(0, MAX_SPEED)``, so the
    optimiser can move ``raw`` freely without ever producing invalid physics.
    """
    raw_angle, raw_speed = raw
    return MAX_ANGLE * jax.nn.sigmoid(raw_angle), MAX_SPEED * jax.nn.sigmoid(raw_speed)


def launch_to_raw(angle, speed):
    """Inverse of :func:`to_launch`: physical ``(angle, speed)`` -> ``raw`` of shape (2,)."""
    def logit(p):
        return jnp.log(p / (1 - p))

    return jnp.array([logit(angle / MAX_ANGLE), logit(speed / MAX_SPEED)])


def initial_state(raw):
    """``(pos, vel)`` at ``t = 0`` for launch params ``raw``."""
    angle, speed = to_launch(raw)
    pos = jnp.array(CANNON_POS)
    vel = speed * jnp.array([jnp.cos(angle), jnp.sin(angle)])
    return pos, vel


def acceleration(vel, drag):
    """Gravity plus quadratic air drag: ``a = g - drag * |v| * v``.

    ``drag`` is ``k = ρ·C_d·A / (2m)`` in 1/m; terminal speed is ``sqrt(GRAVITY / k)``.
    """
    speed = jnp.linalg.norm(vel)
    air_acc = -drag * speed * vel
    return jnp.array([0.0, -GRAVITY]) + air_acc


def step(state, drag):
    """Advance ``(pos, vel)`` by one timestep of ``DT`` (RK2 midpoint)."""
    pos, vel = state
    vel_mid = vel + 0.5 * DT * acceleration(vel, drag)
    return pos + DT * vel_mid, vel + DT * acceleration(vel_mid, drag)


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
