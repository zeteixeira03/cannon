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


def step(state, _):
    """Advance ``(pos, vel)`` by one timestep; emits the new position.

    Uses the constant-acceleration kinematic update, which is exact under gravity alone.
    Signature matches ``jax.lax.scan``.
    """
    pos, vel = state
    g = jnp.array([0.0, -GRAVITY])
    new_pos = pos + vel * DT + 0.5 * g * DT**2
    new_vel = vel + g * DT
    return (new_pos, new_vel), new_pos


def simulate(raw):
    """Ball positions after each timestep, shape ``(N_STEPS, 2)``.

    The flight is not cut at the ground or the wall: a fixed-length output keeps the
    function jit-friendly, and the losses decide what counts.
    """
    _, traj = jax.lax.scan(step, initial_state(raw), xs=None, length=N_STEPS)
    return traj
