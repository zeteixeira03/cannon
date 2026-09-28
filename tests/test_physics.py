import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cannon.physics import (DT, GRAVITY, MAX_ANGLE, MAX_SPEED, N_STEPS, acceleration, launch_to_raw, simulate,
                            to_launch)


def reference_flight(angle, speed, drag, t_end, dt=1e-4):
    """Position at ``t_end`` from classic RK4 with a tiny timestep, in float64 NumPy."""
    g = np.array([0.0, -GRAVITY])
    acc = lambda v: g - drag * np.linalg.norm(v) * v
    p, v = np.zeros(2), speed * np.array([np.cos(angle), np.sin(angle)])
    for _ in range(int(round(t_end / dt))):
        k1 = acc(v)
        k2 = acc(v + 0.5 * dt * k1)
        k3 = acc(v + 0.5 * dt * k2)
        k4 = acc(v + dt * k3)
        p = p + dt / 6 * (v + 2 * (v + 0.5 * dt * k1) + 2 * (v + 0.5 * dt * k2) + (v + dt * k3))
        v = v + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return p


def test_to_launch_centre_and_bounds():
    a, s = to_launch(jnp.array([0.0, 0.0]))
    assert jnp.allclose(a, MAX_ANGLE / 2) and jnp.allclose(s, MAX_SPEED / 2)
    a, s = to_launch(jnp.array([-50.0, 50.0]))
    assert 0 <= a < 1e-3 and s <= MAX_SPEED


def test_launch_to_raw_inverts_to_launch():
    a, s = to_launch(launch_to_raw(0.7, 30.0))
    assert jnp.allclose(a, 0.7, atol=1e-4) and jnp.allclose(s, 30.0, atol=1e-3)


def test_acceleration():
    assert jnp.allclose(acceleration(jnp.array([3.0, 4.0]), 0.0), jnp.array([0.0, -GRAVITY]))
    # |v| = 5, so drag adds -0.1 * 5 * v
    assert jnp.allclose(acceleration(jnp.array([3.0, 4.0]), 0.1), jnp.array([-1.5, -GRAVITY - 2.0]))


def test_range_matches_analytic_formula():
    traj = simulate(launch_to_raw(jnp.pi / 4, 20.0))
    assert traj.shape == (N_STEPS, 2)
    landing = int(np.argmax(np.asarray(traj[:, 1]) < 0))
    assert abs(float(traj[landing, 0]) - 20.0**2 / GRAVITY) < 1.0   # v² sin(2θ) / g


@pytest.mark.parametrize("angle_deg", [30, 60, 80])
@pytest.mark.parametrize("drag", [0.005, 0.02])
def test_drag_matches_fine_timestep_reference(angle_deg, drag):
    # Forward Euler is off by 0.2-0.6 m here after 4 s; a second-order step is off by < 3 mm.
    angle, speed, i = np.radians(angle_deg), 40.0, 199          # row 199 is t = 4 s
    traj = simulate(launch_to_raw(angle, speed), drag)
    expected = reference_flight(angle, speed, drag, (i + 1) * DT)
    assert np.linalg.norm(np.asarray(traj[i]) - expected) < 1e-2


def test_drag_shortens_range_and_approaches_terminal_speed():
    raw = launch_to_raw(np.radians(80), 40.0)
    free, dragged = simulate(raw, 0.0), simulate(raw, 0.02)
    assert float(jnp.max(dragged[:, 1])) < 0.6 * float(jnp.max(free[:, 1]))
    fall_speed = float(jnp.linalg.norm(dragged[-1] - dragged[-2])) / DT
    assert fall_speed == pytest.approx(np.sqrt(GRAVITY / 0.02), rel=0.05)


def test_drag_is_differentiable():
    g = jax.grad(lambda raw, k: simulate(raw, k)[150, 0], argnums=(0, 1))(launch_to_raw(0.6, 30.0), 0.01)
    assert all(jnp.all(jnp.isfinite(x)) for x in g)
    assert g[1] < 0                                             # more drag -> less distance
