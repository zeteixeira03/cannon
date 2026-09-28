import jax.numpy as jnp
import numpy as np

from cannon.physics import GRAVITY, MAX_ANGLE, MAX_SPEED, N_STEPS, launch_to_raw, simulate, to_launch


def test_to_launch_centre_and_bounds():
    a, s = to_launch(jnp.array([0.0, 0.0]))
    assert jnp.allclose(a, MAX_ANGLE / 2) and jnp.allclose(s, MAX_SPEED / 2)
    a, s = to_launch(jnp.array([-50.0, 50.0]))
    assert 0 <= a < 1e-3 and s <= MAX_SPEED


def test_launch_to_raw_inverts_to_launch():
    a, s = to_launch(launch_to_raw(0.7, 30.0))
    assert jnp.allclose(a, 0.7, atol=1e-4) and jnp.allclose(s, 30.0, atol=1e-3)


def test_range_matches_analytic_formula():
    traj = simulate(launch_to_raw(jnp.pi / 4, 20.0))
    assert traj.shape == (N_STEPS, 2)
    landing = int(np.argmax(np.asarray(traj[:, 1]) < 0))
    assert abs(float(traj[landing, 0]) - 20.0**2 / GRAVITY) < 1.0   # v² sin(2θ) / g
