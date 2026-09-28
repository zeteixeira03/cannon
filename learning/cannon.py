"""
Auto-Aiming Cannon — differentiable physics with JAX.

Goal: simulate a 2D projectile, then use gradient descent to find the launch
angle and speed that make the cannonball hit a target.

How to use this file:
  - Work through the TODOs in order (Stage 1 -> Stage 5).
  - Every stub raises NotImplementedError until you fill it in.
  - Run `python cannon.py` often. `main()` runs each stage's sanity check,
    so you get feedback as soon as a stage is done.

JAX concepts you'll touch:
  jnp (NumPy-like arrays), pure functions, jax.grad / jax.value_and_grad,
  jax.jit, jax.lax.scan, pytrees (jax.tree_util.tree_map), and jax.vmap (stretch).
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRAVITY = 9.81          # m/s^2
DT = 0.01               # timestep (s)
N_STEPS = 500           # total time = N_STEPS * DT
CANNON_POS = (0.0, 0.0)
TARGET = (40.0, 5.0)    

LEARNING_RATE = 1e-3
N_ITERATIONS = 1_000


# ---------------------------------------------------------------------------
# Stage 1 — Parameters
# ---------------------------------------------------------------------------
# The things we will *learn* are the launch angle and launch speed.
# Store them in a pytree (a dict works nicely) so JAX can differentiate w.r.t.
# all of them at once, e.g. {"angle": ..., "speed": ...}.
#
# Think about: radians vs degrees; should the values be floats (grad needs
# floats, not ints)? Should you optimise the raw values, or an unconstrained
# version that you map into a valid range (e.g. speed > 0)?

def init_params():
    """Return the initial guess for the launch parameters as a pytree."""
    return {"angle": 0.8, "speed": 10.0}

def pythagoras(r1, r2):
    """Return the norm distance between 2 points using the Pythagorean Theorem
        We drop the sqrt to avoid NaNs later"""
    return (r1[0] - r2[0])**2 + (r1[1] - r2[1])**2


# ---------------------------------------------------------------------------
# Stage 2 — Physics simulation
# ---------------------------------------------------------------------------
# State of the projectile at one moment: position (x, y) and velocity (vx, vy).
# Pick a representation: a tuple (pos, vel) of 2-vectors, or one 4-vector.

def initial_state(params):
    """
    Convert launch params into the projectile's starting state.

    Returns: the state at t=0 (position = CANNON_POS, velocity from angle/speed).

    params: {"angle": radians, "speed": m/s}
    """
    pos = CANNON_POS            # projectile fires from the cannon
    vel = vel = (params["speed"] * jnp.cos(params["angle"]),
       params["speed"] * jnp.sin(params["angle"]))          # with a certain initial speed

    return (jnp.array(pos), jnp.array(vel))


def step(state, _):
    """
    Advance the simulation by one timestep DT.

    Signature is shaped for `jax.lax.scan`: it must return (new_carry, output).
    The carry is the state; the output is whatever you want recorded per step
    (the position is a good choice, for plotting later).

    Hint: explicit Euler integration is fine to start with.
    Must be pure: no Python side effects, no in-place mutation.
    """
    pos, vel = state
    # acceleration vector
    g = jnp.array([0.0, GRAVITY])

    pos = pos + vel * DT - 0.5 * g * DT**2
    vel = vel - g * DT

    return (pos, vel), pos


def simulate(params):
    """
    Run the full simulation for N_STEPS.

    Returns: the trajectory, an array of positions with shape (N_STEPS, 2).

    Hint: a Python for-loop works, but `jax.lax.scan` is the JAX way — it
    compiles to a single loop instead of unrolling N_STEPS copies under jit.
    """
    state = initial_state(params)
    # pass step without () because we're adding the function itself as an
    # input to iterate through N_STEPS time steps
    _, traj = jax.lax.scan(step, state, xs=None, length=N_STEPS)

    return traj

# ---------------------------------------------------------------------------
# Stage 3 — Loss function
# ---------------------------------------------------------------------------
# The loss must be a single scalar that is small when we hit the target.
#
# This is the most interesting design decision in the project. Some options:
#   a) distance from target at a fixed time
#   b) the closest the trajectory ever gets to the target
#   c) where the ball lands (crosses the target's height) vs target x
# Think about which ones are *differentiable* everywhere. Hard `min` / `argmin`
# / boolean masks have gradients, but maybe not useful ones. Look up "softmin"
# (jax.nn.softmax / jax.scipy.special.logsumexp) if you go for (b).

def loss_fn(params, target):
    """Scalar loss: how badly does a shot with `params` miss `target`?"""
    traj = simulate(params)
    return jnp.min(jax.vmap(pythagoras, in_axes=(0, None))(traj, target))


# ---------------------------------------------------------------------------
# Stage 4 — Gradient descent
# ---------------------------------------------------------------------------
@jax.jit
def update(params, target, lr):
    """
    One optimisation step.

    Returns: (new_params, loss_value)

    Hints:
      - jax.value_and_grad gives you the loss and the gradient in one call.
      - The gradient has the same pytree structure as `params`.
      - jax.tree_util.tree_map lets you apply `p - lr * g` to every leaf.
      - Decorate this (or wrap it) with jax.jit once it works. Notice the
        speed difference! Why does the first call take longer?
    """
    f = jax.value_and_grad(loss_fn)    # f takes the same arguments as loss_fn and evaluates
                                    # both f and grad(f) w.r.t. params as a tuple
    value, grads = f(params, target)

    # no JAX:
    #{"angle": params["angle"] - lr * grads["angle"],
    #"speed": params["speed"] - lr * grads["speed"]}

    # yes JAX:
    new_params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    # updated parameters, and value of loss function that was used to get to these.
    # so value relates to the previous set of parameters
    return new_params, value


def train(params, target, lr=LEARNING_RATE, n_iterations=N_ITERATIONS):
    """
    Run the optimisation loop.

    Returns: (final_params, loss_history) where loss_history is a list/array.
    Print progress every so often so you can see it converge (or explode —
    if it explodes, look at your learning rate and the gradient magnitudes).
    """
    loss_history = []
    best_loss = float("inf")

    for n in range(n_iterations):
        curr_params = params            # arrays are immutable in JAX 
        params, new_value = update(params, target, lr)
        loss_history.append(new_value)

        if new_value < best_loss:
            final_params = curr_params
            best_loss = new_value

        if n % 100 == 0: print(n, best_loss)

    return final_params, loss_history


# ---------------------------------------------------------------------------
# Stage 5 — Visualisation
# ---------------------------------------------------------------------------

def plot_results(initial_params, final_params, target, loss_history):
    """
    Show two plots:
      1. The trajectory before and after training, plus the target point.
         (Clip or mask the part of the trajectory below ground for a nicer plot.)
      2. Loss vs iteration (a log-scale y axis usually looks better).
    """
    fig, (ax_traj, ax_loss) = plt.subplots(1, 2, figsize=(12, 5))

    for params, label, style in [(initial_params, "initial guess", "--"),
                                 (final_params, "trained", "-")]:
        traj = simulate(params)
        traj = traj[traj[:, 1] >= 0]    # drop points below ground
        ax_traj.plot(traj[:, 0], traj[:, 1], style, label=label)

    ax_traj.scatter(target[0], target[1], color="red", marker="x", s=100, zorder=3, label="target")
    ax_traj.set_xlabel("x (m)")
    ax_traj.set_ylabel("y (m)")
    ax_traj.set_title("Trajectory")
    ax_traj.set_aspect("equal")
    ax_traj.legend()

    ax_loss.plot(loss_history)
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("iteration")
    ax_loss.set_ylabel("loss (squared distance, m²)")
    ax_loss.set_title("Loss")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Sanity checks — these are NOT answers, just tests your code should pass.
# ---------------------------------------------------------------------------

def check_stage_2():
    params = init_params()
    traj = simulate(params)
    assert traj.shape == (N_STEPS, 2), f"trajectory shape is {traj.shape}"
    # Compare against the analytic range on flat ground: R = v^2 * sin(2θ) / g.
    # Your simulated landing x should be close (Euler error shrinks as DT -> 0).
    print("Stage 2 OK: trajectory shape", traj.shape)


def check_stage_3():
    params = init_params()
    loss = loss_fn(params, jnp.array(TARGET))
    assert jnp.ndim(loss) == 0, "loss must be a scalar"
    grads = jax.grad(loss_fn)(params, jnp.array(TARGET))
    finite = jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda g: bool(jnp.all(jnp.isfinite(g))), grads)
    )
    assert finite, f"non-finite gradient: {grads}"
    print(f"Stage 3 OK: loss={float(loss):.4f}, grads={grads}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    target = jnp.array(TARGET)

    check_stage_2()
    check_stage_3()

    initial_params = init_params()
    final_params, loss_history = train(initial_params, target)
    print("Final params:", final_params)

    plot_results(initial_params, final_params, target, loss_history)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Stretch goals (once everything above works)
# ---------------------------------------------------------------------------
# 1. Air drag: add a force proportional to -|v| * v. The analytic formula stops
#    working, but gradient descent doesn't care — that's the point of JAX.
# 2. Many targets at once: use jax.vmap over `train` or `update` to aim at a
#    batch of targets in parallel.
# 3. Fix the angle, learn only the speed (or vice versa). Use `argnums` or
#    split the params pytree.
# 4. Add a wall between cannon and target that the ball must clear — add a
#    penalty term to the loss.
# 5. Swap your hand-written update for an optimiser from `optax` (e.g. Adam).
# 6. Inspect the compiled code: jax.make_jaxpr(loss_fn)(params, target).
# 7. There are usually TWO angles that hit a target. Can you find both by
#    starting from different initial guesses?
