"""
Cannon Game — an interactive, auto-aiming cannon powered by JAX.

    python cannon_game.py

Controls (once the window is open):
    left-drag       move the target; the cannon fires when you release
    right-drag      move the wall (x = position, y = height)
    w               toggle the wall on/off
    p               toggle live aiming preview while dragging
    r               toggle the "replay the optimisation" animation

What happens on every shot:
    1. JAX solves for the launch angle + speed by gradient descent, starting
       from TWO initial guesses at once (a shallow one and a steep one) via vmap.
       Many (angle, speed) pairs hit the same target, so the two guesses can
       end up on different arcs — or on nearly the same one.
    2. The whole optimisation loop is ONE jit-compiled function (lax.scan over
       iterations), so it takes milliseconds.
    3. The window replays how the arcs bent towards the target, then fires.

describes the logic step by step. The UI PART at the bottom is done; you
don't need to touch it (but read it if you're curious how it calls you).

Run the file after each function: the startup checks tell you which stage is
next and whether what you wrote behaves correctly. The window only opens once
all checks pass.

A lot of this is a remix of cannon.py. The new JAX ideas are marked NEW.
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# World constants
# ---------------------------------------------------------------------------
GRAVITY = 9.81
DT = 0.02               # timestep (s)
N_STEPS = 400           # 8 s of flight
CANNON_POS = (0.0, 0.0)

MAX_SPEED = 45.0        # m/s 
MAX_ANGLE = jnp.pi / 2  # rad  

N_ITERS = 300           # optimisation steps per shot
LEARNING_RATE = 0.1 
SPEED_WEIGHT = 0 # how much the speed penalty counts vs the miss distance    
WALL_WEIGHT = 10.0      # how much the wall penalty counts vs the miss distance
WALL_SMOOTH = 0.3       # (m) how soft the wall's edges are for the gradient
                        # this exists because the wall can't actually be sharp 
                        # as it introduces discontinuities in the loss function

# Wall is a single float32 array: [x_center, thickness, height, enabled]
# enabled is 1.0 or 0.0 — a float, NOT a Python bool. Why? See Stage 4.
DEFAULT_WALL = jnp.array([30.0, 2.0, 15.0, 0.0])

# Starting guesses, in (angle_degrees, fraction_of_max_speed). One shallow, one steep.
INITIAL_GUESSES = [(25.0, 0.6), (70.0, 0.6)]


# ===========================================================================
#                                JAX PART
# ===========================================================================

# ---------------------------------------------------------------------------
# Stage 1 — Parameters that can't go out of bounds   (NEW: reparametrisation)
# ---------------------------------------------------------------------------
# In cannon.py you optimised angle and speed directly. Nothing stopped speed
# from going negative or to c. Here we optimise two *unconstrained* numbers
#     raw = jnp.array([raw_angle, raw_speed])      (any real values)
# and squash them into the valid range every time we use them.
# This time params are a plain array of shape (2,), not a dict — it makes
# vmap and Adam simpler.

def to_launch(raw):
    """
    Map raw (unconstrained) params to physical (angle, speed).

    Logic:
      - sigmoid(z) = 1 / (1 + e^-z) maps any real number into (0, 1).
        JAX has it: jax.nn.sigmoid.
      - angle = MAX_ANGLE * sigmoid(raw_angle)   -> always in (0, 90°)
      - speed = MAX_SPEED * sigmoid(raw_speed)   -> always in (0, MAX_SPEED)
      - return (angle, speed) as a tuple of two scalars
    """
    raw_angle, raw_speed = raw
    angle = MAX_ANGLE * jax.nn.sigmoid(raw_angle)
    speed = MAX_SPEED * jax.nn.sigmoid(raw_speed)

    return angle, speed


def launch_to_raw(angle, speed):
    """
    The exact inverse of to_launch: physical (angle, speed) -> raw array.
    Used to turn INITIAL_GUESSES into raw starting points.

    Logic:
      - the inverse of sigmoid is logit(p) = log(p / (1 - p))
      - p_angle = angle / MAX_ANGLE, p_speed = speed / MAX_SPEED
      - return jnp.array([logit(p_angle), logit(p_speed)])
    Check: to_launch(launch_to_raw(a, s)) should give back (a, s).
    """
    def logit(p):
        return jnp.log(p / (1 - p))

    p_angle = angle / MAX_ANGLE
    p_speed = speed / MAX_SPEED

    return jnp.array([logit(p_angle), logit(p_speed)])


# ---------------------------------------------------------------------------
# Stage 2 — Physics (port from cannon.py)
# ---------------------------------------------------------------------------

def initial_state(raw):
    """
    Logic:
      - (angle, speed) = to_launch(raw)
      - pos = CANNON_POS as an array
      - vel = speed * [cos(angle), sin(angle)]
      - return (pos, vel)
    """
    (angle, speed) = to_launch(raw)
    pos = jnp.array(CANNON_POS)
    vel = speed * jnp.array([jnp.cos(angle), jnp.sin(angle)])

    return pos, vel


def step(state, _):
    """
    Same as cannon.py: one Euler step, returns ((new_pos, new_vel), new_pos).
    """
    pos, vel = state
    # acceleration vector
    g = jnp.array([0.0, -GRAVITY])  # gravity points down

    new_pos = pos + vel * DT + 0.5 * g * DT**2
    new_vel = vel + g * DT

    return (new_pos, new_vel), new_pos

def simulate(raw):
    """
    Same as cannon.py: lax.scan `step` for N_STEPS, return positions (N_STEPS, 2).
    """
    state = initial_state(raw)
    # pass step without () because we're adding the function itself as an
    # input to iterate through N_STEPS time steps
    _, traj = jax.lax.scan(step, state, xs=None, length=N_STEPS)

    return traj


# ---------------------------------------------------------------------------
# Stage 3 — A miss distance without the "dots" problem   (NEW: vmap over segments)
# ---------------------------------------------------------------------------
# In cannon.py the loss got stuck around 1e-3 because the target sits BETWEEN
# trajectory dots. Fix: treat the trajectory as a chain of line segments
# (dot i -> dot i+1) and measure the distance to the closest SEGMENT.

def point_segment_sq_dist(a, b, p):
    """
    Squared distance from point p to the line segment a->b. All inputs shape (2,).

    Logic:
      - ab = b - a,  ap = p - a
      - t = dot(ap, ab) / dot(ab, ab)          how far along the segment the
                                               projection of p lands (0=a, 1=b)
        (add a tiny epsilon like 1e-9 to the denominator: a zero-length
         segment would divide by zero -> nan)
      - clamp t into [0, 1] (jnp.clip) so the closest point stays ON the segment
      - closest = a + t * ab
      - return the squared length of (p - closest)
    Everything here is written for ONE segment. vmap does the rest.
    """
    ab = b - a
    ap = p - a
    eps = 1e-9

    t = jnp.clip(jnp.dot(ap, ab) / (jnp.dot(ab, ab) + eps), 0.0, 1.0)
    closest = a + t * ab

    return jnp.sum((p - closest)**2)


def miss_loss(traj, target):
    """
    Smallest squared distance from target to any segment of the trajectory.

    Logic:
      - starts = traj[:-1], ends = traj[1:]   (each shape (N_STEPS-1, 2))
      - vmap point_segment_sq_dist over starts and ends together (axis 0),
        with the target NOT mapped (None)            -> shape (N_STEPS-1,)
      - return the min
    """
    # traj = [d0, d1, d2, ..., d398, d399]
    # starts = [d0, d1, d2, ..., d398]
    # ends = [d1, d2, d3, ..., d399]
    starts = traj[:-1]
    ends = traj[1:]

    vmapped_dist = jax.vmap(point_segment_sq_dist, in_axes=(0, 0, None))    # batchify point_segment_sq_dist
    sq_distances = vmapped_dist(starts, ends, target)                       # execute over all segments at once

    return jnp.min(sq_distances)


# ---------------------------------------------------------------------------
# Stage 4 — The wall   (NEW: smooth masks, no Python `if` on traced values)
# ---------------------------------------------------------------------------
# Inside jit, values are *tracers* — placeholders with no concrete number yet.
# `if enabled:` or `if x < wall_x:` on a tracer raises an error, because
# Python needs a real True/False right now. Instead, express conditions as
# arithmetic: multiply by 0.0 or 1.0, or use smooth 0..1 weights.
# That's also why the wall's `enabled` flag is a float in an array:
# toggling it changes a VALUE, so the compiled function is reused.
# (If it were a Python bool argument, JAX would have to recompile.)

def wall_penalty(traj, target, wall):
    """
    How much does the trajectory go THROUGH the wall? 0 if it clears it.

    Logic:
      - unpack wall into x_center, thickness, height, enabled
      - x, y = the trajectory's columns
      - left  = x_center - thickness/2,  right = x_center + thickness/2
      - "inside" weight per point, smoothly ~1 inside [left, right], ~0 outside:
            sigmoid((x - left) / WALL_SMOOTH) * sigmoid((right - x) / WALL_SMOOTH)
        (a hard `(x > left) & (x < right)` mask has zero gradient w.r.t. x —
         the smooth version tells the optimiser which way is "out")
      - "below" per point = how far under the top of the wall: relu(height - y)
        (jax.nn.relu(z) = max(z, 0): zero if the point is above the wall)
      - penalty = sum over points of inside * below**2
      - only counts if the wall is on AND between the cannon and the target:
            active = enabled * (x_center < target_x)
        (a comparison gives a bool array; multiplying by it casts to 0/1)
      - return active * penalty
    """
    x_center, thickness, height, enabled = wall
    target_x = target[0]
    x, y = traj[:, 0], traj[:, 1]
    left = x_center - thickness/2
    right = x_center + thickness/2

    inside = jax.nn.sigmoid((x - left) / WALL_SMOOTH) * jax.nn.sigmoid((right - x) / WALL_SMOOTH)  # ~1.0 inside and ~0.0 outside
    below = jax.nn.relu(height - y)     # if arg < 0, below == 0; gets larger the lower the projectile's intersection with the wall

    penalty = jnp.sum(inside * below**2)
    active = enabled * (x_center < target_x)

    return active * penalty


def loss_fn(raw, target, wall):
    """
    Logic:
      - traj = simulate(raw)
      - return miss_loss(traj, target) + WALL_WEIGHT * wall_penalty(traj, target, wall)
    """
    traj = simulate(raw)
    _, speed = to_launch(raw)
    # optimize for reaching the target and avoiding the wall
    return miss_loss(traj, target) + WALL_WEIGHT * wall_penalty(traj, target, wall) + SPEED_WEIGHT * speed**2


# ---------------------------------------------------------------------------
# Stage 5 — Adam, by hand   (NEW: optimiser state as a pytree)
# ---------------------------------------------------------------------------
# Plain gradient descent zig-zagged in cannon.py because angle and speed have
# very different gradient sizes. Adam divides each parameter's step by a
# running estimate of its own gradient size, so both move at a sensible rate.
# Optimisers in JAX are just pure functions: (params, grads, state) -> (params, state).

def adam_init(raw):
    """
    Logic:
      - m = zeros like raw      (running mean of gradients)
      - v = zeros like raw      (running mean of squared gradients)
      - t = 0                   (step counter)
      - return (m, v, t)
    """
    m = jnp.zeros_like(raw)
    v = jnp.zeros_like(raw)
    t = 0

    return m, v, t


def adam_update(raw, grads, opt_state, lr=LEARNING_RATE, b1=0.9, b2=0.999, eps=1e-8):
    """
    One Adam step. Returns (new_raw, new_opt_state).

    Logic:
      - (m, v, t) = opt_state;  t = t + 1
      - m = b1 * m + (1 - b1) * grads
      - v = b2 * v + (1 - b2) * grads**2
      - bias correction (m and v start at 0, so early on they're too small):
            m_hat = m / (1 - b1**t),   v_hat = v / (1 - b2**t)
      - new_raw = raw - lr * m_hat / (sqrt(v_hat) + eps)
      - return new_raw, (m, v, t)
    """
    m, v, t = opt_state
    t = t + 1
    m = b1 * m + (1 - b1) * grads           # momentum
    v = b2 * v + (1 - b2) * grads**2        

    m_hat = m / (1 - b1**t)
    v_hat = v / (1 - b2**t)

    new_raw = raw - lr * m_hat / (jnp.sqrt(v_hat) + eps)

    return new_raw, (m, v, t)


# ---------------------------------------------------------------------------
# Stage 6 — The whole solve as ONE compiled function   (NEW: scan over iterations)
# ---------------------------------------------------------------------------

def solve(raw0, target, wall):
    """
    Run N_ITERS Adam steps from raw0. No Python for-loop — use lax.scan.

    Returns: (final_raw, raw_history, loss_history)
             shapes: (2,), (N_ITERS, 2), (N_ITERS,)

    Logic:
      - define an inner function body(carry, _):
            carry is (raw, opt_state)
            compute (loss, grads) with jax.value_and_grad(loss_fn) — grads w.r.t. raw
            (raw, opt_state) = adam_update(raw, grads, opt_state)
            return new carry, and record (raw, loss) for this iteration
        (body can use target and wall directly — it's a closure; they're
         constants from scan's point of view)
      - initial carry = (raw0, adam_init(raw0))
      - run lax.scan(body, initial carry, None, length=N_ITERS)
      - unpack: final carry gives final_raw; stacked outputs give the histories
    Compare with cannon.py's train(): same idea, but the loop now lives
    inside the compiled program.
    """
    loss_grad = jax.value_and_grad(loss_fn)
    def body(carry, _):
        raw, opt_state = carry
        loss, grads = loss_grad(raw, target, wall)
        raw, opt_state = adam_update(raw, grads, opt_state)
        return (raw, opt_state), (raw, loss)

    carry_init = raw0, adam_init(raw0)

    (final_raw, _), (raw_history, loss_history) = jax.lax.scan(body, carry_init, xs=None, length=N_ITERS)

    return (final_raw, raw_history, loss_history)


@jax.jit
def solve_arcs(raw0_batch, target, wall):
    """
    Solve from SEVERAL starting guesses at once.   (NEW: vmap a whole optimiser)

    raw0_batch has shape (n_guesses, 2). Returns the same three outputs as
    solve, each with an extra leading n_guesses axis.

    Logic:
      - vmap `solve` over axis 0 of raw0_batch; target and wall are shared (None)
      - call it and return the result
    Notice: you wrote solve() for ONE guess and never thought about batches.
    """
    vmapped_solve = jax.vmap(solve, in_axes=(0, None, None))  # batchify solve
    return vmapped_solve(raw0_batch, target, wall)

# ===========================================================================
#                   CHECKS — tests, not answers. Run before the window opens.
# ===========================================================================

def _raw0_batch():
    return jnp.stack([launch_to_raw(jnp.deg2rad(a), f * MAX_SPEED) for a, f in INITIAL_GUESSES])


def _check(name, fn):
    try:
        fn()
    except NotImplementedError:
        print(f"\n>>> Next up: {name}  (a function it needs still raises NotImplementedError)")
        sys.exit(0)
    print(f"  OK  {name}")


def check_stage_1():
    a, s = to_launch(jnp.array([0.0, 0.0]))
    assert jnp.allclose(a, MAX_ANGLE / 2) and jnp.allclose(s, MAX_SPEED / 2), "sigmoid(0) should be 0.5"
    a, s = to_launch(jnp.array([-50.0, 50.0]))
    assert 0 <= a < 1e-3 and s <= MAX_SPEED, "extreme raw values must stay in range"
    a2, s2 = to_launch(launch_to_raw(0.7, 30.0))
    assert jnp.allclose(a2, 0.7, atol=1e-4) and jnp.allclose(s2, 30.0, atol=1e-3), "launch_to_raw is not the inverse of to_launch"


def check_stage_2():
    raw = launch_to_raw(jnp.pi / 4, 20.0)
    traj = simulate(raw)
    assert traj.shape == (N_STEPS, 2), f"trajectory shape {traj.shape}, expected {(N_STEPS, 2)}"
    # landing x on flat ground vs the analytic range v^2 sin(2θ)/g
    y = np.asarray(traj[:, 1])
    i = int(np.argmax(y < 0))
    expected = 20.0**2 / GRAVITY
    assert abs(float(traj[i, 0]) - expected) < 1.0, f"lands at x={float(traj[i, 0]):.1f}, expected ~{expected:.1f}"


def check_stage_3():
    d = point_segment_sq_dist(jnp.array([0.0, 0.0]), jnp.array([10.0, 0.0]), jnp.array([5.0, 3.0]))
    assert jnp.allclose(d, 9.0), f"point above the middle of a segment: got {d}, expected 9"
    d = point_segment_sq_dist(jnp.array([0.0, 0.0]), jnp.array([10.0, 0.0]), jnp.array([13.0, 4.0]))
    assert jnp.allclose(d, 25.0), f"point past the end of a segment: got {d}, expected 25 (did you clip t?)"
    d = point_segment_sq_dist(jnp.array([1.0, 1.0]), jnp.array([1.0, 1.0]), jnp.array([4.0, 5.0]))
    assert jnp.isfinite(d), "zero-length segment gave nan (epsilon in the denominator?)"
    raw = launch_to_raw(jnp.pi / 4, 20.0)
    traj = simulate(raw)
    on_path = traj[123] * 0.5 + traj[124] * 0.5          # halfway between two dots
    assert miss_loss(traj, on_path) < 1e-4, "a target between two dots should have ~0 miss loss"


def check_stage_4():
    target = jnp.array([60.0, 5.0])
    raw = launch_to_raw(jnp.deg2rad(20.0), 30.0)         # low shot that goes through a tall wall
    wall_on = jnp.array([30.0, 2.0, 15.0, 1.0])
    wall_off = wall_on.at[3].set(0.0)
    wall_behind = jnp.array([80.0, 2.0, 15.0, 1.0])
    traj = simulate(raw)
    assert wall_penalty(traj, target, wall_on) > 0, "low shot through the wall should be penalised"
    assert wall_penalty(traj, target, wall_off) == 0, "disabled wall should give 0"
    assert wall_penalty(traj, target, wall_behind) == 0, "wall behind the target shouldn't count"
    g = jax.grad(loss_fn)(raw, target, wall_on)
    assert g.shape == (2,) and jnp.all(jnp.isfinite(g)), f"bad gradient: {g}"
    # the gradient should push the angle UP to clear the wall (gradient descent goes against g)
    assert g[0] < 0, "gradient doesn't push the angle up over the wall — check signs"


def check_stage_5():
    raw = jnp.array([1.0, -2.0])
    opt = adam_init(raw)
    new_raw, (m, v, t) = adam_update(raw, jnp.array([100.0, -0.01]), opt)
    assert t == 1
    # Adam's first step is ~lr in size for EVERY param, whatever the gradient scale
    step_size = jnp.abs(new_raw - raw)
    assert jnp.allclose(step_size, LEARNING_RATE, rtol=1e-3), f"first Adam step sizes {step_size}, expected ~{LEARNING_RATE} each"


def check_stage_6():
    target = jnp.array([40.0, 5.0])
    final, hist, losses = solve_arcs(_raw0_batch(), target, DEFAULT_WALL)
    n = len(INITIAL_GUESSES)
    assert final.shape == (n, 2) and hist.shape == (n, N_ITERS, 2) and losses.shape == (n, N_ITERS), \
        f"shapes: {final.shape}, {hist.shape}, {losses.shape}"
    assert float(losses.min()) < 1e-3, f"didn't converge on an easy target (best loss {float(losses.min()):.3g})"


def run_checks():
    print("Checks:")
    _check("Stage 1 — to_launch / launch_to_raw", check_stage_1)
    _check("Stage 2 — physics", check_stage_2)
    _check("Stage 3 — segment miss distance", check_stage_3)
    _check("Stage 4 — wall penalty & loss", check_stage_4)
    _check("Stage 5 — Adam", check_stage_5)
    _check("Stage 6 — solve / solve_arcs", check_stage_6)


# ===========================================================================
#                   UI PART — done for you (plain NumPy + matplotlib)
# ===========================================================================

HIT_RADIUS = 1.0        # m
REPLAY_EVERY = 10       # show every 10th optimisation iterate in the replay
BALL_SPEEDUP = 3        # trajectory points advanced per animation frame
ARC_COLORS = ["tab:blue", "tab:orange"]


def analyse_shot(traj, target, wall):
    """Non-differentiable bookkeeping for display: where does the ball stop, did it hit?"""
    traj = np.vstack([np.array(CANNON_POS), np.asarray(traj)])
    x, y = traj[:, 0], traj[:, 1]
    wx, wt, wh, on = np.asarray(wall)
    stop = len(traj) - 1
    blocked = False
    under = np.nonzero(y < 0)[0]
    if len(under):
        stop = under[0]
    if on:
        in_wall = np.nonzero((np.abs(x - wx) <= wt / 2) & (y < wh))[0]
        if len(in_wall) and in_wall[0] < stop:
            stop, blocked = in_wall[0], True
    flight = traj[: stop + 1]
    a, b = flight[:-1], flight[1:]
    ab = b - a
    t = np.clip(np.sum((target - a) * ab, 1) / (np.sum(ab * ab, 1) + 1e-9), 0, 1)
    d = np.linalg.norm(target - (a + t[:, None] * ab), axis=1)
    i_close = int(np.argmin(d))
    hit = d[i_close] < HIT_RADIUS
    if hit:
        stop = i_close + 1
        flight = traj[: stop + 1]
    return dict(flight=flight, hit=bool(hit), blocked=blocked, miss=float(d[i_close]))


def visible(traj):
    """Trajectory with the cannon position prepended, cut where it goes underground."""
    traj = np.vstack([np.array(CANNON_POS), np.asarray(traj)])
    under = np.nonzero(traj[:, 1] < 0)[0]
    return traj[: under[0] + 1] if len(under) else traj


class CannonGame:
    def __init__(self):
        self.raw0 = _raw0_batch()
        self.sim_many = jax.jit(jax.vmap(jax.vmap(simulate)))   # (arcs, iters, 2) -> trajectories
        self.target = np.array([40.0, 5.0])
        self.wall = np.array(DEFAULT_WALL)
        self.live_preview = True
        self.replay = True
        self.dragging = None                                    # "target" | "wall" | None
        self.timer = None

        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.fig.canvas.manager.set_window_title("JAX Cannon")
        ax = self.ax
        ax.set_xlim(-5, 125)
        ax.set_ylim(-3, 50)
        ax.set_aspect("equal")
        ax.axhspan(-3, 0, color="#8b6b4a", zorder=0)
        ax.set_title("left-drag: target  ·  right-drag: wall  ·  w: wall  ·  p: live preview  ·  r: replay", fontsize=9)

        self.barrel, = ax.plot([0, 0], [0, 0], lw=6, color="#333", solid_capstyle="round", zorder=5)
        ax.add_patch(plt.Circle(CANNON_POS, 1.5, color="#333", zorder=5))
        self.wall_patch = Rectangle((0, 0), 1, 1, color="#777", zorder=4)
        ax.add_patch(self.wall_patch)
        self.target_marker, = ax.plot([], [], "o", ms=14, mfc="none", mec="red", mew=2.5, zorder=6)
        self.target_dot, = ax.plot([], [], "o", ms=4, color="red", zorder=6)
        self.arcs = [ax.plot([], [], "-", color=c, lw=1.5, alpha=0.5, zorder=3)[0] for c in ARC_COLORS]
        self.ghosts = []
        self.ball, = ax.plot([], [], "o", ms=9, color="black", zorder=7)
        self.boom, = ax.plot([], [], "*", ms=35, color="gold", mec="red", zorder=8)
        self.status = ax.text(0.01, 0.97, "", transform=ax.transAxes, va="top", family="monospace", fontsize=9)
        self.result = ax.text(0.5, 0.6, "", transform=ax.transAxes, ha="center", fontsize=28, weight="bold")

        c = self.fig.canvas
        c.mpl_connect("button_press_event", self.on_press)
        c.mpl_connect("motion_notify_event", self.on_move)
        c.mpl_connect("button_release_event", self.on_release)
        c.mpl_connect("key_press_event", self.on_key)
        self.redraw_scene()
        self.fire()

    # ----- solving ---------------------------------------------------------
    def solve(self):
        t0 = time.perf_counter()
        final, hist, losses = solve_arcs(self.raw0, jnp.asarray(self.target, jnp.float32), jnp.asarray(self.wall))
        final.block_until_ready()          # JAX is async: wait for the result before stopping the clock
        ms = 1000 * (time.perf_counter() - t0)
        return np.asarray(final), np.asarray(hist), np.asarray(losses), ms

    def pick_best(self, final):
        trajs = np.asarray(jax.vmap(simulate)(jnp.asarray(final)))
        shots = [analyse_shot(t, self.target, self.wall) for t in trajs]
        order = sorted(range(len(shots)), key=lambda k: (not shots[k]["hit"], shots[k]["blocked"], shots[k]["miss"]))
        return order[0], trajs, shots

    # ----- drawing ---------------------------------------------------------
    def redraw_scene(self):
        wx, wt, wh, on = self.wall
        self.wall_patch.set_bounds(wx - wt / 2, 0, wt, wh)
        self.wall_patch.set_visible(bool(on))
        self.target_marker.set_data([self.target[0]], [self.target[1]])
        self.target_dot.set_data([self.target[0]], [self.target[1]])

    def aim_barrel(self, raw):
        angle, _ = to_launch(jnp.asarray(raw))
        angle = float(angle)
        self.barrel.set_data([0, 4 * np.cos(angle)], [0, 4 * np.sin(angle)])

    def clear_shot(self):
        for g in self.ghosts:
            g.remove()
        self.ghosts = []
        self.ball.set_data([], [])
        self.boom.set_data([], [])
        self.result.set_text("")

    def stop_animation(self):
        if self.timer is not None:
            self.timer.stop()
            self.timer = None

    # ----- a shot ----------------------------------------------------------
    def fire(self):
        self.stop_animation()
        self.clear_shot()
        final, hist, losses, ms = self.solve()
        best, trajs, shots = self.pick_best(final)
        s = shots[best]
        verdict = "HIT" if s["hit"] else ("BLOCKED" if s["blocked"] else f"MISS by {s['miss']:.1f} m")
        _, speed = to_launch(jnp.asarray(final[best]))
        angle = float(to_launch(jnp.asarray(final[best]))[0])
        self.status.set_text(
            f"solved {len(final)} arcs x {N_ITERS} Adam steps in {ms:6.1f} ms\n"
            f"best: guess {best + 1} (started at {INITIAL_GUESSES[best][0]:.0f}°) -> {np.degrees(angle):5.1f}°  {float(speed):5.1f} m/s  "
            f"loss {losses[best].min():.2e}\nwall: {'ON' if self.wall[3] else 'off'}"
        )
        frames = []
        if self.replay:
            snaps = self.sim_many(jnp.asarray(hist[:, ::REPLAY_EVERY]))   # (arcs, n_snaps, N_STEPS, 2)
            snaps = np.asarray(snaps)
            for k in range(snaps.shape[1]):
                frames.append(lambda k=k: self.show_iterate(snaps, hist, k))
        frames.append(lambda: self.show_final(trajs, best, final))
        flight = s["flight"]
        for i in range(0, len(flight), BALL_SPEEDUP):
            frames.append(lambda i=i: self.ball.set_data([flight[i, 0]], [flight[i, 1]]))
        frames.append(lambda: self.show_result(s, verdict))
        self.run_frames(frames)

    def show_iterate(self, snaps, hist, k):
        for a, line in enumerate(self.arcs):
            v = visible(snaps[a, k])
            line.set_data(v[:, 0], v[:, 1])
            line.set_alpha(0.9)
            if k % 3 == 0:
                g, = self.ax.plot(v[:, 0], v[:, 1], "-", color=ARC_COLORS[a], lw=0.6, alpha=0.15, zorder=2)
                self.ghosts.append(g)
        self.aim_barrel(hist[0, k * REPLAY_EVERY])

    def show_final(self, trajs, best, final):
        for a, line in enumerate(self.arcs):
            v = visible(trajs[a])
            line.set_data(v[:, 0], v[:, 1])
            line.set_alpha(0.9 if a == best else 0.25)
            line.set_linewidth(2.5 if a == best else 1.0)
        self.aim_barrel(final[best])

    def show_result(self, s, verdict):
        if s["hit"]:
            self.boom.set_data([self.target[0]], [self.target[1]])
        self.result.set_text(verdict)
        self.result.set_color("green" if s["hit"] else "firebrick")

    def run_frames(self, frames):
        it = iter(frames)

        def tick():
            try:
                next(it)()
            except StopIteration:
                self.stop_animation()
            self.fig.canvas.draw_idle()

        self.timer = self.fig.canvas.new_timer(interval=16)
        self.timer.add_callback(tick)
        self.timer.start()

    def preview(self):
        final, _, _, ms = self.solve()
        trajs = np.asarray(jax.vmap(simulate)(jnp.asarray(final)))
        for a, line in enumerate(self.arcs):
            v = visible(trajs[a])
            line.set_data(v[:, 0], v[:, 1])
            line.set_alpha(0.6)
            line.set_linewidth(1.5)
        self.status.set_text(f"live aim: {ms:5.1f} ms per solve")

    # ----- input -----------------------------------------------------------
    def on_press(self, e):
        if e.inaxes is not self.ax or e.xdata is None:
            return
        self.stop_animation()
        self.clear_shot()
        self.dragging = "target" if e.button == 1 else "wall" if e.button == 3 else None
        self.on_move(e)

    def on_move(self, e):
        if self.dragging is None or e.inaxes is not self.ax or e.xdata is None:
            return
        if self.dragging == "target":
            self.target = np.array([max(e.xdata, 2.0), max(e.ydata, 0.0)])
        else:
            self.wall[0] = max(e.xdata, 4.0)
            self.wall[2] = max(e.ydata, 1.0)
        self.redraw_scene()
        if self.live_preview:
            self.preview()
        self.fig.canvas.draw_idle()

    def on_release(self, e):
        if self.dragging is not None:
            self.dragging = None
            self.fire()

    def on_key(self, e):
        if e.key == "w":
            self.wall[3] = 0.0 if self.wall[3] else 1.0
            self.redraw_scene()
            self.fire()
        elif e.key == "p":
            self.live_preview = not self.live_preview
            self.status.set_text(f"live preview {'on' if self.live_preview else 'off'}")
            self.fig.canvas.draw_idle()
        elif e.key == "r":
            self.replay = not self.replay
            self.status.set_text(f"replay {'on' if self.replay else 'off'}")
            self.fig.canvas.draw_idle()


def main():
    run_checks()
    # First call compiles; every later call reuses the compiled program.
    t0 = time.perf_counter()
    solve_arcs(_raw0_batch(), jnp.array([40.0, 5.0]), DEFAULT_WALL)[0].block_until_ready()
    t1 = time.perf_counter()
    solve_arcs(_raw0_batch(), jnp.array([50.0, 9.0]), DEFAULT_WALL)[0].block_until_ready()
    t2 = time.perf_counter()
    print(f"\nsolve_arcs: first call (compile + run) {1000 * (t1 - t0):.0f} ms, second call {1000 * (t2 - t1):.1f} ms")
    game = CannonGame()
    plt.show()


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Ideas once it works
# ---------------------------------------------------------------------------
# - Wind: add a constant horizontal acceleration in step(), make it a slider,
#   pass it in like the wall (as a value, so no recompilation).
# - Air drag (-k|v|v): the ball's range changes, the solver doesn't care.
# - More starting guesses: add to INITIAL_GUESSES — vmap handles any number.
# - Many shots hit the same target. Add a small `speed**2` term to the loss
#   and the solver picks the most efficient one instead of any old one.
# - Moving target: solve for where the target WILL be (time becomes a param).
# - Swap miss_loss for a softmin version and compare convergence.
# - jax.make_jaxpr(solve_arcs)(...) to see the whole compiled program.
