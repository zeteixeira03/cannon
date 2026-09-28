# cannon

An auto-aiming cannon built on differentiable physics in [JAX](https://github.com/jax-ml/jax).
Place a target, and the cannon finds a launch angle and speed that hit it by running
gradient descent **through the physics simulation**. The whole solve is one compiled
program and takes milliseconds.

```
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` elsewhere
pip install -r requirements.txt
python -m cannon                # play
python -m pytest                # tests
```

| Input        | Action                                         |
|--------------|------------------------------------------------|
| left-drag    | place the target; fires on release             |
| shift-drag   | set the target's velocity (arrow = 1 s travel) |
| right-drag   | move the wall (x = position, y = height)       |
| `w`          | wall on / off                                  |
| `d`          | air drag: off / light / heavy                  |
| `v`          | stop the target                                |
| `p`          | live aiming while dragging                     |
| `r`          | replay the optimisation before firing          |
| `h`          | help                                           |

## How it works

**Parameters.** The optimiser works on two unconstrained numbers, `raw = [a, s]`, mapped to
`angle = (π/2)·σ(a)` and `speed = v_max·σ(s)`. The sigmoid keeps every iterate a valid launch,
so no constraint handling is needed.

**Simulation.** Gravity plus quadratic air drag, `a = g − k|v|v`, stepped by `lax.scan` with a
second-order integrator (forward Euler drifts 0.2–0.6 m over a flight under drag). The flight
is never cut short (fixed shapes keep it compilable); the losses decide what counts.

**Loss.**
- *Miss*: the closest the ball gets to the target in space *and* time. The target moves at
  constant velocity, so in the target's frame the ball's relative path is still a polyline, and
  a hit means it passes through the origin. Measuring point-to-segment distance (not
  point-to-point) means a hit between two timesteps still scores ~0. A static target is just
  velocity 0; flight time never has to become a parameter.
- *Wall*: `Σ inside(x)·relu(h − y)²` over the flight *before* the closest approach. `inside` is a
  smooth 0..1 mask over the wall's width: a hard mask would have zero gradient, the smooth one
  points the optimiser out of the wall.

**Optimiser.** Hand-written Adam. Angle and speed have very different gradient scales, and
Adam normalises each parameter's step by its own.

**Solve.** Two starting guesses, the flat and the lobbed drag-free arcs through the target
(closed form), are optimised at once with `vmap`. Gradient descent then only corrects for drag,
target motion and the wall, and each guess stays on its own branch. The solve runs in two phases:
1. optimise with the wall off;
2. only for arcs that go through the wall, keep optimising with it on, warm-started from phase 1.

This way a wall that ends up under the arcs doesn't change the answer, and arcs don't start
stuck inside the wall. Under `vmap` there's no per-arc branching, so phase 2 runs for every
arc and `jnp.where` picks the result.

**Choosing the shot.** The loss only asks "does it hit?". Which hitting arc to fire is a separate
decision: the one with the lowest launch speed. A speed penalty in the loss would mix units
(m² vs (m/s)²) and pull the ball off the target.

**No recompilation.** Everything the player changes lives in a `Scene` pytree of float32 arrays
(target, target velocity, wall, drag; the wall's on/off flag is a float), so any change is new
data for the same compiled program. A test checks this.

## Layout

```
cannon/physics.py   launch parametrisation, simulation
cannon/solver.py    losses, Adam, batched two-phase solve
cannon/game.py      matplotlib UI
tests/              pytest suite
learning/           standalone JAX exercises (not part of the game)
```

## Roadmap

- **Wall-weight annealing**: ramp the wall weight from 0 over the iterations (`scan` with
  `xs=jnp.arange(N_ITERS)`); in early tests this helped stuck arcs at no extra cost.
- **Wind**: a constant horizontal acceleration: one more `Scene` field.
- **Accelerating targets**: `target_path` is the only place that assumes constant velocity.
