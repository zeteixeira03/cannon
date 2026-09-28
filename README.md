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
| right-drag   | move the wall (x = position, y = height)       |
| `w`          | wall on / off                                  |
| `p`          | live aiming while dragging                     |
| `r`          | replay the optimisation before firing          |
| `h`          | help                                           |

## How it works

**Parameters.** The optimiser works on two unconstrained numbers, `raw = [a, s]`, mapped to
`angle = (π/2)·σ(a)` and `speed = v_max·σ(s)`. The sigmoid keeps every iterate a valid launch,
so no constraint handling is needed.

**Simulation.** `lax.scan` steps the ball forward for a fixed number of timesteps. The flight
is never cut short (fixed shapes keep it compilable); the losses decide what counts.

**Loss.**
- *Miss*: squared distance from the target to the trajectory, treating the trajectory as a
  polyline, so a target between two timesteps still scores ~0.
- *Wall*: `Σ inside(x)·relu(h − y)²`, where `inside` is a smooth 0..1 mask over the wall's
  width. A hard mask would have zero gradient; the smooth one points the optimiser out of the wall.

**Optimiser.** Hand-written Adam. Angle and speed have very different gradient scales, and
Adam normalises each parameter's step by its own.

**Solve.** Several starting guesses (shallow and steep) are optimised at once with `vmap`, so
the solver usually finds both arcs through the target. The solve runs in two phases:
1. optimise with the wall off;
2. only for arcs that go through the wall, keep optimising with it on, warm-started from phase 1.

This way a wall that ends up under the arcs doesn't change the answer, and arcs don't start
stuck inside the wall. Under `vmap` there's no per-arc branching, so phase 2 runs for every
arc and `jnp.where` picks the result.

**No recompilation.** The target and wall are passed as array *values* (the wall's on/off
flag is a float), so moving or toggling them reuses the compiled program.

## Layout

```
cannon/physics.py   launch parametrisation, simulation
cannon/solver.py    losses, Adam, batched two-phase solve
cannon/game.py      matplotlib UI
tests/              pytest suite
learning/           earlier exercises this grew out of (not part of the game)
```

## Roadmap

- **Air drag** (`a = g − k|v|v`): needs a higher-order integrator. With drag, the cheapest
  arc drops below 45° and steep lobs become expensive, so choosing the lowest-speed hitting
  arc becomes a real decision.
- **Moving target**: measure the miss in the target's frame (`ball(t) − target(t)`). Both move
  linearly within a timestep, so the existing point-to-segment distance still applies and no
  flight-time parameter is needed.
- **Wall timing**: penalise only the flight *before* the closest approach, instead of the
  current "wall is left of the target" rule (which breaks once the target moves).
- **Wall-weight annealing**: ramp the wall weight from 0 over the iterations (`scan` with
  `xs=jnp.arange(N_ITERS)`); in early tests this helped stuck arcs at no extra cost.
- **Wind**: a constant horizontal acceleration, passed in as data like the wall.
