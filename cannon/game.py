"""Interactive matplotlib front end: place a target, watch the solver aim, then fire.

The solver runs in JAX; everything here is plain NumPy bookkeeping and drawing.
"""

import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from cannon.physics import CANNON_POS, simulate, to_launch
from cannon.solver import DEFAULT_WALL, INITIAL_GUESSES, initial_raw_batch, solve

HIT_RADIUS = 1.0        # m
BALL_SPEEDUP = 3        # trajectory points advanced per animation frame
REPLAY_FRAMES = 30      # optimisation snapshots shown in the replay
FRAME_MS = 16
ARC_COLORS = ["tab:blue", "tab:orange"]


def analyse_shot(traj, target, wall):
    """Where the ball actually stops and whether it hits, with a hard (non-smooth) wall and ground.

    Returns ``dict(flight, hit, blocked, miss)``: ``flight`` is the path to draw, cut at the
    ground, the wall or the target; ``miss`` is the closest approach in metres.
    """
    traj = np.vstack([np.array(CANNON_POS), np.asarray(traj)])
    x, y = traj[:, 0], traj[:, 1]
    wx, wt, wh, on = np.asarray(wall)
    stop, blocked = len(traj) - 1, False
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
        flight = traj[: i_close + 2]
    return dict(flight=flight, hit=bool(hit), blocked=blocked, miss=float(d[i_close]))


def visible(traj):
    """Trajectory from the muzzle, cut where it goes underground."""
    traj = np.vstack([np.array(CANNON_POS), np.asarray(traj)])
    under = np.nonzero(traj[:, 1] < 0)[0]
    return traj[: under[0] + 1] if len(under) else traj


class CannonGame:
    def __init__(self):
        self.raw0 = initial_raw_batch()
        self.sim_many = jax.jit(jax.vmap(jax.vmap(simulate)))   # (arcs, snapshots, 2) -> trajectories
        self.target = None
        self.wall = np.array(DEFAULT_WALL)
        self.live_preview = False
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
        ax.set_title("left-drag: target  ·  right-drag: wall  ·  w: wall  ·  p: live preview  ·  r: replay  ·  h: help",
                     fontsize=9)

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
        self.help = ax.text(0.5, 0.5, "", transform=ax.transAxes, ha="center", va="center", family="monospace",
                            multialignment="left", fontsize=10, zorder=20,
                            bbox=dict(boxstyle="round,pad=1.2", fc="white", ec="#333", alpha=0.96))

        c = self.fig.canvas
        c.mpl_connect("button_press_event", self.on_press)
        c.mpl_connect("motion_notify_event", self.on_move)
        c.mpl_connect("button_release_event", self.on_release)
        c.mpl_connect("key_press_event", self.on_key)
        self.barrel.set_data([0, 4 * np.cos(np.pi / 4)], [0, 4 * np.sin(np.pi / 4)])
        self.redraw_scene()
        self.show_help(True)

    # ----- help ------------------------------------------------------------
    def help_text(self):
        on = lambda b: "on " if b else "off"
        return (
            "JAX CANNON\n\n"
            "Place a target and the cannon aims itself: JAX finds the launch\n"
            "angle and speed by gradient descent in milliseconds, then fires.\n\n"
            "left-click / drag    place the target (fires on release)\n"
            "right-click / drag   move the wall (x = position, y = height)\n"
            f"w                    wall on / off            [{on(self.wall[3])}]\n"
            f"p                    live aim while dragging  [{on(self.live_preview)}]\n"
            f"r                    replay the optimisation  [{on(self.replay)}]\n"
            "h                    show / hide this help\n\n"
            "Click anywhere to start."
        )

    def show_help(self, shown):
        self.help.set_text(self.help_text())
        self.help.set_visible(shown)
        self.fig.canvas.draw_idle()

    # ----- solving ---------------------------------------------------------
    def solve(self):
        t0 = time.perf_counter()
        final, hist, losses = solve(self.raw0, jnp.asarray(self.target, jnp.float32), jnp.asarray(self.wall))
        final.block_until_ready()          # JAX dispatches asynchronously; wait before reading the clock
        ms = 1000 * (time.perf_counter() - t0)
        return np.asarray(final), np.asarray(hist), np.asarray(losses), ms

    def pick_best(self, final):
        """Index of the arc to fire: hits first, then unblocked, then smallest miss."""
        trajs = np.asarray(jax.vmap(simulate)(jnp.asarray(final)))
        shots = [analyse_shot(t, self.target, self.wall) for t in trajs]
        order = sorted(range(len(shots)), key=lambda k: (not shots[k]["hit"], shots[k]["blocked"], shots[k]["miss"]))
        return order[0], trajs, shots

    # ----- drawing ---------------------------------------------------------
    def redraw_scene(self):
        wx, wt, wh, on = self.wall
        self.wall_patch.set_bounds(wx - wt / 2, 0, wt, wh)
        self.wall_patch.set_visible(bool(on))
        if self.target is not None:
            self.target_marker.set_data([self.target[0]], [self.target[1]])
            self.target_dot.set_data([self.target[0]], [self.target[1]])

    def aim_barrel(self, raw):
        angle = float(to_launch(jnp.asarray(raw))[0])
        self.barrel.set_data([0, 4 * np.cos(angle)], [0, 4 * np.sin(angle)])

    def clear_shot(self):
        for g in self.ghosts:
            g.remove()
        self.ghosts = []
        self.ball.set_data([], [])
        self.boom.set_data([], [])
        self.result.set_text("")

    def hide_arcs(self):
        for line in self.arcs:
            line.set_data([], [])

    def stop_animation(self):
        if self.timer is not None:
            self.timer.stop()
            self.timer = None

    # ----- a shot ----------------------------------------------------------
    def fire(self):
        self.stop_animation()
        self.clear_shot()
        if self.target is None:
            return
        final, hist, losses, ms = self.solve()
        best, trajs, shots = self.pick_best(final)
        s = shots[best]
        verdict = "HIT" if s["hit"] else ("BLOCKED" if s["blocked"] else f"MISS by {s['miss']:.1f} m")
        angle, speed = to_launch(jnp.asarray(final[best]))
        self.status.set_text(
            f"{len(final)} arcs x {hist.shape[1]} Adam steps in {ms:6.1f} ms\n"
            f"best: guess {best + 1} (started at {INITIAL_GUESSES[best][0]:.0f}°) -> "
            f"{np.degrees(float(angle)):5.1f}°  {float(speed):5.1f} m/s  loss {losses[best].min():.2e}\n"
            f"wall: {'ON' if self.wall[3] else 'off'}"
        )
        frames = []
        if self.replay:
            every = max(1, hist.shape[1] // REPLAY_FRAMES)
            snaps = np.asarray(self.sim_many(jnp.asarray(hist[:, ::every])))   # (arcs, snapshots, N_STEPS, 2)
            for k in range(snaps.shape[1]):
                frames.append(lambda k=k: self.show_iterate(snaps, hist, k, every))
        frames.append(lambda: self.show_final(trajs, best, final))
        flight = s["flight"]
        for i in range(0, len(flight), BALL_SPEEDUP):
            frames.append(lambda i=i: self.ball.set_data([flight[i, 0]], [flight[i, 1]]))
        frames.append(lambda: self.show_result(s, verdict))
        self.run_frames(frames)

    def show_iterate(self, snaps, hist, k, every):
        for a, line in enumerate(self.arcs):
            v = visible(snaps[a, k])
            line.set_data(v[:, 0], v[:, 1])
            line.set_alpha(0.9)
            line.set_linewidth(1.5)
            if k % 3 == 0:
                g, = self.ax.plot(v[:, 0], v[:, 1], "-", color=ARC_COLORS[a], lw=0.6, alpha=0.15, zorder=2)
                self.ghosts.append(g)
        self.aim_barrel(hist[0, k * every])

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

        self.timer = self.fig.canvas.new_timer(interval=FRAME_MS)
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
        self.show_help(False)
        self.stop_animation()
        self.clear_shot()
        self.hide_arcs()              # the old arcs belong to the old target
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
        if self.live_preview and self.target is not None:
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
        elif e.key == "r":
            self.replay = not self.replay
            self.status.set_text(f"replay {'on' if self.replay else 'off'}")
        elif e.key == "h":
            self.show_help(not self.help.get_visible())
        if self.help.get_visible():
            self.help.set_text(self.help_text())     # keep the [on]/[off] markers current
        self.fig.canvas.draw_idle()


def main():
    # The first call compiles; later calls with new targets or walls reuse the compiled program.
    t0 = time.perf_counter()
    solve(initial_raw_batch(), jnp.array([40.0, 5.0]), DEFAULT_WALL)[0].block_until_ready()
    t1 = time.perf_counter()
    solve(initial_raw_batch(), jnp.array([50.0, 9.0]), DEFAULT_WALL)[0].block_until_ready()
    t2 = time.perf_counter()
    print(f"solve: first call (compile + run) {1000 * (t1 - t0):.0f} ms, later calls {1000 * (t2 - t1):.1f} ms")
    game = CannonGame()   # keep a reference: matplotlib holds only weak refs to event handlers
    plt.show()
