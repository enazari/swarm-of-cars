"""Privileged teacher: prioritized sampling MPC. Sees every car, plans in true dynamics, may wait, detour or reverse."""
import math
import numpy as np
import core as C

H = 12                      # planning horizon, decisions (3 s)
T = H * C.SUB
VCHOICES = np.array([-30.0, -15.0, 0.0, 20.0, 45.0, 75.0])


class Teacher:
    def __init__(self, env, rng, n_samples=512):
        self.env = env
        self.rng = rng
        self.S = n_samples
        n = env.n
        self.prev_plan = [None] * n
        self.dyn_field = [None] * n
        self.dyn_age = np.zeros(n, np.int64)
        self.rank = np.argsort(env.scene["priority"])   # car with the lowest priority value goes first
        self.best_cost = np.zeros(n)

    def _samples(self, i, x, y, th, v):
        rng, S = self.rng, self.S
        U = np.zeros((S, H, 2))
        k = 0
        U[k] = (0.0, 0.0); k += 1                                   # brake
        U[k] = (-25.0, 0.0); k += 1                                 # back straight up
        prev = self.prev_plan[i]
        if prev is not None:
            warm = np.vstack([prev[1:], prev[-1:]])
            nw = 96
            noise = np.stack([rng.normal(0, 12, (nw, H)), rng.normal(0, 0.12, (nw, H))], -1)
            U[k:k + nw] = warm[None] + noise
            k += nw
        # structured random: piecewise-constant speeds, random-walk steering
        nr = 128
        cuts = np.sort(rng.integers(1, H, (nr, 2)), axis=1)
        vals = VCHOICES[rng.integers(0, len(VCHOICES), (nr, 3))]
        hh = np.arange(H)[None]
        spd = np.where(hh < cuts[:, :1], vals[:, :1], np.where(hh < cuts[:, 1:], vals[:, 1:2], vals[:, 2:3]))
        steer = np.cumsum(rng.normal(0, 0.18, (nr, H)), 1) + rng.normal(0, 0.3, (nr, 1))
        U[k:k + nr, :, 0] = spd
        U[k:k + nr, :, 1] = steer
        k += nr
        # bang-bang maneuvers: full-lock steering, reverse and forward segments (three-point turns)
        nbb = 96
        cuts = np.sort(rng.integers(1, H, (nbb, 2)), axis=1)
        sv = np.array([-30.0, -20.0, 20.0, 40.0])[rng.integers(0, 4, (nbb, 3))]
        ss = np.array([-C.MAXST, 0.0, C.MAXST])[rng.integers(0, 3, (nbb, 3))]
        hh = np.arange(H)[None]
        seg = (hh >= cuts[:, :1]).astype(int) + (hh >= cuts[:, 1:]).astype(int)
        U[k:k + nbb, :, 0] = np.take_along_axis(sv, seg, 1)
        U[k:k + nbb, :, 1] = np.take_along_axis(ss, seg, 1)
        k += nbb
        # creep: a short move along the field, then stop (final approach to the parking spot)
        ncr = 64
        cv = rng.choice([6.0, 10.0, 15.0, 22.0, 30.0], ncr)
        clen = rng.integers(1, 5, ncr)
        U[k:k + ncr, :, 0] = np.where(np.arange(H)[None] < clen[:, None], cv[:, None], 0.0)
        U[k:k + ncr, :, 1] = np.nan
        k += ncr
        # follow the field with a few speed profiles
        ng = S - k
        base = rng.choice([20.0, 40.0, 60.0, 75.0], ng)[:, None] * np.ones((ng, H))
        stop_at = rng.integers(0, H + 4, ng)
        base[np.arange(H)[None] >= stop_at[:, None]] = rng.choice([0.0, 30.0, 75.0], (np.arange(H)[None] >= stop_at[:, None]).sum())
        U[k:, :, 0] = base
        U[k:, :, 1] = np.nan
        U[:, :, 0] = np.clip(U[:, :, 0], C.VMIN, C.VMAX)
        return U

    def act(self, order_bonus=True):
        env = self.env
        n = env.n
        vt = np.zeros(n); dl = np.zeros(n)
        drivers = [i for i in self.rank if env.state[i] == 1]
        if not drivers:
            return vt, dl
        # constant-action predictions for every car on the road; committed plans replace them in priority order
        pred = np.zeros((n, T, 3))
        for j in range(n):
            if env.state[j] in (1, 2):
                a_vt, a_dl = (env.last_a[j] if env.state[j] == 1 else (0.0, 0.0))
                C.rollout_traj(env.X[j], env.Y[j], env.TH[j], env.V[j], a_vt, a_dl, T, pred[j])
        order = sorted(drivers, key=lambda i: (0 if env.stuck[i] > 4.0 else 1))
        for i in order:
            x, y, th, v = env.X[i], env.Y[i], env.TH[i], env.V[i]
            if self.dyn_field[i] is None or self.dyn_age[i] >= 3:
                extra = C.jam_extra(env.X, env.Y, env.V, env.state, i, env.gh, env.gw)
                gx, gy = int(env.goals[i, 0] / C.CS), int(env.goals[i, 1] / C.CS)
                self.dyn_field[i] = C.dijkstra(env.blocked, extra, gx, gy)
                self.dyn_age[i] = 0
            self.dyn_age[i] += 1
            field = self.dyn_field[i]
            near = [j for j in range(n) if j != i and env.state[j] in (1, 2) and math.hypot(env.X[j] - x, env.Y[j] - y) < 300]
            nb = pred[near] if near else np.zeros((0, T, 3))
            U = self._samples(i, x, y, th, v)
            costs = C.eval_samples(x, y, th, v, U, field, env.cards, env.W, env.H, nb, len(near), env.last_a[i, 1], env.last_a[i, 0])
            b = int(np.argmin(costs))
            ub = C.resolve_steer(x, y, th, v, U[b], field)
            self.best_cost[i] = costs[b]
            self.prev_plan[i] = ub
            vt[i], dl[i] = ub[0]
            C.seq_traj(x, y, th, v, ub, field, C.SUB, pred[i])
        return vt, dl


def run_teacher(scene, seed=0, safety=True, record=None, max_decisions=None):
    """Run the teacher on a scene. record: callable(env, idx, plans, costs) per decision, called before stepping."""
    from env import Env
    env = Env(scene)
    rng = np.random.default_rng(seed)
    tea = Teacher(env, rng)
    d = 0
    while not env.finished() and (max_decisions is None or d < max_decisions):
        vt, dl = tea.act()
        if record is not None:
            idx = env.active()
            record(env, idx, tea)
        env.step(vt, dl, safety=safety)
        d += 1
    return env
