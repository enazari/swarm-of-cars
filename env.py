"""Scenes and the multi-car environment."""
import math
import numpy as np
import core as C

TMAX = 80.0


def make_scene(rng, kind=None, hard=False):
    kind = "crowd" if hard else (kind or rng.choice(["stream", "crowd"]))
    W = rng.uniform(720, 1180)
    H = rng.uniform(430, 640)
    gap = rng.uniform(44, 58) if hard else rng.uniform(46, 90)
    cards = []
    for _ in range(rng.integers(4, 10)):
        for _try in range(100):
            w, h = rng.uniform(120, 230), rng.uniform(55, 100)
            x, y = rng.uniform(20, W - w - 20), rng.uniform(20, H - h - 20)
            if all(not (x < c[2] + gap and x + w > c[0] - gap and y < c[3] + gap and y + h > c[1] - gap) for c in cards):
                cards.append((x, y, x + w, y + h))
                break
    cards = np.array(cards, dtype=np.float64)
    blocked = C.build_blocked(cards, W, H, C.INFLATE)
    gh, gw = blocked.shape
    free = np.argwhere(blocked == 0)
    n = int(rng.integers(6, 21)) if kind == "stream" else (int(rng.integers(20, 29)) if hard else int(rng.integers(8, 25)))
    goals = []
    tries = 0
    while len(goals) < n and tries < 2000:
        tries += 1
        m = rng.integers(len(cards))
        x0, y0, x1, y1 = cards[m]
        side = rng.integers(4)
        f = rng.uniform(0.12, 0.88)
        o = 22.0
        gx, gy = [(x0 + f * (x1 - x0), y0 - o), (x0 + f * (x1 - x0), y1 + o), (x0 - o, y0 + f * (y1 - y0)), (x1 + o, y0 + f * (y1 - y0))][side]
        if not (26 < gx < W - 26 and 26 < gy < H - 26) or blocked[int(gy / C.CS), int(gx / C.CS)]:
            continue
        if any(math.hypot(gx - a, gy - b) < 30 for a, b in goals):
            continue
        goals.append((gx, gy))
    n = len(goals)
    goals = np.array(goals)
    starts = np.zeros((n, 3))
    t_spawn = np.zeros(n)
    if kind == "stream":
        j, i = free[rng.integers(len(free))]
        sp = ((i + 0.5) * C.CS, (j + 0.5) * C.CS)
        dt_spawn = rng.uniform(0.3, 0.8)
        for k in range(n):
            starts[k] = (sp[0] + rng.uniform(-3, 3), sp[1] + rng.uniform(-3, 3), np.nan)
            t_spawn[k] = k * dt_spawn
    else:
        placed = []
        for k in range(n):
            for _try in range(200):
                j, i = free[rng.integers(len(free))]
                px, py = (i + 0.5) * C.CS, (j + 0.5) * C.CS
                if all(math.hypot(px - a, py - b) > 36 for a, b in placed):
                    placed.append((px, py))
                    break
            else:
                placed.append((px, py))
            starts[k] = (placed[-1][0], placed[-1][1], np.nan if rng.random() < 0.5 else rng.uniform(-math.pi, math.pi))
    return dict(kind=kind, W=W, H=H, cards=cards, blocked=blocked, goals=goals, starts=starts, t_spawn=t_spawn,
                priority=rng.permutation(n))


class Env:
    def __init__(self, scene):
        s = self.scene = scene
        self.W, self.H, self.cards, self.blocked = s["W"], s["H"], s["cards"], s["blocked"]
        self.goals = s["goals"]
        n = self.n = len(self.goals)
        self.gh, self.gw = self.blocked.shape
        self.X = np.zeros(n); self.Y = np.zeros(n); self.TH = np.zeros(n); self.V = np.zeros(n)
        self.state = np.zeros(n, np.int64)
        self.park_t = np.zeros(n)
        self.t = 0.0
        self.arrived = np.zeros(n, bool)
        self.t_arr = np.full(n, np.nan)
        self.t_start = np.full(n, np.nan)
        self.last_a = np.zeros((n, 2))          # (target speed, steer) in real units
        self.prev_a = np.zeros((n, 2))          # normalised, for observations
        self.stuck = np.zeros(n)
        self.fields = []
        for k in range(n):
            gx, gy = int(self.goals[k, 0] / C.CS), int(self.goals[k, 1] / C.CS)
            self.fields.append(C.dijkstra(self.blocked, np.zeros((self.gh, self.gw), np.float64), gx, gy))
        self.cc_prev = np.zeros((n, n), np.int64); self.cd_prev = np.zeros(n, np.int64)
        self.cc = np.zeros((n, n), np.int64); self.cd = np.zeros(n, np.int64)
        self.n_cc = 0; self.n_cd = 0; self.n_override = 0
        self.dead = np.zeros(n, bool)           # start unreachable
        for k in range(n):
            x, y = s["starts"][k, 0], s["starts"][k, 1]
            if self.fields[k][min(int(y / C.CS), self.gh - 1), min(int(x / C.CS), self.gw - 1)] >= 1e8:
                self.dead[k] = True
                self.state[k] = 3
        self._spawn()

    # --------------------------------------------------------------
    def _spawn(self):
        for k in range(self.n):
            if self.state[k] != 0 or self.t < self.scene["t_spawn"][k]:
                continue
            x, y, th = self.scene["starts"][k]
            ok = True
            for j in range(self.n):
                if self.state[j] in (1, 2) and math.hypot(self.X[j] - x, self.Y[j] - y) < 30:
                    ok = False
                    break
            if not ok:
                continue
            if math.isnan(th):
                rp = np.zeros((C.NROUTE, 2))
                C.route_points(self.fields[k], x, y, self.goals[k, 0], self.goals[k, 1], rp)
                th = math.atan2(rp[1, 1] - y, rp[1, 0] - x)
            self.X[k], self.Y[k], self.TH[k], self.V[k] = x, y, th, 0.0
            self.state[k] = 1
            self.t_start[k] = self.t

    def active(self):
        return np.flatnonzero(self.state == 1)

    def finished(self):
        return bool(np.all(self.state == 3)) or self.t >= TMAX

    def observe(self, idx):
        m = len(idx)
        crop = np.zeros((m, 4, C.CROP, C.CROP), np.float32)
        vec = np.zeros((m, C.VEC_DIM), np.float32)
        tok = np.zeros((m, C.KTOK, 8), np.float32)
        mask = np.zeros((m, C.KTOK), np.float32)
        for r, i in enumerate(idx):
            C.build_obs(i, self.X, self.Y, self.TH, self.V, self.state, self.goals, self.cards, self.W, self.H,
                        self.fields[i], self.prev_a, crop[r], vec[r], tok[r], mask[r])
        return crop, vec, tok, mask

    def step(self, vt, dl, safety=False):
        vt = np.array(vt, dtype=np.float64); dl = np.array(dl, dtype=np.float64)
        vt = np.clip(vt, C.VMIN, C.VMAX); dl = np.clip(dl, -C.MAXST, C.MAXST)
        if safety == 2:
            self.n_override += C.safety_filter2(self.X, self.Y, self.TH, self.V, self.state, vt, dl, self.cards, self.W, self.H)
        elif safety:
            self.n_override += C.safety_filter(self.X, self.Y, self.TH, self.V, self.state, vt, dl, self.cards, self.W, self.H)
        C.advance(self.X, self.Y, self.TH, self.V, self.state, vt, dl)
        self.last_a[:] = np.stack([vt, dl], 1)
        self.prev_a[:, 0] = np.clip((vt - 22.5) / 52.5, -1, 1); self.prev_a[:, 1] = dl / C.MAXST
        # contacts: count new ones
        C.contact_pairs(self.X, self.Y, self.TH, self.state, self.cards, self.W, self.H, self.cc, self.cd)
        self.n_cc += int(((self.cc == 1) & (self.cc_prev == 0)).sum())
        self.n_cd += int(((self.cd == 1) & (self.cd_prev == 0)).sum())
        self.cc_prev[:] = self.cc; self.cd_prev[:] = self.cd
        self.t += C.DTC
        for k in range(self.n):
            if self.state[k] == 1:
                if math.hypot(self.X[k] - self.goals[k, 0], self.Y[k] - self.goals[k, 1]) < C.GOAL_R:
                    self.state[k] = 2; self.park_t[k] = 1.5; self.V[k] = 0.0
                    self.arrived[k] = True; self.t_arr[k] = self.t - self.t_start[k]
                moving = abs(self.V[k]) > 2.0
                self.stuck[k] = 0.0 if moving else self.stuck[k] + C.DTC
            elif self.state[k] == 2:
                self.park_t[k] -= C.DTC
                if self.park_t[k] <= 0:
                    self.state[k] = 3
        self._spawn()

    def metrics(self):
        started = ~np.isnan(self.t_start)
        return dict(n=int((~self.dead).sum()), started=int(started.sum()), arrived=int(self.arrived.sum()),
                    cc=self.n_cc, cd=self.n_cd, override=self.n_override,
                    mean_t=float(np.nanmean(self.t_arr)) if self.arrived.any() else float("nan"), t_end=self.t)
