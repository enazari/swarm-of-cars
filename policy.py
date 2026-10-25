import math
import numpy as np
import torch
import model as M
import core as C


class StudentPolicy:
    """Closed-loop wrapper: samples an action chunk per car, executes the first exec_n actions, then samples again."""
    def __init__(self, ckpt, device="cpu", steps=8, exec_n=1, nsamp=1):
        d = torch.load(ckpt, map_location=device)
        self.net = M.Policy(d["kind"], **d.get("cfg", {})).to(device).eval()
        self.net.load_state_dict(d["ema"])
        self.dev = device; self.steps = steps; self.exec_n = exec_n; self.nsamp = nsamp
        self.chunk = None; self.ptr = None

    def reset(self, n):
        self.chunk = np.zeros((n, M.HC, 2), np.float32)
        self.ptr = np.full(n, 99)

    @torch.no_grad()
    def act(self, env):
        n = env.n
        if self.chunk is None or len(self.chunk) != n:
            self.reset(n)
        idx = env.active()
        vt = np.zeros(n); dl = np.zeros(n)
        need = [i for i in idx if self.ptr[i] >= self.exec_n]
        if need:
            crop, vec, tok, mask = env.observe(need)
            N = self.nsamp
            if N > 1:                       # several samples per car (same observation, different noise)
                crop, vec, tok, mask = (np.repeat(a, N, 0) for a in (crop, vec, tok, mask))
            t = M.to_tensors(crop, vec, tok, mask, self.dev)
            out = self.net.sample(*t, steps=self.steps).cpu().numpy()
            if N > 1:
                out = out.reshape(len(need), N, M.HC, 2)
                btraj = C.brake_trajs(env.X, env.Y, env.TH, env.V, env.state)
            for r, i in enumerate(need):
                if N == 1:
                    self.chunk[i] = out[r]
                else:
                    cvt = out[r, :, 0, 0] * 52.5 + 22.5
                    cdl = out[r, :, 0, 1] * C.MAXST
                    sc = C.score_actions(i, cvt.astype(np.float64), cdl.astype(np.float64), env.X, env.Y, env.TH, env.V, env.state, btraj, env.cards, env.W, env.H, env.fields[i])
                    b = int(np.argmin(sc))
                    self.chunk[i] = out[r, b]
                    if sc[b] >= 1e9:        # nothing safe among the samples: brake
                        self.chunk[i, 0] = ((0.0 - 22.5) / 52.5, 0.0)
                self.ptr[i] = 0
        for i in idx:
            a = self.chunk[i, self.ptr[i]]
            vt[i] = a[0] * 52.5 + 22.5
            dl[i] = a[1] * C.MAXST
            self.ptr[i] += 1
        return vt, dl


def rule_act(env):
    """Rule-based driver like the page: follow the static A* route with pure pursuit, brake for cars ahead, older car first.
    No reversing, no re-routing."""
    n = env.n
    vt = np.zeros(n); dl = np.zeros(n)
    for i in env.active():
        x, y, th, v = env.X[i], env.Y[i], env.TH[i], env.V[i]
        rp = np.zeros((C.NROUTE, 2))
        C.route_points(env.fields[i], x, y, env.goals[i, 0], env.goals[i, 1], rp)
        tx, ty = rp[0]
        ld = max(15.0, math.hypot(tx - x, ty - y))
        err = math.atan2(ty - y, tx - x) - th
        err = math.atan2(math.sin(err), math.cos(err))
        d = float(np.clip(math.atan(2 * C.WB * math.sin(err) / ld), -C.MAXST, C.MAXST))
        vd = max(22.0, C.VMAX / (1 + 2.5 * abs(d)))
        to_goal = math.hypot(env.goals[i, 0] - x, env.goals[i, 1] - y)
        if to_goal < 50:
            vd = min(vd, 12 + (C.VMAX - 12) * to_goal / 50)
        ch, sh = math.cos(th), math.sin(th)
        for j in range(n):
            if j == i or env.state[j] not in (1, 2):
                continue
            dx, dy = env.X[j] - x, env.Y[j] - y
            f = dx * ch + dy * sh
            if f < 0 or f > 60 or abs(-dx * sh + dy * ch) > 13:
                continue
            ox, oy = x - env.X[j], y - env.Y[j]
            mutual = (ox * math.cos(env.TH[j]) + oy * math.sin(env.TH[j]) > 0 and abs(-ox * math.sin(env.TH[j]) + oy * math.cos(env.TH[j])) < 13)
            if mutual and env.scene["priority"][i] < env.scene["priority"][j]:
                continue
            vd = min(vd, max(0.0, f - 22 - 6) * 1.5)
        vt[i], dl[i] = vd, d
    return vt, dl
