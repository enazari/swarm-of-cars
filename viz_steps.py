import sys, math, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import env as E, teacher as T, core as C, model as M

ckpt = sys.argv[1]; out = sys.argv[2]; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 100001
d = torch.load(ckpt, map_location="cpu"); net = M.Policy(d["kind"], **d.get("cfg", {})); net.load_state_dict(d["ema"]); net.eval()


def pick_state(seed):
    rng = np.random.default_rng(seed)
    sc = E.make_scene(rng, "crowd")
    env = E.Env(sc); tea = T.Teacher(env, np.random.default_rng(seed))
    best = None
    for it in range(120):
        vt, dl = tea.act(); env.step(vt, dl)
        idx = env.active()
        if it > 6 and len(idx) >= 6:
            # a car with several neighbours close by and still far from its goal
            for i in idx:
                nb = sum(1 for j in idx if j != i and math.hypot(env.X[j] - env.X[i], env.Y[j] - env.Y[i]) < 90)
                far = math.hypot(env.goals[i, 0] - env.X[i], env.goals[i, 1] - env.Y[i])
                score = nb * 100 + min(far, 300)
                if best is None or score > best[0]:
                    best = (score, it, i, [a.copy() if hasattr(a, "copy") else a for a in (env.X, env.Y, env.TH, env.V, env.state, env.prev_a)], None)
            if it > 60:
                break
    return sc, best


sc, best = pick_state(seed)
env = E.Env(sc); tea = T.Teacher(env, np.random.default_rng(seed))
for it in range(best[1] + 1):
    vt, dl = tea.act()
    if it < best[1]:
        env.step(vt, dl)
i = best[2]
teacher_plan = tea.prev_plan[i]
K = 32
crop, vec, tok, mask = env.observe([i])
t = M.to_tensors(np.repeat(crop, K, 0), np.repeat(vec, K, 0), np.repeat(tok, K, 0), np.repeat(mask, K, 0), "cpu")
torch.manual_seed(3)
with torch.no_grad():
    res, traj = net.sample(*t, steps=8, return_traj=True)


def rollout(chunk):
    x, y, th, v = env.X[i], env.Y[i], env.TH[i], env.V[i]
    pts = [(x, y)]
    for h in range(M.HC):
        vt_ = float(np.clip(chunk[h, 0], -1, 1)) * 52.5 + 22.5
        dl_ = float(np.clip(chunk[h, 1], -1, 1)) * C.MAXST
        for _ in range(C.SUB):
            x, y, th, v = C.dyn(x, y, th, v, vt_, dl_)
        pts.append((x, y))
    return np.array(pts)


def car_poly(x, y, th):
    c, s = math.cos(th), math.sin(th)
    return [(x + c * a - s * b, y + s * a + c * b) for a, b in [(-11, -5.5), (11, -5.5), (11, 5.5), (-11, 5.5)]]


fig, axes = plt.subplots(3, 3, figsize=(16, 10.5))
axes = axes.ravel()
steps = list(range(9))
cx, cy = env.X[i], env.Y[i]
for ax, k in zip(axes, steps):
    for c in env.cards:
        ax.add_patch(Rectangle((c[0], c[1]), c[2] - c[0], c[3] - c[1], fc="#eee", ec="#999"))
    for j in range(env.n):
        if j != i and env.state[j] in (1, 2):
            ax.add_patch(Polygon(car_poly(env.X[j], env.Y[j], env.TH[j]), fc="#8899aa", ec="none"))
    ax.add_patch(Polygon(car_poly(cx, cy, env.TH[i]), fc="#e8590c", ec="k"))
    ax.plot(*env.goals[i], marker="*", color="#2b8a3e", ms=16)
    for q in range(K):
        p = rollout(traj[k][q].numpy())
        ax.plot(p[:, 0], p[:, 1], color="#1c7ed6", alpha=0.35 if k < 8 else 0.5, lw=1.2)
    if k == 8:
        pt = rollout(np.stack([np.clip((teacher_plan[:M.HC, 0] - 22.5) / 52.5, -1, 1), teacher_plan[:M.HC, 1] / C.MAXST], 1))
        ax.plot(pt[:, 0], pt[:, 1], color="#2b8a3e", lw=2.5, ls="--", label="teacher plan")
        ax.legend(loc="upper right")
    gx, gy = env.goals[i]
    x0, x1, y0, y1 = cx - 170, cx + 170, cy - 110, cy + 110
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if not (x0 < gx < x1 and y0 < gy < y1):          # goal outside the view: arrow toward it
        ang = math.atan2(gy - cy, gx - cx)
        ax.annotate("goal", xy=(cx + 105 * math.cos(ang), cy + 105 * math.sin(ang)), xytext=(cx + 40 * math.cos(ang), cy + 40 * math.sin(ang)),
                    arrowprops=dict(arrowstyle="-|>", color="#2b8a3e", lw=2), color="#2b8a3e", fontsize=11, ha="center")
    ax.set_title(f"denoising step {k}/8" + ("  (pure noise)" if k == 0 else ("  (final plans)" if k == 8 else "")))
plt.suptitle(f"32 plans sampled for the orange car (star = its goal), {steps[-1]} flow-matching steps", y=1.0)
plt.tight_layout(); plt.savefig(out, dpi=85)
print("saved", out, "car", i, "seed", seed)
