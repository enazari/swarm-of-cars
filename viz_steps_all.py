import sys, math, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import env as E, teacher as T, core as C, model as M

ckpt = sys.argv[1]; out = sys.argv[2]; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 100001
K = int(sys.argv[4]) if len(sys.argv) > 4 else 6
d = torch.load(ckpt, map_location="cpu"); net = M.Policy(d["kind"], **d.get("cfg", {})); net.load_state_dict(d["ema"]); net.eval()

rng = np.random.default_rng(seed)
sc = E.make_scene(rng, "crowd")
env = E.Env(sc); tea = T.Teacher(env, np.random.default_rng(seed))
best = (-1, 0)
for it in range(90):                                 # find the busiest moment
    vt, dl = tea.act(); env.step(vt, dl)
    n = len(env.active())
    if it > 5 and n > best[0]: best = (n, it)
env = E.Env(sc); tea = T.Teacher(env, np.random.default_rng(seed))
for it in range(best[1] + 1):
    vt, dl = tea.act()
    if it < best[1]: env.step(vt, dl)
idx = [int(i) for i in env.active()]
crop, vec, tok, mask = env.observe(idx)
t = M.to_tensors(np.repeat(crop, K, 0), np.repeat(vec, K, 0), np.repeat(tok, K, 0), np.repeat(mask, K, 0), "cpu")
torch.manual_seed(3)
with torch.no_grad():
    res, traj = net.sample(*t, steps=8, return_traj=True)


def rollout(i, chunk):
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


cmap = plt.get_cmap("tab20")
fig, axes = plt.subplots(3, 3, figsize=(17, 11)); axes = axes.ravel()
xs_ = [c[0] for c in env.cards] + [c[2] for c in env.cards]; ys_ = [c[1] for c in env.cards] + [c[3] for c in env.cards]
for k, ax in enumerate(axes):
    for c in env.cards:
        ax.add_patch(Rectangle((c[0], c[1]), c[2] - c[0], c[3] - c[1], fc="#eee", ec="#999"))
    for r, i in enumerate(idx):
        col = cmap(r % 20)
        ax.add_patch(Polygon(car_poly(env.X[i], env.Y[i], env.TH[i]), fc=col, ec="k", zorder=5))
        ax.plot(*env.goals[i], marker="*", color=col, ms=9, mec="k", mew=0.4, zorder=4)
        for q in range(K):
            p = rollout(i, traj[k][r * K + q].numpy())
            ax.plot(p[:, 0], p[:, 1], color=col, alpha=0.55, lw=1.1)
    ax.set_xlim(min(xs_) - 40, max(xs_) + 40); ax.set_ylim(max(ys_) + 40, min(ys_) - 40); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"denoising step {k}/8" + ("  (pure noise)" if k == 0 else ("  (final plans)" if k == 8 else "")))
plt.suptitle(f"all {len(idx)} cars at once: {K} plans per car (colour = car, star = its goal), 8 flow-matching steps", y=1.0)
plt.tight_layout(); plt.savefig(out, dpi=85)
print("saved", out, "cars", len(idx), "seed", seed)
