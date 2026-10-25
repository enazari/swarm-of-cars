import ast, re, sys, torch, numpy as np, collections
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows = collections.defaultdict(dict)        # (model, set, safety) -> dict
cur = None; sset = None
for l in open("results_ladder.txt"):
    if l.startswith("#####"): sset = "hard" if "--hard" in l else "std"
    elif l.startswith("=="):
        cur = re.search(r"--ckpt ckpt/(\S+)\.pt", l).group(1); sf = "--safety2" in l
    elif l.startswith("{"):
        d = ast.literal_eval(l.split("} wall")[0] + "}"); rows[(cur, sset, "rule" if sf else "raw")] = d
def params(m):
    d = torch.load(f"ckpt/{m}.pt", map_location="cpu"); return sum(v.numel() for v in d["ema"].values())
models = sorted({k[0] for k in rows}); P = {m: params(m) for m in models}
# group: base name without seed
def base(m):
    m2 = re.sub(r"_s\d$", "", m)
    return {"d32k": "d32k", "bc32k": "bc32k"}.get(m2, m2)
def kind(m): return "bc" if m.startswith("bc") else "diff"
groups = collections.defaultdict(list)
for m in models:
    b = re.sub(r"_s\d$", "", m)
    if m in ("diff_r3", "diff_big_r3", "bc_r3"): b = {"diff_r3": "d5m8_dagger", "diff_big_r3": "d13m7_dagger", "bc_r3": "bc5m8_dagger"}[m]
    groups[b].append(m)
out = []
for b, ms in sorted(groups.items(), key=lambda kv: (kind(kv[1][0]), P[kv[1][0]])):
    g = lambda s, f: [rows[(m, s, "rule")][f] for m in ms if (m, s, "rule") in rows]
    r = lambda s, f: [rows[(m, s, "raw")][f] for m in ms if (m, s, "raw") in rows]
    fmt = lambda v: (f"{np.mean(v):.2f}" + (f" [{min(v):.2f}-{max(v):.2f}]" if len(v) > 1 else "")) if v else "-"
    out.append((kind(ms[0]), b, P[ms[0]], len(ms), fmt(g("std", "arrived_pct")), fmt(g("hard", "arrived_pct")),
                int(np.mean(g("std", "override"))), int(np.mean(g("hard", "override"))), (r("std", "cc") or ["-"])[0], (r("hard", "cc") or ["-"])[0]))
print(f"{'type':5} {'model':13} {'params':>9} {'n':>2} {'arrive std':>20} {'arrive hard':>20} {'ovr std':>8} {'ovr hard':>8} {'rawcc std':>9} {'rawcc hard':>10}")
for o in out: print(f"{o[0]:5} {o[1]:13} {o[2]:>9} {o[3]:>2} {o[4]:>20} {o[5]:>20} {o[6]:>8} {o[7]:>8} {str(o[8]):>9} {str(o[9]):>10}")
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
for a, (s, f, t) in zip(ax, [("std", "arrived_pct", "arrival, standard set (%)"), ("hard", "arrived_pct", "arrival, hard set (%)"), ("hard", "override", "safety overrides, hard set")]):
    for k, c in (("diff", "#1c7ed6"), ("bc", "#e8590c")):
        pts = collections.defaultdict(list)
        for b, ms in groups.items():
            if kind(ms[0]) != k or b.endswith("dagger"): continue
            for m in ms:
                if (m, s, "rule") in rows: pts[P[m]].append(rows[(m, s, "rule")][f])
        xs = sorted(pts); a.plot(xs, [np.mean(pts[x]) for x in xs], "-o", color=c, label="diffusion" if k == "diff" else "regression")
        for x in xs: a.scatter([x] * len(pts[x]), pts[x], s=8, color=c, alpha=0.5)
    a.set_xscale("log"); a.set_xlabel("parameters"); a.set_title(t); a.grid(alpha=0.3)
    if f == "override": a.set_yscale("log")
ax[0].legend(); plt.tight_layout(); plt.savefig("results_ladder.png", dpi=90)
