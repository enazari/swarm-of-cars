import os, sys, time, numpy as np
from multiprocessing import Pool
import env as E, teacher as T, core as C

HC = 8   # label chunk length


def quant_crop(crop):
    return np.clip(np.round(crop * 100.0), -127, 127).astype(np.int8)


def gen_scene(seed, noise_levels=(0, 0, 1, 2)):
    rng = np.random.default_rng(seed)
    sc = E.make_scene(rng, "crowd" if rng.random() < 0.5 else "stream")
    env = E.Env(sc)
    tea = T.Teacher(env, np.random.default_rng(seed + 7))
    ns = float(rng.choice(noise_levels))
    ar = np.zeros((env.n, 2))
    buf = []                      # (car, crop, vec, tok, mask, act)
    while not env.finished():
        vt, dl = tea.act()
        idx = env.active()
        if len(idx):
            crop, vec, tok, mask = env.observe(idx)
            for r, i in enumerate(idx):
                p = tea.prev_plan[i]
                if p is None or tea.best_cost[i] > 2500:
                    continue
                a = np.stack([np.clip((p[:HC, 0] - 22.5) / 52.5, -1, 1), np.clip(p[:HC, 1] / C.MAXST, -1, 1)], 1)
                buf.append((i, quant_crop(crop[r]), vec[r].copy(), tok[r].copy(), mask[r].copy(), a.astype(np.float16)))
        if ns > 0:       # execute noisy actions: visits states the clean teacher would not
            ar = 0.8 * ar + 0.2 * rng.normal(0, 1, (env.n, 2))
            vt = vt + ns * 12.0 * ar[:, 0]
            dl = dl + ns * 0.14 * ar[:, 1]
        env.step(vt, dl, safety=False)
    keep = [b for b in buf if env.arrived[b[0]]]      # drop cars that never made it
    m = env.metrics()
    if not keep:
        return None, m
    out = dict(crop=np.stack([b[1] for b in keep]), vec=np.stack([b[2] for b in keep]), tok=np.stack([b[3] for b in keep]).astype(np.float16),
               mask=np.stack([b[4] for b in keep]).astype(np.uint8), act=np.stack([b[5] for b in keep]))
    return out, m


def worker(args):
    wid, seeds, outdir = args
    parts = []; stats = []
    for s in seeds:
        o, m = gen_scene(s)
        stats.append(m)
        if o is not None:
            parts.append(o)
    if parts:
        d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        np.savez(os.path.join(outdir, f"shard_{wid:04d}.npz"), **d)
        n = len(d["act"])
    else:
        n = 0
    return n, stats


if __name__ == "__main__":
    nscenes = int(sys.argv[1]); outdir = sys.argv[2]; seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    os.makedirs(outdir, exist_ok=True)
    nw = 22
    per = 6
    jobs = []
    seeds = list(range(seed0, seed0 + nscenes))
    for w, k in enumerate(range(0, nscenes, per)):
        jobs.append((w, seeds[k:k + per], outdir))
    t0 = time.time(); tot = 0; arr = 0; cars = 0; cc = 0
    with Pool(nw) as p:
        for n, stats in p.imap_unordered(worker, jobs):
            tot += n; arr += sum(s["arrived"] for s in stats); cars += sum(s["n"] for s in stats); cc += sum(s["cc"] for s in stats)
            print(f"samples {tot}  cars {cars} arrived {arr}  cc {cc}  elapsed {time.time()-t0:.0f}s", flush=True)
