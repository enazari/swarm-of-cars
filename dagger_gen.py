"""DAgger data: the student drives (some cars, some of the time), the teacher labels every visited state."""
import os, sys, time, numpy as np, torch
from multiprocessing import Pool
import env as E, teacher as T, core as C, policy as P
from gen_data import quant_crop, HC

POL = None


def init(ckpt, steps):
    global POL
    torch.set_num_threads(1)
    POL = P.StudentPolicy(ckpt, "cpu", steps=steps, exec_n=1)


def gen_scene(seed):
    rng = np.random.default_rng(seed)
    sc = E.make_scene(rng, "crowd" if rng.random() < 0.5 else "stream", hard=rng.random() < 0.3)   # some crowded, tight layouts
    env = E.Env(sc)
    tea = T.Teacher(env, np.random.default_rng(seed + 7))
    torch.manual_seed(seed); POL.reset(env.n)
    p_student = float(rng.choice([0.5, 1.0, 1.0]))
    use_student = rng.random(env.n) < p_student
    buf = []
    while not env.finished():
        tvt, tdl = tea.act()                       # teacher plans for everyone (labels)
        idx = env.active()
        if len(idx):
            crop, vec, tok, mask = env.observe(idx)
            for r, i in enumerate(idx):
                pl = tea.prev_plan[i]
                if pl is None or tea.best_cost[i] > 2500:
                    continue
                a = np.stack([np.clip((pl[:HC, 0] - 22.5) / 52.5, -1, 1), np.clip(pl[:HC, 1] / C.MAXST, -1, 1)], 1)
                buf.append((i, quant_crop(crop[r]), vec[r].copy(), tok[r].copy(), mask[r].copy(), a.astype(np.float16)))
        svt, sdl = POL.act(env)
        vt = np.where(use_student, svt, tvt); dl = np.where(use_student, sdl, tdl)
        env.step(vt, dl, safety=False)
    m = env.metrics()
    if not buf:
        return None, m
    return dict(crop=np.stack([b[1] for b in buf]), vec=np.stack([b[2] for b in buf]), tok=np.stack([b[3] for b in buf]).astype(np.float16),
                mask=np.stack([b[4] for b in buf]).astype(np.uint8), act=np.stack([b[5] for b in buf])), m


def worker(args):
    wid, seeds, outdir = args
    parts, stats = [], []
    for s in seeds:
        o, m = gen_scene(s)
        stats.append(m)
        if o is not None:
            parts.append(o)
    n = 0
    if parts:
        d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        np.savez(os.path.join(outdir, f"shard_{wid:04d}.npz"), **d); n = len(d["act"])
    return n, stats


if __name__ == "__main__":
    nscenes, outdir, seed0, ckpt = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    os.makedirs(outdir, exist_ok=True)
    per = 4
    seeds = list(range(seed0, seed0 + nscenes))
    jobs = [(w, seeds[k:k + per], outdir) for w, k in enumerate(range(0, nscenes, per))]
    t0 = time.time(); tot = arr = cars = cc = 0
    with Pool(22, initializer=init, initargs=(ckpt, steps)) as p:
        for n, stats in p.imap_unordered(worker, jobs):
            tot += n; arr += sum(s["arrived"] for s in stats); cars += sum(s["n"] for s in stats); cc += sum(s["cc"] for s in stats)
            print(f"samples {tot} cars {cars} arrived {arr} cc {cc} elapsed {time.time()-t0:.0f}s", flush=True)
