import sys, time, argparse, math, numpy as np
from multiprocessing import Pool
import torch
import env as E, teacher as T, policy as P

args = None
POL = None


def init(a):
    global args, POL
    args = a
    torch.set_num_threads(1)
    if a.policy in ("diff", "bc"):
        POL = P.StudentPolicy(a.ckpt, "cpu", steps=a.steps, exec_n=a.exec_n, nsamp=a.nsamp)


def one(seed):
    rng = np.random.default_rng(100000 + seed)
    sc = E.make_scene(rng, "crowd" if seed % 2 else "stream", hard=args.hard)
    env = E.Env(sc)
    if args.policy == "teacher":
        tea = T.Teacher(env, np.random.default_rng(seed))
    if POL is not None:
        torch.manual_seed(seed); POL.reset(env.n)
    t0 = time.time(); nact = 0
    while not env.finished():
        if args.policy == "teacher":
            vt, dl = tea.act()
        elif args.policy == "rule":
            vt, dl = P.rule_act(env)
        else:
            vt, dl = POL.act(env)
        env.step(vt, dl, safety=(2 if args.safety2 else args.safety))
    m = env.metrics(); m["wall"] = time.time() - t0; m["kind"] = sc["kind"]; m["seed"] = seed
    return m


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("policy"); p.add_argument("--ckpt", default=None); p.add_argument("--n", type=int, default=96)
    p.add_argument("--safety", action="store_true"); p.add_argument("--safety2", action="store_true"); p.add_argument("--steps", type=int, default=8); p.add_argument("--exec_n", type=int, default=1)
    p.add_argument("--workers", type=int, default=20); p.add_argument("--hard", action="store_true"); p.add_argument("--nsamp", type=int, default=1)
    a = p.parse_args()
    t0 = time.time()
    with Pool(a.workers, initializer=init, initargs=(a,)) as pool:
        res = pool.map(one, range(a.n), chunksize=1)
    tot = sum(r["n"] for r in res); arr = sum(r["arrived"] for r in res)
    out = dict(hard=a.hard, nsamp=a.nsamp, policy=a.policy, safety=('v2' if a.safety2 else a.safety), scenes=a.n, cars=tot, arrived=arr, arrived_pct=round(100 * arr / tot, 2),
               cc=sum(r["cc"] for r in res), cd=sum(r["cd"] for r in res), override=sum(r["override"] for r in res),
               mean_t=round(float(np.nanmean([r["mean_t"] for r in res])), 2))
    for kind in ("stream", "crowd"):
        rr = [r for r in res if r["kind"] == kind]
        out[kind + "_arrived_pct"] = round(100 * sum(r["arrived"] for r in rr) / max(1, sum(r["n"] for r in rr)), 1)
    print(out, f"wall {time.time()-t0:.0f}s", flush=True)
