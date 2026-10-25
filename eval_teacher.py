import sys, time, math, numpy as np
from multiprocessing import Pool
import env as E, teacher as T

def one(seed):
    rng = np.random.default_rng(100000 + seed)
    sc = E.make_scene(rng, "crowd" if seed % 2 else "stream")
    e = T.run_teacher(sc, seed=seed, safety=False)
    m = e.metrics(); m["seed"] = seed; m["kind"] = sc["kind"]
    fails = []
    for i in range(e.n):
        if e.state[i] == 1:
            fails.append((i, round(math.hypot(e.X[i]-e.goals[i,0], e.Y[i]-e.goals[i,1])), round(e.stuck[i])))
    m["fails"] = fails
    return m

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 96
    t0 = time.time()
    with Pool(22) as p:
        res = p.map(one, range(n))
    tot = sum(r["n"] for r in res); arr = sum(r["arrived"] for r in res)
    print("scenes", n, "cars", tot, "arrived %d (%.1f%%)" % (arr, 100*arr/tot), "cc", sum(r["cc"] for r in res), "cd", sum(r["cd"] for r in res),
          "mean time %.1f" % np.nanmean([r["mean_t"] for r in res]), "wall %.0fs" % (time.time()-t0))
    for r in res:
        if r["arrived"] < r["n"]:
            print(" seed", r["seed"], r["kind"], "n", r["n"], "arrived", r["arrived"], "cc", r["cc"], "cd", r["cd"], r["fails"])
