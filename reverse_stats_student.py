import sys, numpy as np, torch
from multiprocessing import Pool
import env as E, policy as P
POL = None
def init(ck):
    global POL
    torch.set_num_threads(1); POL = P.StudentPolicy(ck, "cpu", steps=8, exec_n=1)
def one(seed):
    rng = np.random.default_rng(100000 + seed)
    sc = E.make_scene(rng, "crowd", hard=True)
    env = E.Env(sc); POL.reset(env.n); torch.manual_seed(seed)
    rev = drv = 0; rc = set()
    while not env.finished():
        vt, dl = POL.act(env); env.step(vt, dl, safety=2)
        for i in env.active():
            drv += 1
            if env.V[i] < -1.0: rev += 1; rc.add(i)
    return rev, drv, len(rc), env.n
if __name__ == "__main__":
    with Pool(8, initializer=init, initargs=(sys.argv[1],)) as p:
        r = p.map(one, range(24))
    rs, ds, rc, n = map(sum, zip(*r))
    print(f"student {sys.argv[1]}, hard scenes: {100*rs/ds:.1f}% of driving time in reverse; {rc}/{n} cars ({100*rc/n:.0f}%) reversed at least once")
