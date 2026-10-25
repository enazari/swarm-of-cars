import sys, numpy as np
from multiprocessing import Pool
import env as E, teacher as T

def one(seed):
    rng = np.random.default_rng(100000 + seed)
    sc = E.make_scene(rng, "crowd", hard=True)
    env = E.Env(sc); tea = T.Teacher(env, np.random.default_rng(seed))
    rev_steps = 0; drive_steps = 0; reversed_cars = set()
    while not env.finished():
        vt, dl = tea.act(); env.step(vt, dl)
        for i in env.active():
            drive_steps += 1
            if env.V[i] < -1.0:
                rev_steps += 1; reversed_cars.add(i)
    return rev_steps, drive_steps, len(reversed_cars), env.n

if __name__ == "__main__":
    with Pool(10) as p:
        r = p.map(one, range(24))
    rs, ds, rc, n = map(sum, zip(*r))
    print(f"teacher, hard scenes: {100*rs/ds:.1f}% of driving time in reverse; {rc}/{n} cars ({100*rc/n:.0f}%) reversed at least once")
