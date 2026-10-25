import sys, math, numpy as np, torch
from collections import Counter
import env as E, core as C, policy as P
torch.set_num_threads(4)
pol = P.StudentPolicy(sys.argv[1], "cpu", steps=8, exec_n=2)
cnt = Counter(); examples = []
for seed in range(100000, 100024):
    rng = np.random.default_rng(seed)
    sc = E.make_scene(rng, "crowd" if seed % 2 else "stream")
    env = E.Env(sc); torch.manual_seed(seed); pol.reset(env.n)
    prev = np.zeros((env.n, env.n), np.int64)
    while not env.finished():
        vt, dl = pol.act(env)
        env.step(vt, dl, safety=False)
        for i in range(env.n):
            for j in range(i + 1, env.n):
                if env.cc[i, j] and not prev[i, j]:
                    a, b = i, j
                    age = (env.t - env.t_start[a], env.t - env.t_start[b])
                    dth = abs(math.atan2(math.sin(env.TH[a] - env.TH[b]), math.cos(env.TH[a] - env.TH[b])))
                    # who is ahead of whom, closing speed
                    dx, dy = env.X[b] - env.X[a], env.Y[b] - env.Y[a]; d = math.hypot(dx, dy) + 1e-9
                    va = np.array([env.V[a] * math.cos(env.TH[a]), env.V[a] * math.sin(env.TH[a])]); vb = np.array([env.V[b] * math.cos(env.TH[b]), env.V[b] * math.sin(env.TH[b])])
                    closing = float(np.dot(va - vb, [dx / d, dy / d]))
                    cat = ("parked" if (env.state[a] == 2 or env.state[b] == 2) else
                           "head-on" if dth > 2.3 else "crossing" if dth > 0.8 else "same-dir")
                    cat += "/early" if min(age) < 1.5 else ""
                    cat += "/slow" if closing < 15 else "/fast"
                    cnt[cat] += 1
        prev[:] = env.cc
print(sum(cnt.values()), "contacts"); [print(k, v) for k, v in cnt.most_common()]
