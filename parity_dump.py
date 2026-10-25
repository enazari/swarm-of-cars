import json, numpy as np
import env as E, teacher as T, core as C
rng = np.random.default_rng(777)
cases = []
for k in range(4):
    while True:
        sc = E.make_scene(rng, "crowd" if k % 2 else "stream")
        env = E.Env(sc); tea = T.Teacher(env, np.random.default_rng(k))
        ok = False
        for it in range(300):
            vt, dl = tea.act(); env.step(vt, dl)
            if it > 10 + 4 * k and len(env.active()) >= 3:
                ok = True; break
        if ok:
            break
    idx = env.active()
    crop, vec, tok, mask = env.observe(idx)
    # safety filter test on random actions
    r = np.random.default_rng(k)
    vt = r.uniform(-30, 75, env.n); dl = r.uniform(-0.62, 0.62, env.n)
    vt3, dl3 = vt.copy(), dl.copy()
    ch3 = C.safety_filter2(env.X, env.Y, env.TH, env.V, env.state, vt3, dl3, env.cards, env.W, env.H)
    vt2, dl2 = vt.copy(), dl.copy()
    ch = C.safety_filter(env.X, env.Y, env.TH, env.V, env.state, vt2, dl2, env.cards, env.W, env.H)
    # advance 5 substeps with the (filtered) actions
    X, Y, TH, V = env.X.copy(), env.Y.copy(), env.TH.copy(), env.V.copy()
    C.advance(X, Y, TH, V, env.state, vt2, dl2)
    cases.append(dict(W=env.W, H=env.H, cards=env.cards.tolist(), goals=env.goals.tolist(), X=env.X.tolist(), Y=env.Y.tolist(), TH=env.TH.tolist(),
                      V=env.V.tolist(), state=env.state.tolist(), prev_a=env.prev_a.tolist(), idx=idx.tolist(),
                      crop=crop.reshape(len(idx), -1).tolist(), vec=vec.tolist(), tok=tok.reshape(len(idx), -1).tolist(), mask=mask.tolist(),
                      field0=np.minimum(env.fields[0], 1e8).tolist(), vt=vt.tolist(), dl=dl.tolist(), vt_f=vt2.tolist(), dl_f=dl2.tolist(), changed=int(ch), vt_f2=vt3.tolist(), dl_f2=dl3.tolist(), changed2=int(ch3),
                      adv=dict(X=X.tolist(), Y=Y.tolist(), TH=TH.tolist(), V=V.tolist())))
json.dump(cases, open("web/parity.json", "w"))
print("dumped", [len(c["idx"]) for c in cases])
