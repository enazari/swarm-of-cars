"""Car physics, grid planning, observations and rollout costs (numba)."""
import heapq
import math
import numpy as np
from numba import njit

CS = 8.0                      # planning grid cell, px
L, WD, WB = 22.0, 11.0, 14.0  # car length, width, wheelbase
MAXST = 0.62
VMAX, VMIN = 75.0, -30.0
ACC, BRK = 120.0, 320.0
DT, SUB, DTC = 0.05, 5, 0.25  # sim step, sim steps per decision, decision interval (s)
RC = 6.0                      # body circle radius (3 circles at -7, 0, +7 along the car)
INFLATE = 15.0                # clearance around cards for route planning
GOAL_R = 18.0                 # arrival radius
CROP, CROP_RES, CROP_FWD = 32, 6.0, 40.0
KTOK = 6
NROUTE = 6
ROUTE_D = np.array([25.0, 50.0, 75.0, 100.0, 150.0, 200.0])
VEC_DIM = 2 + 1 + 1 + 2 + 2 * NROUTE


@njit(cache=True)
def dyn(x, y, th, v, vt, dl):
    dv = vt - v
    lim = ACC if (vt * v > 0 and abs(vt) > abs(v)) else BRK
    if dv > lim * DT:
        dv = lim * DT
    elif dv < -lim * DT:
        dv = -lim * DT
    v = v + dv
    th = th + v / WB * math.tan(dl) * DT
    x = x + math.cos(th) * v * DT
    y = y + math.sin(th) * v * DT
    return x, y, th, v


@njit(cache=True)
def rect_dist(px, py, x0, y0, x1, y1):
    cx = min(max(px, x0), x1)
    cy = min(max(py, y0), y1)
    return math.hypot(px - cx, py - cy)


@njit(cache=True)
def card_hit(x, y, th, cards, W, H, margin):
    """True if any body circle touches a card (or leaves the world)."""
    for o in (-7.0, 0.0, 7.0):
        px = x + math.cos(th) * o
        py = y + math.sin(th) * o
        if px < RC or py < RC or px > W - RC or py > H - RC:
            return True
        for m in range(cards.shape[0]):
            if rect_dist(px, py, cards[m, 0], cards[m, 1], cards[m, 2], cards[m, 3]) < RC + margin:
                return True
    return False


@njit(cache=True)
def car_dist(x1, y1, th1, x2, y2, th2):
    m = 1e9
    for o1 in (-7.0, 0.0, 7.0):
        ax = x1 + math.cos(th1) * o1
        ay = y1 + math.sin(th1) * o1
        for o2 in (-7.0, 0.0, 7.0):
            d = math.hypot(ax - (x2 + math.cos(th2) * o2), ay - (y2 + math.sin(th2) * o2))
            if d < m:
                m = d
    return m


# ---------------------------------------------------------------- grid and fields
@njit(cache=True)
def build_blocked(cards, W, H, inflate):
    gw = int(math.ceil(W / CS))
    gh = int(math.ceil(H / CS))
    b = np.zeros((gh, gw), np.uint8)
    for j in range(gh):
        for i in range(gw):
            cx = (i + 0.5) * CS
            cy = (j + 0.5) * CS
            if cx < 12 or cy < 12 or cx > W - 12 or cy > H - 12:
                b[j, i] = 1
                continue
            for m in range(cards.shape[0]):
                if (cx > cards[m, 0] - inflate and cx < cards[m, 2] + inflate and
                        cy > cards[m, 1] - inflate and cy < cards[m, 3] + inflate):
                    b[j, i] = 1
                    break
    return b


@njit(cache=True)
def dijkstra(blocked, extra, gx, gy):
    gh, gw = blocked.shape
    dist = np.full((gh, gw), 1e9, np.float64)
    # start at the nearest free cell to the goal
    sx, sy = gx, gy
    if blocked[sy, sx]:
        best = 1e9
        for r in range(1, 10):
            for j in range(max(0, gy - r), min(gh, gy + r + 1)):
                for i in range(max(0, gx - r), min(gw, gx + r + 1)):
                    if not blocked[j, i]:
                        d = (i - gx) ** 2 + (j - gy) ** 2
                        if d < best:
                            best = d
                            sx, sy = i, j
            if best < 1e9:
                break
    dist[sy, sx] = 0.0
    h = [(0.0, sy * gw + sx)]
    while len(h) > 0:
        d, idx = heapq.heappop(h)
        y = idx // gw
        x = idx % gw
        if d > dist[y, x]:
            continue
        for k in range(8):
            dx = 1 if k in (0, 4, 5) else (-1 if k in (1, 6, 7) else 0)
            dy = 1 if k in (2, 4, 6) else (-1 if k in (3, 5, 7) else 0)
            nx = x + dx
            ny = y + dy
            if nx < 0 or ny < 0 or nx >= gw or ny >= gh or blocked[ny, nx]:
                continue
            if dx != 0 and dy != 0 and (blocked[y, nx] or blocked[ny, x]):
                continue
            nd = d + (CS * 1.4142 if (dx != 0 and dy != 0) else CS) + extra[ny, nx]
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                heapq.heappush(h, (nd, ny * gw + nx))
    return dist


@njit(cache=True)
def field_at(field, x, y):
    gh, gw = field.shape
    i = min(max(int(x / CS), 0), gw - 1)
    j = min(max(int(y / CS), 0), gh - 1)
    v = field[j, i]
    return v if v < 3000.0 else 3000.0


@njit(cache=True)
def grad_target(field, x, y):
    """Centre of the lowest-field cell within 3 cells; returns (found, tx, ty)."""
    gh, gw = field.shape
    ci = min(max(int(x / CS), 0), gw - 1)
    cj = min(max(int(y / CS), 0), gh - 1)
    cur = field[cj, ci]
    best = cur
    bi, bj = ci, cj
    for j in range(max(0, cj - 3), min(gh, cj + 4)):
        for i in range(max(0, ci - 3), min(gw, ci + 4)):
            f = field[j, i]
            if f < best:
                best = f
                bi, bj = i, j
    if bi == ci and bj == cj:
        return False, x, y
    return True, (bi + 0.5) * CS, (bj + 0.5) * CS


@njit(cache=True)
def grad_steer(field, x, y, th):
    ok, tx, ty = grad_target(field, x, y)
    if not ok:
        return 0.0
    ld = max(15.0, math.hypot(tx - x, ty - y))
    err = math.atan2(ty - y, tx - x) - th
    err = math.atan2(math.sin(err), math.cos(err))
    return math.atan(2.0 * WB * math.sin(err) / ld)


@njit(cache=True)
def route_points(field, x, y, gx, gy, out):
    """Points along the steepest-descent path of the static field at path lengths ROUTE_D (world coords)."""
    gh, gw = field.shape
    ci = min(max(int(x / CS), 0), gw - 1)
    cj = min(max(int(y / CS), 0), gh - 1)
    if field[cj, ci] >= 1e8:                      # inside an inflated zone: use the nearest finite cell
        best = 1e18
        bi, bj = ci, cj
        for r in range(1, 8):
            for j in range(max(0, cj - r), min(gh, cj + r + 1)):
                for i in range(max(0, ci - r), min(gw, ci + r + 1)):
                    if field[j, i] < 1e8:
                        d = (i - ci) ** 2 + (j - cj) ** 2
                        if d < best:
                            best = d
                            bi, bj = i, j
            if best < 1e18:
                break
        ci, cj = bi, bj
    px, py = x, y
    plen = 0.0
    k = 0
    for _ in range(600):
        if k >= out.shape[0]:
            break
        cur = field[cj, ci]
        best = cur
        bi, bj = ci, cj
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni = ci + di
                nj = cj + dj
                if ni < 0 or nj < 0 or ni >= gw or nj >= gh:
                    continue
                f = field[nj, ni]
                if f < best:
                    best = f
                    bi, bj = ni, nj
        if bi == ci and bj == cj:
            break
        nx = (bi + 0.5) * CS
        ny = (bj + 0.5) * CS
        seg = math.hypot(nx - px, ny - py)
        while k < out.shape[0] and plen + seg >= ROUTE_D[k]:
            t = (ROUTE_D[k] - plen) / (seg + 1e-9)
            out[k, 0] = px + (nx - px) * t
            out[k, 1] = py + (ny - py) * t
            k += 1
        plen += seg
        px, py = nx, ny
        ci, cj = bi, bj
    while k < out.shape[0]:
        out[k, 0] = gx
        out[k, 1] = gy
        k += 1


# ---------------------------------------------------------------- observations
@njit(cache=True)
def build_obs(i, X, Y, TH, V, state, goals, cards, W, H, field, prev_a, crop, vec, tok, mask):
    """Fill the ego-frame observation of car i. state: 0 waiting, 1 driving, 2 parked, 3 done."""
    n = X.shape[0]
    x, y, th, v = X[i], Y[i], TH[i], V[i]
    c, s = math.cos(th), math.sin(th)
    crop[:] = 0
    # other cars near the crop
    near = np.zeros(n, np.int64)
    nn = 0
    for j in range(n):
        if j == i or (state[j] != 1 and state[j] != 2):
            continue
        if math.hypot(X[j] - x, Y[j] - y) < 170.0:
            near[nn] = j
            nn += 1
    half = CROP / 2.0 - 0.5
    for r in range(CROP):
        for q in range(CROP):
            ef = (q - half) * CROP_RES + CROP_FWD
            el = (r - half) * CROP_RES
            wx = x + c * ef - s * el
            wy = y + s * ef + c * el
            if wx < 0 or wy < 0 or wx > W or wy > H:
                crop[0, r, q] = 1.0
            else:
                for m in range(cards.shape[0]):
                    if wx > cards[m, 0] and wx < cards[m, 2] and wy > cards[m, 1] and wy < cards[m, 3]:
                        crop[0, r, q] = 1.0
                        break
            for t in range(nn):
                j = near[t]
                hit = False
                for o in (-7.0, 0.0, 7.0):
                    if math.hypot(wx - (X[j] + math.cos(TH[j]) * o), wy - (Y[j] + math.sin(TH[j]) * o)) < RC + 3.0:
                        hit = True
                        break
                if hit:
                    crop[1, r, q] = 1.0
                    vx = V[j] * math.cos(TH[j])
                    vy = V[j] * math.sin(TH[j])
                    crop[2, r, q] = (vx * c + vy * s) / 75.0
                    crop[3, r, q] = (-vx * s + vy * c) / 75.0
    # vector: goal in ego frame, distance, speed, previous action, route
    gx, gy = goals[i, 0], goals[i, 1]
    dxg, dyg = gx - x, gy - y
    vec[0] = min(max((dxg * c + dyg * s) / 100.0, -6.0), 6.0)
    vec[1] = min(max((-dxg * s + dyg * c) / 100.0, -6.0), 6.0)
    vec[2] = min(math.hypot(dxg, dyg) / 300.0, 4.0)
    vec[3] = v / 75.0
    vec[4] = prev_a[i, 0]
    vec[5] = prev_a[i, 1]
    rp = np.zeros((NROUTE, 2))
    route_points(field, x, y, gx, gy, rp)
    for k in range(NROUTE):
        dx, dy = rp[k, 0] - x, rp[k, 1] - y
        vec[6 + 2 * k] = min(max((dx * c + dy * s) / 100.0, -3.0), 3.0)
        vec[7 + 2 * k] = min(max((-dx * s + dy * c) / 100.0, -3.0), 3.0)
    # tokens: nearest cars within 250 px
    tok[:] = 0
    mask[:] = 0
    d2 = np.full(n, 1e18)
    for j in range(n):
        if j == i or (state[j] != 1 and state[j] != 2):
            continue
        d = math.hypot(X[j] - x, Y[j] - y)
        if d < 250.0:
            d2[j] = d
    for k in range(tok.shape[0]):
        jb = -1
        bd = 1e17
        for j in range(n):
            if d2[j] < bd:
                bd = d2[j]
                jb = j
        if jb < 0:
            break
        d2[jb] = 1e18
        dx, dy = X[jb] - x, Y[jb] - y
        dth = TH[jb] - th
        vx = V[jb] * math.cos(TH[jb])
        vy = V[jb] * math.sin(TH[jb])
        tok[k, 0] = (dx * c + dy * s) / 100.0
        tok[k, 1] = (-dx * s + dy * c) / 100.0
        tok[k, 2] = math.cos(dth)
        tok[k, 3] = math.sin(dth)
        tok[k, 4] = (vx * c + vy * s) / 75.0
        tok[k, 5] = (-vx * s + vy * c) / 75.0
        tok[k, 6] = 1.0 if state[jb] == 2 else 0.0
        tok[k, 7] = 1.0
        mask[k] = 1.0


# ---------------------------------------------------------------- teacher rollouts
@njit(cache=True)
def eval_samples(x0, y0, th0, v0, U, field, cards, W, H, nb, nb_n, prev_dl, prev_vt):
    """Cost of S control sequences. U[s,h] = (target speed, steer); NaN steer = follow the field gradient.
    nb[k, t, :] = (x, y, th) of neighbour k at sim step t."""
    S, Hn, _ = U.shape
    costs = np.zeros(S)
    for s in range(S):
        x, y, th, v = x0, y0, th0, v0
        c = 0.0
        pdl = prev_dl
        vend = 0.0
        for h in range(Hn):
            vt = U[s, h, 0]
            dl = U[s, h, 1]
            if math.isnan(dl):
                dl = grad_steer(field, x, y, th)
            if dl > MAXST:
                dl = MAXST
            elif dl < -MAXST:
                dl = -MAXST
            for k in range(SUB):
                x, y, th, v = dyn(x, y, th, v, vt, dl)
                t = h * SUB + k
                if card_hit(x, y, th, cards, W, H, 2.0):
                    c += 3000.0
                for m in range(nb_n):
                    d = car_dist(x, y, th, nb[m, t, 0], nb[m, t, 1], nb[m, t, 2])
                    if d < 14.0:
                        c += 3000.0
                    elif d < 22.0:
                        c += (22.0 - d) * 1.5
            V = field_at(field, x, y)
            c += 0.1 * V + 3.0 * abs(dl) + 25.0 * abs(dl - pdl)
            if v < 0:
                c += 6.0
            pdl = dl
            vend = V
        c += vend
        costs[s] = c
    return costs


@njit(cache=True)
def rollout_traj(x0, y0, th0, v0, vt, dl, steps, out):
    """Constant-action rollout, writing (x, y, th) for each sim step."""
    x, y, th, v = x0, y0, th0, v0
    for t in range(steps):
        x, y, th, v = dyn(x, y, th, v, vt, dl)
        out[t, 0] = x
        out[t, 1] = y
        out[t, 2] = th


@njit(cache=True)
def seq_traj(x0, y0, th0, v0, U, field, steps_per, out):
    x, y, th, v = x0, y0, th0, v0
    for h in range(U.shape[0]):
        vt = U[h, 0]
        dl = U[h, 1]
        if math.isnan(dl):
            dl = grad_steer(field, x, y, th)
        dl = min(max(dl, -MAXST), MAXST)
        for k in range(steps_per):
            x, y, th, v = dyn(x, y, th, v, vt, dl)
            t = h * steps_per + k
            out[t, 0] = x
            out[t, 1] = y
            out[t, 2] = th
    return


@njit(cache=True)
def jam_extra(X, Y, V, state, skip, gh, gw):
    """Extra path cost around cars that are standing still (traffic jams)."""
    extra = np.zeros((gh, gw), np.float64)
    for j in range(X.shape[0]):
        if j == skip or (state[j] != 1 and state[j] != 2):
            continue
        if abs(V[j]) > 8.0:
            continue
        cx = int(X[j] / CS)
        cy = int(Y[j] / CS)
        for dj in range(-4, 5):
            for di in range(-4, 5):
                i2 = cx + di
                j2 = cy + dj
                if 0 <= i2 < gw and 0 <= j2 < gh:
                    d = math.hypot(di, dj)
                    if d <= 4.0:
                        extra[j2, i2] += 60.0 * (1.0 - d / 5.0)
    return extra


@njit(cache=True)
def safety_filter(X, Y, TH, V, state, vt, dl, cards, W, H):
    """Replace the action of any car whose next 0.25 s would touch a car or card by braking.
    Cars already touching are only blocked if the move makes it worse. Returns number of overridden cars."""
    n = X.shape[0]
    traj = np.zeros((n, SUB, 3))
    changed = 0
    for it in range(2):
        for i in range(n):
            x, y, th, v = X[i], Y[i], TH[i], V[i]
            for k in range(SUB):
                if state[i] == 1:
                    x, y, th, v = dyn(x, y, th, v, vt[i], dl[i])
                traj[i, k, 0] = x
                traj[i, k, 1] = y
                traj[i, k, 2] = th
        for i in range(n):
            if state[i] != 1 or (vt[i] == 0.0 and dl[i] == 0.0):
                continue
            bad = False
            for k in range(SUB):
                if card_hit(traj[i, k, 0], traj[i, k, 1], traj[i, k, 2], cards, W, H, 1.0) and \
                        not card_hit(X[i], Y[i], TH[i], cards, W, H, 1.0):
                    bad = True
                    break
                for j in range(n):
                    if j == i or state[j] == 0 or state[j] == 3:
                        continue
                    d1 = car_dist(traj[i, k, 0], traj[i, k, 1], traj[i, k, 2], traj[j, k, 0], traj[j, k, 1], traj[j, k, 2])
                    if d1 < 13.0 and d1 < car_dist(X[i], Y[i], TH[i], X[j], Y[j], TH[j]):
                        bad = True
                        break
                if bad:
                    break
            if bad:
                vt[i] = 0.0
                dl[i] = 0.0
                changed += 1
    return changed


@njit(cache=True)
def advance(X, Y, TH, V, state, vt, dl):
    for i in range(X.shape[0]):
        if state[i] != 1:
            V[i] = 0.0
            continue
        x, y, th, v = X[i], Y[i], TH[i], V[i]
        for k in range(SUB):
            x, y, th, v = dyn(x, y, th, v, vt[i], dl[i])
        X[i], Y[i], TH[i], V[i] = x, y, th, v


@njit(cache=True)
def contact_pairs(X, Y, TH, state, cards, W, H, cc_out, cd_out):
    """Mark which car pairs / cars are in contact right now."""
    n = X.shape[0]
    for i in range(n):
        cd_out[i] = 0
        for j in range(n):
            cc_out[i, j] = 0
    for i in range(n):
        if state[i] != 1 and state[i] != 2:
            continue
        if state[i] == 1 and card_hit(X[i], Y[i], TH[i], cards, W, H, 0.0):
            cd_out[i] = 1
        for j in range(i + 1, n):
            if state[j] != 1 and state[j] != 2:
                continue
            if car_dist(X[i], Y[i], TH[i], X[j], Y[j], TH[j]) < 12.0:
                cc_out[i, j] = 1


@njit(cache=True)
def resolve_steer(x0, y0, th0, v0, U, field):
    """Replace NaN steering (= follow the field) in a control sequence by the steering that would be applied."""
    out = U.copy()
    x, y, th, v = x0, y0, th0, v0
    for h in range(U.shape[0]):
        dl = U[h, 1]
        if math.isnan(dl):
            dl = grad_steer(field, x, y, th)
        dl = min(max(dl, -MAXST), MAXST)
        out[h, 1] = dl
        for k in range(SUB):
            x, y, th, v = dyn(x, y, th, v, U[h, 0], dl)
    return out


@njit(cache=True)
def safety_filter2(X, Y, TH, V, state, vt, dl, cards, W, H):
    """Stronger rule: after the next 0.25 s every car is assumed to brake; any action that would still lead to a contact
    with a car or card is replaced by braking (for both cars of a threatened pair)."""
    n = X.shape[0]
    NBK = 12
    TT = SUB + NBK
    traj = np.zeros((n, TT, 3))
    changed = 0
    for it in range(4):
        for i in range(n):
            x, y, th, v = X[i], Y[i], TH[i], V[i]
            for k in range(TT):
                if state[i] == 1:
                    if k < SUB:
                        x, y, th, v = dyn(x, y, th, v, vt[i], dl[i])
                    else:
                        x, y, th, v = dyn(x, y, th, v, 0.0, 0.0)
                traj[i, k, 0] = x
                traj[i, k, 1] = y
                traj[i, k, 2] = th
        bad = np.zeros(n, np.uint8)
        for i in range(n):
            if state[i] != 1:
                continue
            nowc = card_hit(X[i], Y[i], TH[i], cards, W, H, 1.0)
            for k in range(TT):
                if (not nowc) and card_hit(traj[i, k, 0], traj[i, k, 1], traj[i, k, 2], cards, W, H, 1.0):
                    bad[i] = 1
                for j in range(n):
                    if j == i or state[j] == 0 or state[j] == 3:
                        continue
                    d1 = car_dist(traj[i, k, 0], traj[i, k, 1], traj[i, k, 2], traj[j, k, 0], traj[j, k, 1], traj[j, k, 2])
                    if d1 < 12.5 and d1 < car_dist(X[i], Y[i], TH[i], X[j], Y[j], TH[j]):
                        bad[i] = 1
                        if state[j] == 1:
                            bad[j] = 1
        any_change = False
        for i in range(n):
            if bad[i] == 1 and state[i] == 1 and not (vt[i] == 0.0 and dl[i] == 0.0):
                vt[i] = 0.0
                dl[i] = 0.0
                changed += 1
                any_change = True
        if not any_change:
            break
    return changed


@njit(cache=True)
def brake_trajs(X, Y, TH, V, state):
    """Where every car ends up if it brakes right now (SUB + 12 sim steps)."""
    n = X.shape[0]
    TT = SUB + 12
    out = np.zeros((n, TT, 3))
    for i in range(n):
        x, y, th, v = X[i], Y[i], TH[i], V[i]
        for k in range(TT):
            if state[i] == 1:
                x, y, th, v = dyn(x, y, th, v, 0.0, 0.0)
            out[i, k, 0] = x
            out[i, k, 1] = y
            out[i, k, 2] = th
    return out


@njit(cache=True)
def score_actions(i, cvt, cdl, X, Y, TH, V, state, btraj, cards, W, H, field):
    """Score N candidate first actions of car i: take it for 0.25 s, then brake, while every other car brakes right away.
    Unsafe candidates get +inf; safe ones the field value (remaining distance) where the car would be after 0.25 s."""
    N = cvt.shape[0]
    TT = SUB + 12
    n = X.shape[0]
    sc = np.zeros(N)
    nowc = card_hit(X[i], Y[i], TH[i], cards, W, H, 1.0)
    dnow = np.full(n, 1e9)
    for j in range(n):
        if j != i and state[j] != 0 and state[j] != 3:
            dnow[j] = car_dist(X[i], Y[i], TH[i], X[j], Y[j], TH[j])
    for c in range(N):
        x, y, th, v = X[i], Y[i], TH[i], V[i]
        bad = False
        xe, ye = x, y
        for k in range(TT):
            if k < SUB:
                x, y, th, v = dyn(x, y, th, v, cvt[c], cdl[c])
            else:
                x, y, th, v = dyn(x, y, th, v, 0.0, 0.0)
            if k == SUB - 1:
                xe, ye = x, y
            if (not nowc) and card_hit(x, y, th, cards, W, H, 1.0):
                bad = True
                break
            for j in range(n):
                if j == i or state[j] == 0 or state[j] == 3:
                    continue
                d1 = car_dist(x, y, th, btraj[j, k, 0], btraj[j, k, 1], btraj[j, k, 2])
                if d1 < 12.5 and d1 < dnow[j]:
                    bad = True
                    break
            if bad:
                break
        sc[c] = 1e9 if bad else field_at(field, xe, ye)
    return sc
