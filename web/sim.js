/* JavaScript port of core.py: car physics, grid fields, ego observations and the safety filter.
   Works in the browser (global Sim) and in node (module.exports) so it can be checked against the Python version. */
(function (root) {
  const CS = 8, L = 22, WB = 14, MAXST = 0.62, VMAX = 75, VMIN = -30, ACC = 120, BRK = 320;
  const DT = 0.05, SUB = 5, DTC = 0.25, RC = 6, INFLATE = 15, GOAL_R = 18;
  const CROP = 32, CROP_RES = 6, CROP_FWD = 40, KTOK = 6, NROUTE = 6, VEC_DIM = 18;
  const ROUTE_D = [25, 50, 75, 100, 150, 200];
  const OFFS = [-7, 0, 7];

  function dyn(x, y, th, v, vt, dl) {
    let dv = vt - v;
    const lim = (vt * v > 0 && Math.abs(vt) > Math.abs(v)) ? ACC : BRK;
    if (dv > lim * DT) dv = lim * DT; else if (dv < -lim * DT) dv = -lim * DT;
    v += dv;
    th += v / WB * Math.tan(dl) * DT;
    x += Math.cos(th) * v * DT;
    y += Math.sin(th) * v * DT;
    return [x, y, th, v];
  }
  function rectDist(px, py, x0, y0, x1, y1) {
    const cx = Math.min(Math.max(px, x0), x1), cy = Math.min(Math.max(py, y0), y1);
    return Math.hypot(px - cx, py - cy);
  }
  function cardHit(x, y, th, cards, W, H, margin) {
    for (const o of OFFS) {
      const px = x + Math.cos(th) * o, py = y + Math.sin(th) * o;
      if (px < RC || py < RC || px > W - RC || py > H - RC) return true;
      for (const c of cards) if (rectDist(px, py, c[0], c[1], c[2], c[3]) < RC + margin) return true;
    }
    return false;
  }
  function carDist(x1, y1, th1, x2, y2, th2) {
    let m = 1e9;
    for (const o1 of OFFS) {
      const ax = x1 + Math.cos(th1) * o1, ay = y1 + Math.sin(th1) * o1;
      for (const o2 of OFFS) {
        const d = Math.hypot(ax - (x2 + Math.cos(th2) * o2), ay - (y2 + Math.sin(th2) * o2));
        if (d < m) m = d;
      }
    }
    return m;
  }

  // ---------------------------------------------------------------- grid and field
  function grid(W, H) { return { gw: Math.ceil(W / CS), gh: Math.ceil(H / CS) }; }
  function buildBlocked(cards, W, H, inflate) {
    const { gw, gh } = grid(W, H), b = new Uint8Array(gw * gh);
    for (let j = 0; j < gh; j++) for (let i = 0; i < gw; i++) {
      const cx = (i + 0.5) * CS, cy = (j + 0.5) * CS;
      if (cx < 12 || cy < 12 || cx > W - 12 || cy > H - 12) { b[j * gw + i] = 1; continue; }
      for (const c of cards) {
        if (cx > c[0] - inflate && cx < c[2] + inflate && cy > c[1] - inflate && cy < c[3] + inflate) { b[j * gw + i] = 1; break; }
      }
    }
    return b;
  }
  const NB = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];
  function dijkstra(blocked, gw, gh, gx, gy) {
    const dist = new Float64Array(gw * gh).fill(1e9);
    let sx = gx, sy = gy;
    if (blocked[sy * gw + sx]) {
      let best = 1e9;
      for (let r = 1; r < 10; r++) {
        for (let j = Math.max(0, gy - r); j < Math.min(gh, gy + r + 1); j++) for (let i = Math.max(0, gx - r); i < Math.min(gw, gx + r + 1); i++) {
          if (!blocked[j * gw + i]) { const d = (i - gx) ** 2 + (j - gy) ** 2; if (d < best) { best = d; sx = i; sy = j; } }
        }
        if (best < 1e9) break;
      }
    }
    dist[sy * gw + sx] = 0;
    const hd = [0], hi = [sy * gw + sx];            // binary heap (d, idx)
    const push = (d, i) => {
      hd.push(d); hi.push(i); let k = hd.length - 1;
      while (k > 0) { const p = (k - 1) >> 1; if (hd[p] <= hd[k]) break; [hd[p], hd[k]] = [hd[k], hd[p]]; [hi[p], hi[k]] = [hi[k], hi[p]]; k = p; }
    };
    const pop = () => {
      const d = hd[0], i = hi[0], ld = hd.pop(), li = hi.pop();
      if (hd.length) {
        hd[0] = ld; hi[0] = li; let k = 0;
        for (;;) {
          const l = 2 * k + 1, r = l + 1; let m = k;
          if (l < hd.length && hd[l] < hd[m]) m = l;
          if (r < hd.length && hd[r] < hd[m]) m = r;
          if (m === k) break;
          [hd[m], hd[k]] = [hd[k], hd[m]]; [hi[m], hi[k]] = [hi[k], hi[m]]; k = m;
        }
      }
      return [d, i];
    };
    while (hd.length) {
      const [d, idx] = pop();
      const y = (idx / gw) | 0, x = idx % gw;
      if (d > dist[idx]) continue;
      for (const [dx, dy] of NB) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= gw || ny >= gh || blocked[ny * gw + nx]) continue;
        if (dx !== 0 && dy !== 0 && (blocked[y * gw + nx] || blocked[ny * gw + x])) continue;
        const nd = d + ((dx !== 0 && dy !== 0) ? CS * 1.4142 : CS);
        if (nd < dist[ny * gw + nx]) { dist[ny * gw + nx] = nd; push(nd, ny * gw + nx); }
      }
    }
    return dist;
  }
  function routePoints(field, gw, gh, x, y, gx, gy) {
    const out = [];
    let ci = Math.min(Math.max(Math.trunc(x / CS), 0), gw - 1), cj = Math.min(Math.max(Math.trunc(y / CS), 0), gh - 1);
    if (field[cj * gw + ci] >= 1e8) {
      let best = 1e18, bi = ci, bj = cj;
      for (let r = 1; r < 8; r++) {
        for (let j = Math.max(0, cj - r); j < Math.min(gh, cj + r + 1); j++) for (let i = Math.max(0, ci - r); i < Math.min(gw, ci + r + 1); i++) {
          if (field[j * gw + i] < 1e8) { const d = (i - ci) ** 2 + (j - cj) ** 2; if (d < best) { best = d; bi = i; bj = j; } }
        }
        if (best < 1e18) break;
      }
      ci = bi; cj = bj;
    }
    let px = x, py = y, plen = 0, k = 0;
    for (let it = 0; it < 600; it++) {
      if (k >= NROUTE) break;
      const cur = field[cj * gw + ci];
      let best = cur, bi = ci, bj = cj;
      for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
        if (di === 0 && dj === 0) continue;
        const ni = ci + di, nj = cj + dj;
        if (ni < 0 || nj < 0 || ni >= gw || nj >= gh) continue;
        const f = field[nj * gw + ni];
        if (f < best) { best = f; bi = ni; bj = nj; }
      }
      if (bi === ci && bj === cj) break;
      const nx = (bi + 0.5) * CS, ny = (bj + 0.5) * CS, seg = Math.hypot(nx - px, ny - py);
      while (k < NROUTE && plen + seg >= ROUTE_D[k]) {
        const t = (ROUTE_D[k] - plen) / (seg + 1e-9);
        out.push([px + (nx - px) * t, py + (ny - py) * t]); k++;
      }
      plen += seg; px = nx; py = ny; ci = bi; cj = bj;
    }
    while (k < NROUTE) { out.push([gx, gy]); k++; }
    return out;
  }

  // ---------------------------------------------------------------- observation
  // cars: arrays X, Y, TH, V, state (1 driving, 2 parked); prevA[i] = [vt_norm, dl_norm]
  function buildObs(i, S, goals, cards, W, H, field, gw, gh, prevA, crop, vec, tok, mask) {
    const n = S.X.length, x = S.X[i], y = S.Y[i], th = S.TH[i], v = S.V[i], c = Math.cos(th), s = Math.sin(th);
    crop.fill(0);
    const near = [];
    for (let j = 0; j < n; j++) {
      if (j === i || (S.state[j] !== 1 && S.state[j] !== 2) || (S.grp && S.grp[j] !== S.grp[i])) continue;
      if (Math.hypot(S.X[j] - x, S.Y[j] - y) < 170) near.push(j);
    }
    const half = CROP / 2 - 0.5, P = CROP * CROP;
    for (let r = 0; r < CROP; r++) for (let q = 0; q < CROP; q++) {
      const ef = (q - half) * CROP_RES + CROP_FWD, el = (r - half) * CROP_RES;
      const wx = x + c * ef - s * el, wy = y + s * ef + c * el, cell = r * CROP + q;
      if (wx < 0 || wy < 0 || wx > W || wy > H) crop[cell] = 1;
      else for (const m of cards) if (wx > m[0] && wx < m[2] && wy > m[1] && wy < m[3]) { crop[cell] = 1; break; }
      for (const j of near) {
        let hit = false;
        for (const o of OFFS) if (Math.hypot(wx - (S.X[j] + Math.cos(S.TH[j]) * o), wy - (S.Y[j] + Math.sin(S.TH[j]) * o)) < RC + 3) { hit = true; break; }
        if (hit) {
          crop[P + cell] = 1;
          const vx = S.V[j] * Math.cos(S.TH[j]), vy = S.V[j] * Math.sin(S.TH[j]);
          crop[2 * P + cell] = (vx * c + vy * s) / 75;
          crop[3 * P + cell] = (-vx * s + vy * c) / 75;
        }
      }
    }
    const gx = goals[i][0], gy = goals[i][1], dxg = gx - x, dyg = gy - y;
    const cl = (u, m) => Math.min(Math.max(u, -m), m);
    vec[0] = cl((dxg * c + dyg * s) / 100, 6); vec[1] = cl((-dxg * s + dyg * c) / 100, 6);
    vec[2] = Math.min(Math.hypot(dxg, dyg) / 300, 4); vec[3] = v / 75; vec[4] = prevA[i][0]; vec[5] = prevA[i][1];
    const rp = routePoints(field, gw, gh, x, y, gx, gy);
    for (let k = 0; k < NROUTE; k++) {
      const dx = rp[k][0] - x, dy = rp[k][1] - y;
      vec[6 + 2 * k] = cl((dx * c + dy * s) / 100, 3); vec[7 + 2 * k] = cl((-dx * s + dy * c) / 100, 3);
    }
    tok.fill(0); mask.fill(0);
    const cand = [];
    for (let j = 0; j < n; j++) {
      if (j === i || (S.state[j] !== 1 && S.state[j] !== 2) || (S.grp && S.grp[j] !== S.grp[i])) continue;
      const d = Math.hypot(S.X[j] - x, S.Y[j] - y);
      if (d < 250) cand.push([d, j]);
    }
    cand.sort((a, b) => a[0] - b[0]);
    for (let k = 0; k < Math.min(KTOK, cand.length); k++) {
      const j = cand[k][1], dx = S.X[j] - x, dy = S.Y[j] - y, dth = S.TH[j] - th;
      const vx = S.V[j] * Math.cos(S.TH[j]), vy = S.V[j] * Math.sin(S.TH[j]);
      const t = k * 8;
      tok[t] = (dx * c + dy * s) / 100; tok[t + 1] = (-dx * s + dy * c) / 100; tok[t + 2] = Math.cos(dth); tok[t + 3] = Math.sin(dth);
      tok[t + 4] = (vx * c + vy * s) / 75; tok[t + 5] = (-vx * s + vy * c) / 75; tok[t + 6] = S.state[j] === 2 ? 1 : 0; tok[t + 7] = 1;
      mask[k] = 1;
    }
  }

  // ---------------------------------------------------------------- stepping
  // replaces the action of cars whose next 0.25 s would touch a car or a card by braking (as in safety_filter)
  function safetyFilter(S, vt, dl, cards, W, H) {
    const n = S.X.length;
    const traj = Array.from({ length: n }, () => Array.from({ length: SUB }, () => [0, 0, 0]));
    let changed = 0;
    for (let it = 0; it < 2; it++) {
      for (let i = 0; i < n; i++) {
        let x = S.X[i], y = S.Y[i], th = S.TH[i], v = S.V[i];
        for (let k = 0; k < SUB; k++) {
          if (S.state[i] === 1) [x, y, th, v] = dyn(x, y, th, v, vt[i], dl[i]);
          traj[i][k][0] = x; traj[i][k][1] = y; traj[i][k][2] = th;
        }
      }
      for (let i = 0; i < n; i++) {
        if (S.state[i] !== 1 || (vt[i] === 0 && dl[i] === 0)) continue;
        let bad = false;
        const nowHit = cardHit(S.X[i], S.Y[i], S.TH[i], cards, W, H, 1);
        for (let k = 0; k < SUB && !bad; k++) {
          const a = traj[i][k];
          if (!nowHit && cardHit(a[0], a[1], a[2], cards, W, H, 1)) { bad = true; break; }
          for (let j = 0; j < n; j++) {
            if (j === i || S.state[j] === 0 || S.state[j] === 3) continue;
            const b = traj[j][k];
            const d1 = carDist(a[0], a[1], a[2], b[0], b[1], b[2]);
            if (d1 < 13 && d1 < carDist(S.X[i], S.Y[i], S.TH[i], S.X[j], S.Y[j], S.TH[j])) { bad = true; break; }
          }
        }
        if (bad) { vt[i] = 0; dl[i] = 0; changed++; }
      }
    }
    return changed;
  }

  // stronger rule: after the next 0.25 s every car is assumed to brake; actions that would still end in a contact become braking
  function safetyFilter2(S, vt, dl, cards, W, H) {
    const n = S.X.length, NBK = 12, TT = SUB + NBK;
    const traj = Array.from({ length: n }, () => Array.from({ length: TT }, () => [0, 0, 0]));
    let changed = 0;
    for (let it = 0; it < 4; it++) {
      for (let i = 0; i < n; i++) {
        let x = S.X[i], y = S.Y[i], th = S.TH[i], v = S.V[i];
        for (let k = 0; k < TT; k++) {
          if (S.state[i] === 1) [x, y, th, v] = k < SUB ? dyn(x, y, th, v, vt[i], dl[i]) : dyn(x, y, th, v, 0, 0);
          traj[i][k][0] = x; traj[i][k][1] = y; traj[i][k][2] = th;
        }
      }
      const bad = new Uint8Array(n);
      for (let i = 0; i < n; i++) {
        if (S.state[i] !== 1) continue;
        const nowc = cardHit(S.X[i], S.Y[i], S.TH[i], cards, W, H, 1);
        for (let k = 0; k < TT; k++) {
          const a = traj[i][k];
          if (!nowc && cardHit(a[0], a[1], a[2], cards, W, H, 1)) bad[i] = 1;
          for (let j = 0; j < n; j++) {
            if (j === i || S.state[j] === 0 || S.state[j] === 3) continue;
            const b = traj[j][k];
            const d1 = carDist(a[0], a[1], a[2], b[0], b[1], b[2]);
            if (d1 < 12.5 && d1 < carDist(S.X[i], S.Y[i], S.TH[i], S.X[j], S.Y[j], S.TH[j])) { bad[i] = 1; if (S.state[j] === 1) bad[j] = 1; }
          }
        }
      }
      let any = false;
      for (let i = 0; i < n; i++) if (bad[i] && S.state[i] === 1 && !(vt[i] === 0 && dl[i] === 0)) { vt[i] = 0; dl[i] = 0; changed++; any = true; }
      if (!any) break;
    }
    return changed;
  }
  function substep(S, vt, dl) {            // one 0.05 s step for all driving cars
    for (let i = 0; i < S.X.length; i++) {
      if (S.state[i] !== 1) { S.V[i] = 0; continue; }
      const r = dyn(S.X[i], S.Y[i], S.TH[i], S.V[i], vt[i], dl[i]);
      S.X[i] = r[0]; S.Y[i] = r[1]; S.TH[i] = r[2]; S.V[i] = r[3];
    }
  }

  // ---- collision physics (demo only; the training simulator lets cars pass through each other). Each car is a rigid body of mass 1 made of the three body circles.
  // The bicycle model drives the car; an impact adds a slide velocity (bx, by) and a spin om that die out like tyre friction, so a hit car slides on in the direction of the push.
  const INERTIA = 68, E_CAR = 0.45, E_WALL = 0.5, MU_CAR = 0.3, MU_WALL = 0.4, TAU_SLIDE = 0.45, TAU_SPIN = 0.5;
  function physics(S, B, rects, W, H, bays, owners, wvel) {   // wvel[i] = [vx, vy] of wall i (a card being dragged): a moving wall shoves the cars it hits with its momentum   // bays[k] = tunnel bay rect of card k, owners[i] = card of wall i: walls are always solid (no overlap), but a car whose centre is inside a bay only touches that card's walls softly: no bounce, no friction, no impact flash
    const n = S.X.length, events = [], big = 400;
    const walls = rects.concat([[-big, -big, 0, H + big], [W, -big, W + big, H + big], [-big, -big, W + big, 0], [-big, H, W + big, H + big]]);
    const SPEEDCAP = 420;                                         // px/s: keeps a hit car from tunnelling through a wall in one 0.05 s step
    const fs = Math.exp(-DT / TAU_SLIDE), fo = Math.exp(-DT / TAU_SPIN);
    for (let i = 0; i < n; i++) if (S.state[i] === 1) {          // slide and spin carry on, then die out
      S.X[i] += B.bx[i] * DT; S.Y[i] += B.by[i] * DT; S.TH[i] += B.om[i] * DT;
      B.bx[i] *= fs; B.by[i] *= fs; B.om[i] *= fo;
    }
    const vel = (i) => [S.V[i] * Math.cos(S.TH[i]) + B.bx[i], S.V[i] * Math.sin(S.TH[i]) + B.by[i]];
    function push(i, jx, jy, px, py) {                            // impulse (jx, jy) on car i at point (px, py)
      const v = vel(i), rx = px - S.X[i], ry = py - S.Y[i];
      const nvx = v[0] + jx, nvy = v[1] + jy;
      B.om[i] += (rx * jy - ry * jx) / INERTIA;
      const hx = Math.cos(S.TH[i]), hy = Math.sin(S.TH[i]);
      const vl = Math.max(VMIN, Math.min(VMAX, nvx * hx + nvy * hy));
      S.V[i] = vl; B.bx[i] = nvx - vl * hx; B.by[i] = nvy - vl * hy;
      const sp = Math.hypot(B.bx[i], B.by[i]); if (sp > SPEEDCAP) { B.bx[i] *= SPEEDCAP / sp; B.by[i] *= SPEEDCAP / sp; }
    }
    // one contact between car a and car b (b < 0: a fixed wall). n points from a to b (for a wall: from the wall to the car a).
    function resolve(a, b, px, py, nx, ny, pen, soft, wx, wy) {
      const va = vel(a), wa = B.om[a], rax = px - S.X[a], ray = py - S.Y[a];
      let ux = va[0] - wa * ray, uy = va[1] + wa * rax, rbx = 0, rby = 0;
      if (b >= 0) { const vb = vel(b), wb = B.om[b]; rbx = px - S.X[b]; rby = py - S.Y[b]; ux = (vb[0] - wb * rby) - ux; uy = (vb[1] + wb * rbx) - uy; }
      else { ux = (wx || 0) - ux; uy = (wy || 0) - uy; }                                // relative velocity of "b" with respect to "a" (wall: b is at rest)
      if (b < 0) { nx = -nx; ny = -ny; }                          // from here n points from a toward b / into the wall
      const vn = ux * nx + uy * ny;
      if (vn >= 0) return 0;
      const can = rax * ny - ray * nx, cbn = rbx * ny - rby * nx;
      const e = soft ? 0 : b >= 0 ? E_CAR : E_WALL, den = (b >= 0 ? 2 : 1) + can * can / INERTIA + (b >= 0 ? cbn * cbn / INERTIA : 0);
      const j = -(1 + e) * vn / den;
      let jx = j * nx, jy = j * ny;
      const tx = -ny, ty = nx, vt = ux * tx + uy * ty, cat = rax * ty - ray * tx, cbt = rbx * ty - rby * tx;
      const dt_ = (b >= 0 ? 2 : 1) + cat * cat / INERTIA + (b >= 0 ? cbt * cbt / INERTIA : 0);
      const mu = soft ? 0 : b >= 0 ? MU_CAR : MU_WALL, jt = Math.max(-mu * j, Math.min(mu * j, -vt / dt_));
      jx += jt * tx; jy += jt * ty;
      push(a, -jx, -jy, px, py);                                  // a is pushed against n, b along n
      if (b >= 0) push(b, jx, jy, px, py);
      return soft ? 0 : j;
    }
    for (let a = 0; a < n; a++) {
      if (S.state[a] !== 1) continue;
      for (let b = a + 1; b < n; b++) {
        if (S.state[b] !== 1 || (S.grp && S.grp[a] !== S.grp[b]) || Math.abs(S.X[a] - S.X[b]) > 40 || Math.abs(S.Y[a] - S.Y[b]) > 40) continue;
        let best = 0, bn = null;
        for (const o1 of OFFS) { const ax = S.X[a] + Math.cos(S.TH[a]) * o1, ay = S.Y[a] + Math.sin(S.TH[a]) * o1;
          for (const o2 of OFFS) { const bx = S.X[b] + Math.cos(S.TH[b]) * o2, by = S.Y[b] + Math.sin(S.TH[b]) * o2, d = Math.hypot(bx - ax, by - ay), pen = 2 * RC - d;
            if (pen > best) { best = pen; bn = [(ax + bx) / 2, (ay + by) / 2, d > 1e-6 ? (bx - ax) / d : 1, d > 1e-6 ? (by - ay) / d : 0]; } } }
        if (!bn) continue;
        const j = resolve(a, b, bn[0], bn[1], bn[2], bn[3], best);
        S.X[a] -= bn[2] * best / 2; S.Y[a] -= bn[3] * best / 2; S.X[b] += bn[2] * best / 2; S.Y[b] += bn[3] * best / 2;
        if (j > 0) events.push([bn[0], bn[1], j, a]);
      }
      for (let it = 0; it < 4; it++) for (const o of OFFS) {       // several passes: pushing one body circle out of a wall can push another into a corner
        const cx = S.X[a] + Math.cos(S.TH[a]) * o, cy = S.Y[a] + Math.sin(S.TH[a]) * o;
        for (let wi = 0; wi < walls.length; wi++) {
          const w = walls[wi];
          let soft = false;
          if (bays && owners[wi] !== undefined) { const bb = bays[owners[wi]]; soft = S.X[a] > bb[0] && S.X[a] < bb[2] && S.Y[a] > bb[1] && S.Y[a] < bb[3]; }
          if (cx < w[0] - RC || cx > w[2] + RC || cy < w[1] - RC || cy > w[3] + RC) continue;
          const qx = Math.min(Math.max(cx, w[0]), w[2]), qy = Math.min(Math.max(cy, w[1]), w[3]), d = Math.hypot(cx - qx, cy - qy);
          let nx, ny, pen;
          if (d > 1e-6) { if (d >= RC) continue; nx = (cx - qx) / d; ny = (cy - qy) / d; pen = RC - d; }
          else {                                                   // circle centre inside the wall: leave by the nearest face
            const f = [[cx - w[0], -1, 0], [w[2] - cx, 1, 0], [cy - w[1], 0, -1], [w[3] - cy, 0, 1]].sort((p, q) => p[0] - q[0])[0];
            nx = f[1]; ny = f[2]; pen = RC + f[0];
          }
          const wv = wvel && wvel[wi], j = resolve(a, -1, cx - nx * RC, cy - ny * RC, nx, ny, pen, soft, wv ? wv[0] : 0, wv ? wv[1] : 0);
          S.X[a] += nx * pen; S.Y[a] += ny * pen;
          if (j > 0) events.push([cx - nx * RC, cy - ny * RC, j, a]);
        }
      }
    }
    return events;
  }

  const api = { CS, L, WB, MAXST, VMAX, VMIN, DT, SUB, DTC, RC, INFLATE, GOAL_R, CROP, KTOK, NROUTE, VEC_DIM, dyn, cardHit, carDist, rectDist,
    buildBlocked, dijkstra, routePoints, buildObs, safetyFilter, safetyFilter2, substep, grid, physics };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.Sim = api;
})(typeof window !== 'undefined' ? window : globalThis);
