const Sim = require('./sim.js'); const fs = require('fs');
const cases = JSON.parse(fs.readFileSync(__dirname + '/parity.json'));
let worst = { crop: 0, vec: 0, tok: 0, mask: 0, field: 0, safety: 0, safety2: 0, adv: 0 };
for (const c of cases) {
  const blocked = Sim.buildBlocked(c.cards, c.W, c.H, Sim.INFLATE), { gw, gh } = Sim.grid(c.W, c.H);
  const S = { X: c.X, Y: c.Y, TH: c.TH, V: c.V, state: c.state };
  const field0 = Sim.dijkstra(blocked, gw, gh, Math.trunc(c.goals[0][0] / Sim.CS), Math.trunc(c.goals[0][1] / Sim.CS));
  for (let k = 0; k < field0.length; k++) worst.field = Math.max(worst.field, Math.abs(Math.min(field0[k], 1e8) - c.field0[Math.trunc(k / gw)][k % gw]));
  c.idx.forEach((i, r) => {
    const f = Sim.dijkstra(blocked, gw, gh, Math.trunc(c.goals[i][0] / Sim.CS), Math.trunc(c.goals[i][1] / Sim.CS));
    const crop = new Float32Array(4 * 1024), vec = new Float32Array(18), tok = new Float32Array(48), mask = new Float32Array(6);
    Sim.buildObs(i, S, c.goals, c.cards, c.W, c.H, f, gw, gh, c.prev_a, crop, vec, tok, mask);
    const mx = (a, b) => a.reduce((m, v, q) => Math.max(m, Math.abs(v - b[q])), 0);
    worst.crop = Math.max(worst.crop, mx(crop, c.crop[r])); worst.vec = Math.max(worst.vec, mx(vec, c.vec[r]));
    worst.tok = Math.max(worst.tok, mx(tok, c.tok[r])); worst.mask = Math.max(worst.mask, mx(mask, c.mask[r]));
  });
  const vt = c.vt.slice(), dl = c.dl.slice();
  const ch = Sim.safetyFilter(S, vt, dl, c.cards, c.W, c.H);
  worst.safety = Math.max(worst.safety, Math.abs(ch - c.changed), ...vt.map((v, i) => Math.abs(v - c.vt_f[i])), ...dl.map((v, i) => Math.abs(v - c.dl_f[i])));
  const vt2 = c.vt.slice(), dl2 = c.dl.slice();
  const ch2 = Sim.safetyFilter2(S, vt2, dl2, c.cards, c.W, c.H);
  worst.safety2 = Math.max(worst.safety2, Math.abs(ch2 - c.changed2), ...vt2.map((v, i) => Math.abs(v - c.vt_f2[i])), ...dl2.map((v, i) => Math.abs(v - c.dl_f2[i])));
  const S2 = { X: c.X.slice(), Y: c.Y.slice(), TH: c.TH.slice(), V: c.V.slice(), state: c.state };
  for (let k = 0; k < Sim.SUB; k++) Sim.substep(S2, vt, dl);
  for (const key of ['X', 'Y', 'TH', 'V']) S2[key].forEach((v, i) => { worst.adv = Math.max(worst.adv, Math.abs(v - c.adv[key][i])); });
}
console.log('max abs diff JS vs Python:', worst);
