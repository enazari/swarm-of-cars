# Diffusion policy for cars: work log
Started Oct 23 23:57 (budget 8 h, ends about Oct 24 07:50). 2x RTX 4080.

Plan: python car sim -> privileged teacher (prioritized sampling MPC, sees everything, can wait/reverse/detour)
-> dataset (+DAgger) -> flow-matching diffusion policy (+ plain regression baseline) -> closed-loop eval -> export.

## Progress
- 00:10 sim (core.py, env.py), observations (ego crop 4x32x32 + route/goal vec + 6 neighbour tokens), teacher.py (prioritized sampling MPC, 512 samples, 3 s horizon, reverse + bang-bang + creep samples).
- 01:10 teacher on 96 held-out scenes: 98.7% cars arrive, 1 car-car contact in 1338 cars, 0 car-card; mean time to goal 7.2 s.
- 00:20 pipeline smoke-tested end to end (data gen, train ~80 it/s at bs 512, closed-loop eval in parallel on CPU). Rule baseline (like the page driver) 86% arrive (crowd 70%) on 8 scenes.
- 00:42 dataset r0: 1.06M samples / 2400 scenes (noise-injected teacher). Training started: diff_r0 (GPU0), bc_r0 (GPU1), 120k steps, bs 512.
- 01:02 round 0 trained (120k steps). Test set: 96 unseen scenes, 1338 cars.
  teacher 98.7% arrive, cc 1, cd 0 | rule-based (page-like) 75.0% w/ safety (cc 39, cd 12), 99.7% without (cc 258, cd 49)
  diff_r0: 99.6% arrive, cc 202, cd 12 (no safety) | 99.3%, cc 55, cd 6 (safety) ; bc_r0: 98.3%, cc 183 | 93.8% (crowd 90.8%), cc 58 (safety)
  => diffusion >> regression once the safety filter is on (bc jams), but contacts are still far above the teacher. Next: DAgger.
- 01:20 DAgger r1 data (student drives, teacher labels): 387k samples / 1200 scenes. Loss weighting (first 2 chunk steps x3) tested: diff_r0w (60k steps) 99.85% arrive with safety, cc 57, cd 5, safety overrides 1138 (vs 4771 for diff_r0). Round-1 training started (r0+r1, init from r0w / bc_r0).
- 01:45 stronger safety rule (all cars assumed to brake after the next step; both cars of a threatened pair brake): diff_r0w + rule 99.7% arrive, cc 1, cd 1 on the 96 scenes; page-like rule-based driver + same rule only 75% (crowd 55%).
  hard set (48 crowded scenes, gaps 44-58 px, 20-28 cars): rule+safety 44.5% arrive; teacher 98.9% (cc 0, cd 1); diff_r0w + safety 99.65% (cc 1, cd 4).
  round 1 (r0+r1 data): diff_r1 99.4% raw (cc 140) / 99.48% with safety (cc 0, cd 1); bc_r1 98.9% raw / 98.3% with safety.
  finding: executing only the first action of each sampled chunk (replan at 4 Hz) cuts raw contacts 85 -> 20 on 48 scenes at equal arrival; denoising steps 4..8 enough.
  Round 2 DAgger launched with exec_n=1 and 30% hard scenes.
- 02:20 DAgger r2 data: 948k samples / 1600 scenes (30% hard). Round-2 training (r0+r1+r2 = 2.4M samples, first-step weight x4, second x2, 120k steps) for diff_r2 (GPU0) and bc_r2 (GPU1).
- 02:40 round 2 (exec_n=1, 8 steps): diff_r2 raw 99.33% arrive, cc 30, cd 1 | +safety v2: 99.1%, cc 0, cd 0, 175 overrides (96 scenes). Hard (48 scenes): raw 99.29% cc 78 cd 11 | v2: 98.77%, cc 0, cd 2.
  bc_r2: v2 98.28% (std), 98.32% (hard) -> diffusion ahead by 0.5-0.8 pts of arrival, similar collision numbers. teacher reverses in 28% of cars on hard scenes.
- 03:00 tried best-of-8 sampling with a safety/progress selector (diffusion-only trick): 98.5% arrive vs 99.7% single sample (48 scenes, v2) -> greedy picks rush into jams; not used (negative result).
  diff_big (w768, 8 blocks, 150k steps, GPU0) and DAgger r3 (student diff_r2, 1200 scenes) running.
- 03:15 DAgger r3 data: 708k samples / 1200 scenes (student diff_r2, exec 1). Training diff_r3 + bc_r3 (r0..r3 = 3.1M samples, init from r2) on GPU1; diff_big continues on GPU0.
- 03:35 diff_big (w768, 8 blocks, r0-r2, 150k steps): std raw 99.4% cc 17 | v2 99.48% cc 0 cd 1 (117 overrides); hard raw 99.21% cc 59 | v2 99.03% cc 0 cd 2 -> better than diff_r2 (99.1 / 98.77 with v2). Fine-tuning diff_big_r3 on r0-r3 (GPU0).
- 03:52 all training done (diff_r3 small, bc_r3, diff_big_r3). Final evaluation running (final_eval.sh -> results.txt): 192 standard + 96 hard unseen scenes.
- 04:20 FINAL EVAL (results.txt). std 192 scenes / 2766 cars; hard 96 scenes / 2275 cars. diff_big_r3 + safety: 99.35% arrive, cc 0, cd 1 (std); 99.03%, cc 0, cd 3 (hard).
  bc_r3 + safety: 98.59% (std), 97.32% (hard); rule + safety: 75.7% / 42.9%; teacher 98.84% / 98.73%.
- 04:35 DONE. Final system = diffusion policy (final/diffusion_policy_big.pt or _small.pt) + stronger safety rule. Browser demo in web/ (small model default, ?model=big).
  Extras: steps ablation (2/4/8/16 -> 99.18/99.25/99.40/98.95%), student reverses in 45% of cars on hard scenes, README.md with results and limits.
