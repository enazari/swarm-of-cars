# Swarm of cars: rule-based vs rule-based + diffusion

A browser demo and training code for a small flow-matching ("diffusion") driving policy. Cars spawn at the mouse and drive to a tunnel
entrance cut into the bottom edge of a card, around the cards and around each other. Every spawn releases **two cars with the same goal**:

- **R**: a strictly rule-based driver (pure pursuit on a Dijkstra route, a speed rule, braking for a car straight ahead).
- **D**: the same route given as an *input* to a neural network that starts from noise every 0.25 s, denoises a 2 s plan of
  (target speed, steering) and drives the first step.

Both get the same physics (rigid-body collisions, grab and throw with the mouse, cards that shove cars). There is no safety rule in the page.
Details: [`DEMO_EXPLAINER.md`](DEMO_EXPLAINER.md) (what the demo shows) and [`GPS_ANALOGY.md`](GPS_ANALOGY.md) (exactly what the model sees and decides).

## Run the demo

```bash
cd web && python3 -m http.server 8001      # open http://localhost:8001 (hard-reload after updates)
```

The page loads onnxruntime-web from a CDN. The default model is the 32k-parameter one; `?model=2k|small|big` switches.
The World button has four modes (only R, only D, both in the same space but not interacting, one shared traffic).
Other buttons: plan display (slow or live), drive-per-plan, a denoising-step inspector, and an optional "unstick" mode (`?unstick=1`).

## Results (headless runs of the demo's own setup)

4 seeded runs of 375 s, up to 40 cars per world, 10 spawns/s, no safety rule; sums over the 4 runs.

| | arrived | car-car contacts | car-card contacts | avg trip | gave up |
|---|---|---|---|---|---|
| R (rules) | 941-1028 | 0 | 46-64 | 7.7-7.9 s | 0 |
| D 13.7M | 1023 | 0 | 0 | 7.3 s | 7 |
| D 5.8M | 918 | 3 | 0 | 7.4 s | 21 |
| D 32k (default) | 946 | 7 | 0 | 8.6 s | 5 |
| D 2k | 939 | 28 | 12 | 9.1 s | 0 |

Python closed-loop evaluation (384 standard scenes, 5470 cars; 192 hard scenes, 4603 cars; safety rule on):
diffusion keeps 99%+ arrival from 13.7M down to 2k parameters (about 98.7% on the hard set at 2k), while a regression policy of the same
design collapses when small (32k: 94.6% / 87.2%, 2k: 74.8% / 41.9%). See `results_ladder_table.txt` and `results_ladder.png`.

Honest reading:

- The rules never hit other cars (they brake for a car ahead) but scrape cards. Diffusion keeps clear of cards from 32k up.
- Small models bump other cars a little. Without a safety rule the 2k model avoids cars no better than the rules (0.25 vs 0.20 contacts per car);
  the 13.7M model is about 10x better (0.021). The clearly learned traffic behaviour shows from 32k up.
- A few D cars get stuck for good in jams the model never saw in training. The fix is more training data of that kind, not programmed driving.
- Browser speed is not measured. Native ONNX on one CPU thread for 20 cars: 13.7M about 53 ms, 5.8M 22 ms, 32k 0.6 ms, 2k 0.45 ms.

## How it works

- **Observation per car** (own frame): a 4x32x32 map (cards, cars, their velocities; 6 px cells, centred 40 px ahead), an 18-number vector
  (goal, speed, last action, 6 route points from a Dijkstra field), and the 6 nearest cars as tokens.
- **Action:** 8 steps of (target speed in [-30, 75] px/s, steering), 0.25 s each; the page uses the first step and re-plans.
- **Teacher:** a privileged sampling MPC that sees every car (512 plans, 3 s horizon, reverse and three-point-turn samples).
- **Training:** 3.1M teacher samples, one round of teacher data with injected noise plus three DAgger rounds. Flow matching, 8 Euler steps.
- **Programmed parts:** the route hint, car physics, the tunnel arrival and exit script, removal of a car that stays put for 60 s, and the rule-based R driver.

## Files

| file | what |
|---|---|
| `web/` | the demo: `index.html`, `sim.js` (JavaScript simulator, checked against Python with `node web/parity_test.js`, which should print zeros), ONNX models |
| `final/` | PyTorch checkpoints: diffusion policy at 13.7M / 5.8M / 1M / 340k / 100k / 32k / 2k parameters, and the regression baseline |
| `core.py`, `env.py` | car physics, grid fields, observations, random scenes and the multi-car environment |
| `teacher.py` | the privileged teacher |
| `gen_data.py`, `dagger_gen.py`, `consolidate.py`, `policy.py` | teacher data generation, DAgger rounds, dataset merging, closed-loop student policy |
| `model.py`, `train.py` | encoder and denoiser (width options `--ec`, `--d`, `--pool`), training |
| `eval_policy.py`, `eval_ladder.sh`, `summarize_ladder.py`, `run_jobs.sh` | closed-loop evaluation and the model-size ladder |
| `export.py` | ONNX export with all denoising steps unrolled |
| `viz_*.py`, `denoising_*.png` | denoising visualisations |
| `LOG.md` | what was done, in order, with numbers |

Training data (13 GB) and intermediate checkpoints are not in the repository.

## Limits

- Layouts need gaps of at least about 45 px between cards. The 2k model was never trained on notches or crowds (training scenes had up to 28 cars).
- At very high density D trips become much slower than R's.
- The student imitates the teacher; it is not expected to beat it.
