# The GPS analogy: what the model is given and what it decides

Short version: the model is a driver who gets **turn-by-turn GPS directions** and also **watches the road**.
It decides the gas, the brake and the steering wheel. It does not decide which way around the cards to go.

## What the model receives (every 0.25 s, per car)

Everything is in the car's own frame: "front" is the direction the nose points, "sideways" is to the car's left or right.
The page turns screen coordinates into these numbers with the car's heading (cosine and sine); it is plain code in `buildObs` (`web/sim.js`, `core.py`).

| input | size | what it is |
|---|---|---|
| map image | 4 x 32 x 32 | 6 px cells covering 192 px, centred 40 px ahead of the car. Layers: cards and stage edge, other cars, their forward velocity, their sideways velocity |
| vector | 18 | goal (front, sideways) / 100; goal distance / 300; own speed / 75; previous action (speed, steering); **6 route points** |
| nearest cars | 6 x 8 | per car within 250 px: relative position (front, sideways), relative heading (cos, sin), its velocity (front, sideways), parked flag, valid flag; plus a mask |
| noise | 16 | starting point of the denoising, not information about the scene |

## The route points = the GPS

- One route: the shortest path from the car to its goal, around the cards. A grid Dijkstra search computes it (the cousin of A*). It is not learned.
- The 6 route points are samples along that single path, **25, 50, 75, 100, 150 and 200 px** from the car (`ROUTE_D` in `sim.js`).
- Each point is written as 2 numbers (front, sideways) / 100, so the 12 numbers sketch the shape of the road ahead.
- If the goal is closer than a point, that point is set to the goal.
- The route only goes around cards. It knows nothing about other cars.
- If the car sits in a cell the planner marks as blocked (inside the 15 px margin of a card), the page starts the walk from the nearest free cell within 8 cells. That can be on the far side of a wall and gives a misleading route.

## Who does what

| the model | the page (plain code) |
|---|---|
| chooses 8 steps of target speed (negative = reverse) and steering angle; the first step is used, then it re-plans | computes the route (GPS), the map image, the neighbour list |
| slows for and swings around other cars (it sees them in the image and the list) | car limits: acceleration 120 px/s^2, braking 320 px/s^2, speed range -30..75 px/s, steering limit |
| decides when to reverse, how to line up with the tunnel | collision physics, mouse grab/throw, cards shoving cars, the tunnel exit after arrival, removal of a car that stays put for 60 s |

So: **the commands are the model's, the movement is those commands run through the car's physics.**
The part handed to the model for free is the long-range route around obstacles.

## Consequences to remember

- Removing the route points means retraining. The model would have to find routes from the local image alone; expect it to get stuck behind cards. The 2k model leans on the hint heavily.
- A car in a spot where the route is wrong (blocked cell, sealed gap) can stay stuck, because the model alone does not recover from a situation it never saw in training. Fix by training (DAgger with jammed starts), not by programmed driving.
- Each car only ever gets one route. The front-only / open route maps I once built per card are leftovers and make almost no difference with the notch tunnel.

## Why not just an explicit controller on the route? (learned model vs rule-based driver)

The route is given to the model, so a fixed controller (steer toward a route point, speed from distance, no learnable parameters) can follow it. The question is what learning adds.

**What a rule-based driver does** (`rule` in `results.txt`, the page's old driver; standard set, 2766 cars):
- Without a safety rule: 99.9% arrive, but 545 car-car contacts (0.20 per car). It ignores traffic.
- With a brake-if-about-to-crash rule: 0 contacts but only 75.7% arrive (42.9% on the hard set). It jams, because braking never resolves a deadlock.
- So an explicit driver is either reckless or stuck.

**What the learned model adds:** behaviour toward other cars that the route does not contain (yielding, waiting, backing out of a jam, going around). It copies it from the teacher, which plans with the other cars in view.
- 13.7M model, no safety rule: 0.021 car-car contacts per car (115 on 5470 cars), about 10x fewer than the rule-based driver, with about the same arrival.

**Raw car-car contacts per car by model size** (no safety rule; `results_ladder.txt`, 384 standard scenes = 5470 cars; the rule-based figure is from the older 192-scene set, so the sets overlap but are not identical):

| driver | per car |
|---|---|
| rule-based driver | 0.20 |
| 13.7M | 0.021 |
| 32k | 0.097 |
| 2k | 0.248 |
| 1k | 0.38 |

**Caveat to keep in mind:** at 2k parameters the model avoids other cars no better than the explicit driver. It is mostly a route follower plus randomness; its high arrival rate comes largely from the route hint (and from the safety rule in the earlier tests). The clearly learned behaviour shows up at 32k and above.

**Not done yet:** the pure-pursuit controller in the demo's "A* route follower" toggle has never been run through the Python evaluation. Proper comparison = run it on the same 384 standard and 192 hard scenes next to every model size, to see at which size the model starts to beat the explicit controller.
