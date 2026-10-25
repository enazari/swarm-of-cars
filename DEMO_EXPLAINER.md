# Rule-based vs rule-based + diffusion (what the demo shows)

**The setup.** Cars start at your mouse and drive to the tunnel entrance of a card. A classic search (Dijkstra, like a GPS) works out the shortest route around the cards. Both systems get that same route. Physics is the same for both: cars have limits, collide, bounce and can be grabbed and thrown.

- **Strictly rule-based:** hand-written rules turn the GPS route into gas, brake and steering. Fixed rules, no learning.
- **Rule-based + diffusion:** the same GPS route is an *input* to a small neural network. Every 0.25 s it starts from random noise, denoises a 2-second plan of speed and steering (the denoising you can watch), and the car drives the first step. It was trained by copying a slow expert planner that can see every car.

**What the diffusion model brings.** The GPS route only goes around cards; it knows nothing about other cars and nothing about how much room a car needs to turn. Rules written by hand follow the route and add a few reflexes (here: brake for a car straight ahead). The model learned the rest from an expert planner that sees every car: keep clear of cards, give way, swing around, back out, line up with the tunnel. In the older Python tests a rule driver was either reckless (hundreds of car-car contacts) or, with a brake rule, jammed (only 76% arrived); in this demo's simple traffic setup the brake rule is enough to avoid crashes, so the visible difference is mostly cards (below).

**What you see in the demo.** Every spawn makes two cars at the same spot with the same goal: **R** (rules) and **D** (rules + diffusion). The two worlds run at once and never touch each other. Counters per world: arrived, average trip, contacts, stuck cars (red ring), cars given up.

**Measured in the demo's own setup** (4 seeded runs of 375 s each, up to 40 cars per world, 10 spawns/s, no safety rule; sums over the 4 runs, so ~950-1030 arrivals each):

| | arrived | car-car contacts | car-card contacts | avg trip | gave up |
|---|---|---|---|---|---|
| R (rules) | 941-1028 | 0 | 46-64 | 7.7-7.9 s | 0 |
| D 13.7M | 1023 | 0 | 0 | 7.3 s | 7 |
| D 5.8M | 918 | 3 | 0 | 7.4 s | 21 |
| D 32k (demo default) | 946 | 7 | 0 | 8.6 s | 5 |
| D 2k | 939 | 28 | 12 | 9.1 s | 0 |

**Honest reading.**
- The rules never hit another car (they brake for a car straight ahead) but scrape the cards all the time: they cut corners with no clearance. Diffusion keeps its distance from cards (0 contacts from 32k up).
- Diffusion is not strictly better: the small models bump other cars a little, and a few D cars get stuck for good (the model alone cannot recover from a jam it never saw).
- The largest model is faster than the rules and has no contacts, but its cost per decision (about 53 ms for 20 cars natively, not measured in a browser) is likely too high to run live; the demo uses 32k (`?model=2k|small|big` to switch).
- At 2k parameters the model is mostly a route follower and is worse than the rules at traffic.
