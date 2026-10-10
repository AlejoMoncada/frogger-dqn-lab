# Leaderboard: run_2m

20 greedy episodes per checkpoint (epsilon=0.01, seeds 900000+i), random policy as baseline.

| Checkpoint | Episodes | Score | Max rows (mean / max) | Homes (mean / max) | 🐸 first_hop | 🥉 road_crossed | 🪵 first_log | 🥈 mid_river | 🥇 frog_home* | 🏆 three_homes* | 👑 level_cleared* |
|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 20 | 9.4 ± 2.3 | 3.3 / 7 | 0.00 / 0 | 🐸 100% | 5% | 5% | 0% | 0% | 0% | 0% |
| 100,000 | 20 | 12.2 ± 1.9 | 3.5 / 7 | 0.00 / 0 | 🐸 100% | 5% | 5% | 0% | 0% | 0% | 0% |
| 250,000 | 20 | 20.6 ± 3.2 | 7.7 / 10 | 0.00 / 0 | 🐸 100% | 🥉 95% | 🪵 95% | 20% | 0% | 0% | 0% |
| 500,000 | 20 | 18.0 ± 2.0 | 8.5 / 9 | 0.00 / 0 | 🐸 100% | 🥉 90% | 🪵 90% | 🥈 90% | 0% | 0% | 0% |
| 1,000,000 | 20 | 32.0 ± 4.3 | 9.7 / 10 | 0.00 / 0 | 🐸 100% | 🥉 100% | 🪵 100% | 🥈 95% | 0% | 0% | 0% |
| 2,000,000 | 20 | 39.3 ± 2.9 | 9.9 / 10 | 0.00 / 0 | 🐸 100% | 🥉 100% | 🪵 100% | 🥈 100% | 0% | 0% | 0% |

Medal thresholds (medal emoji shown when at least 50% of episodes achieved it):

- 🐸 `first_hop`: reach row 1 in one life (home row is 12)
- 🥉 `road_crossed`: reach row 6 in one life (home row is 12)
- 🪵 `first_log`: reach row 7 in one life (home row is 12)
- 🥈 `mid_river`: reach row 9 in one life (home row is 12)
- 🥇 `frog_home`*: 1 frog(s) home in one episode
- 🏆 `three_homes`*: 3 frog(s) home in one episode
- 👑 `level_cleared`*: 5 frog(s) home in one episode

* Unverified proxy: home detection is not validated against RAM traces, so this medal is an estimate.
