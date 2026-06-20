# Reflow empirical proof (Contribution C1)

Toy 2-D two-cluster transport, CPU. Round 0 = 1-rectified flow
(Contribution A). Round 1 = one C1 reflow round on the round-0 model's
own (z0 -> transport(z0)) pairs.

## Path straightness
- Round 0 path length: **3.5486**
- Round 1 path length: **3.4328**  (shorter ✓)
- Straight-line lower bound: 3.4510
  (See path_straightness_reflow.png.)

## Few-step sample quality (mean distance to data, lower better)
| NFE | Round 0 | Round 1 |
|----:|--------:|--------:|
| 1 | 2.5260 | 0.2671 |
| 2 | 0.9763 | 0.1503 |
| 4 | 0.1368 | 0.0740 |
| 8 | 0.0385 | 0.0416 |
| 16 | 0.0219 | 0.0290 |

- At NFE=1, reflow improves quality by 89.4% on this toy problem.
- The real-clip CLIP/LPIPS-T-vs-NFE rows require the GPU pipeline (C.6).