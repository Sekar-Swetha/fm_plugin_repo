# Reflow empirical proof (Contribution C1)

Toy 2-D two-cluster transport, CPU. Round 0 = 1-rectified flow
(Contribution A). Round 1 = one C1 reflow round on the round-0 model's
own (z0 -> transport(z0)) pairs.

## Path straightness
- Round 0 path length: **3.5262**
- Round 1 path length: **3.3835**  (shorter ✓)
- Straight-line lower bound: 3.4364
  (See path_straightness_reflow.png.)

## Few-step sample quality (mean distance to data, lower better)
| NFE | Round 0 | Round 1 |
|----:|--------:|--------:|
| 1 | 2.4693 | 0.2271 |
| 2 | 1.0041 | 0.1449 |
| 4 | 0.1528 | 0.0838 |
| 8 | 0.0399 | 0.0543 |
| 16 | 0.0265 | 0.0396 |

- At NFE=1, reflow improves quality by 90.8% on this toy problem.
- The real-clip CLIP/LPIPS-T-vs-NFE rows require the GPU pipeline (C.6).