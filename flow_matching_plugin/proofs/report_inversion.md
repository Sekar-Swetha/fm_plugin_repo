# Flow inversion empirical proof (Contribution B)

## Round-trip error vs steps
| N | Euler | Heun |
|---|---|---|
| 1 | 9.5202e-01 | 8.1623e-01 |
| 2 | 7.4708e-01 | 9.9663e-02 |
| 4 | 4.1066e-01 | 1.7295e-02 |
| 8 | 2.0800e-01 | 2.3601e-03 |
| 16 | 1.0443e-01 | 3.0097e-04 |
| 32 | 5.2341e-02 | 3.7829e-05 |

- Euler error scales ~O(1/N); Heun ~O(1/N^2) (see roundtrip_vs_steps.png).
- Heun @ N=8 round-trip L2 = **2.3601e-03**.

## Path straightness
- Max deviation from straight line along inversion trajectory: **2.5830e-01** (inversion_trace.png).

See inversion_vs_ddim.md for the DDIM comparison (real rows are GPU-only).