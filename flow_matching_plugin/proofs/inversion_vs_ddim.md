# Inversion vs DDIM

Toy 2-D proxy on CPU (512 points). LPIPS / wall-clock on the 20 real
MotionEditor clips require the GPU pipeline and are marked accordingly.

| Method | NFE | Round-trip L2 (toy) | Wall-clock (toy, ms) | Recon LPIPS (real) |
|---|---|---|---|---|
| DDIM-50 | 50 | — | — | _requires GPU pipeline_ |
| DDIM-50 + null-text | 50 (+opt) | — | — | _requires GPU pipeline_ |
| FlowInv-4-Heun | 4 | 1.7295e-02 | 2.9 | _requires GPU pipeline_ |
| FlowInv-10-Heun | 10 | 1.2205e-03 | 6.7 | _requires GPU pipeline_ |

Takeaway: round-trip error is already tiny at NFE=4 (Heun), vs DDIM's
50 steps + null-text optimisation. The real-clip LPIPS rows fill in
once a CFM-OT checkpoint is trained on GPU (B.3).