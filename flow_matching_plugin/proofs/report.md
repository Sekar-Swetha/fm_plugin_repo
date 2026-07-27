# CFM-OT empirical proof

## Setup
- Toy data: two-moons, 2-D, 2048 points.
- Model: 3-layer MLP, hidden 128, ~33794 params.
- Training: 4000 Adam steps, lr 0.002, batch 128.
- Loss: CFM-OT with sigma_min = 0.0.

## Loss
- Loss(first 200 steps avg)  = **1.1639**
- Loss(last 200 steps avg)   = **0.9849**
- Ratio (first / last)        = **1.2x**

## Sample distribution (mean, std per dim)
- Data:                       mean=[0.5005566477775574, 0.25019484758377075], std=[0.8877577781677246, 0.4997972846031189]
- 1-step Euler samples:       mean=[0.4908592700958252, 0.20184426009655], std=[0.07103163748979568, 0.08958760648965836]
- 10-step Euler samples:      mean=[0.46433737874031067, 0.2505809962749481], std=[0.8330025672912598, 0.4615219235420227]

## Path straightness
- Mean per-step deviation from straight-line interpolation: **0.1921**