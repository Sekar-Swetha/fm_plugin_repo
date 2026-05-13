# MediaPipe pose extraction — verification report

- Frames processed: **91**
- Frames with at least one detected keypoint: **91** (100.0%)
- Mean landmark visibility (over detected): **0.966**

## Per-keypoint detection rate (fraction of frames detecting each OP-18 keypoint)

| idx | name | detection rate |
|----:|:-----|---------------:|
| 0 | `nose` | 100.0% |
| 1 | `neck` | 100.0% |
| 2 | `r_shoulder` | 100.0% |
| 3 | `r_elbow` | 100.0% |
| 4 | `r_wrist` | 100.0% |
| 5 | `l_shoulder` | 100.0% |
| 6 | `l_elbow` | 100.0% |
| 7 | `l_wrist` | 100.0% |
| 8 | `r_hip` | 100.0% |
| 9 | `r_knee` | 100.0% |
| 10 | `r_ankle` | 100.0% |
| 11 | `l_hip` | 100.0% |
| 12 | `l_knee` | 100.0% |
| 13 | `l_ankle` | 100.0% |
| 14 | `r_eye` | 100.0% |
| 15 | `l_eye` | 100.0% |
| 16 | `r_ear` | 100.0% |
| 17 | `l_ear` | 100.0% |

## How to read the artefacts

- `side_by_side/` — original frame | (ref OpenPose, if provided) | MediaPipe (ours). Use this to eyeball that joints land in the right places and limb colours match.
- `overlay/` — input frame blended with our skeleton. Use this to spot mis-localised joints.
- `diff/` — pixel-absolute-difference against the OpenPose reference. Used to quantify drift on a per-frame basis.
- `report.json` — machine-readable version of every number above.