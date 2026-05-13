# Attribution — MotionEditor

This subdirectory contains an unmodified copy of the official MotionEditor
implementation, redistributed under its original license for use as the
baseline of a dissertation project at Trinity College Dublin.

## Original work

**MotionEditor: Editing Video Motion via Content-Aware Diffusion**
Shuyuan Tu, Qi Dai, Zhi-Qi Cheng, Han Hu, Xintong Han, Zuxuan Wu, Yu-Gang Jiang.
*CVPR 2024.*

- arXiv: https://arxiv.org/abs/2311.18830
- Project page: https://francis-rings.github.io/MotionEditor/
- Original repository: https://github.com/Francis-Rings/MotionEditor

## License

The MotionEditor codebase is licensed under the **Apache License 2.0**, as
declared by the badge in the original README. A copy of the Apache 2.0
license is included alongside this NOTICE as `LICENSE.apache-2.0`. All
copyright in the original MotionEditor source belongs to the authors above.

If you use any of the MotionEditor source code from this subdirectory,
please cite the original CVPR 2024 paper:

```bibtex
@inproceedings{tu2024motioneditor,
  title     = {MotionEditor: Editing Video Motion via Content-Aware Diffusion},
  author    = {Tu, Shuyuan and Dai, Qi and Cheng, Zhi-Qi and Hu, Han and Han, Xintong and Wu, Zuxuan and Jiang, Yu-Gang},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024}
}
```

## Modifications

No modifications have been made to MotionEditor's source files in this
subdirectory. All extensions and improvements live in sibling top-level
directories of this repository:

- `flow_matching_plugin/` — replaces MotionEditor's DDPM-epsilon training
  loss with Conditional Flow Matching on Optimal Transport paths
  (Lipman et al. 2023; Liu et al. 2022).
- `mediapipe_motioneditor_plugin/` — drop-in replacement for the
  `controlnet_aux.OpenposeDetector` preprocessing step using MediaPipe
  Pose, with byte-format-compatible OpenPose-COCO-18 skeleton rendering.

Both plugins sit alongside MotionEditor and are invoked instead of the
corresponding upstream entry points; they do not patch upstream files.
