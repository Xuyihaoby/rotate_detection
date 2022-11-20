# PVTv2: Improved Baselines with Pyramid Vision Transformer

[PVTv2: Improved Baselines with Pyramid Vision Transformer](http://arxiv.org/abs/2106.13797)

## Abstract

Transformer recently has shown encouraging progresses in computer vision. In this work, we present new baselines by improving the original Pyramid Vision Transformer (abbreviated as PVTv1) by adding three designs, including (1) overlapping patch embedding, (2) convolutional feed-forward networks, and (3) linear complexity attention layers. With these modifications, our PVTv2 significantly improves PVTv1 on three tasks e.g., classification, detection, and segmentation. Moreover, PVTv2 achieves comparable or better performances than recent works such as Swin Transformer. We hope this work will facilitate state-of-the-art Transformer researches in computer vision. Code is available at https://github.com/whai362/PVT.

## Results and Models

| Method     | Backbone | Angle     | Loss  | Lr schd | Dataset         | preprocess    | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| ---------- | -------- | --------- | ----- | ------- | --------------- | ------------- | ---------- | ----------- | ----- |
| gfl_retina | pvt_tiny | v2(le135) | R_IoU | 1x      | DOTA-v1.0,train | 1024x1024,512 | 69.85      | 41.32       | 40.48 |
| gfl_retina | pvt_b0   | v2(le135) | R_IoU | 1x      | DOTA-v1.0,train | 1024x1024,512 | 72.91      | 44.00       | 42.84 |