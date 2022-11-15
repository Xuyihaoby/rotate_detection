# Beyond Bounding-Box: Convex-hull Feature Adaptation for Oriented and Densely Packed Object Detection

[Beyond Bounding-Box: Convex-hull Feature Adaptation for Oriented and Densely Packed Object Detection](https://ieeexplore.ieee.org/document/9578090/)

## Abstract

Detecting oriented and densely packed objects remains challenging for spatial feature aliasing caused by the intersection of reception ﬁelds between objects. In this paper, we propose a convex-hull feature adaptation (CFA) approach for conﬁguring convolutional features in accordance with oriented and densely packed object layouts. CFA is rooted in convex-hull feature representation, which deﬁnes a set of dynamically predicted feature points guided by the convex intersection over union (CIoU) to bound the extent of objects. CFA pursues optimal feature assignment by constructing convex-hull sets and dynamically splitting positive or negative convex-hulls. By simultaneously considering overlapping convex-hulls and objects and penalizing convex-hulls shared by multiple objects, CFA alleviates spatial feature aliasing towards optimal feature adaptation. Experiments on DOTA and SKU110KR datasets show that CFA signiﬁcantly outperforms the baseline approach, achieving new state-of-the-art detection performance. Code is available at github.com/SDLGuoZonghao/BeyondBoundingBox.

## Results and Models

| Method | Backbone | Angle   | Loss      | Lr schd | Dataset         | preprocess    | pos_weight     | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| ------ | -------- | ------- | --------- | ------- | --------------- | ------------- | -------------- | ---------- | ----------- | ----- |
| cfa    | ResNet50 | v1(oc)  | ConvexIoU | 1x      | DOTA-v1.0,train | 1024x1024,512 | \              | 60.29      | \           | \     |
| cfa    | ResNet50 | v1(loc) | ConvexIoU | 1x      | DOTA-v1.0,train | 1024x1024,512 | anti-aliasing  | 68.10      | 41.63       | 40.71 |
| cfa    | ResNet50 | v1(loc) | ConvexIoU | 1x      | DOTA-v1.0,train | 1024x1024,512 | dynamic_weight | 69.96      | 39.88       | 40.15 |

dw:
$$
cls_{weight}=(1 + (-cls_{loss}[:thr + 1]).exp()-bbox_{loss}[:thr + 1])^2\\
bbox_{weight}=(1 - (-cls_{loss}[:thr + 1]).exp()+bbox_{loss}[:thr + 1])^2
$$
