# Rethinking Rotated Object Detection with Gaussian Wasserstein Distance Loss

[Rethinking Rotated Object Detection with Gaussian Wasserstein Distance Loss](http://arxiv.org/abs/2101.11952)

## Abstract

Boundary discontinuity and its inconsistency to the final detection metric have been the bottleneck for rotating detection regression loss design. In this paper, we propose a novel regression loss based on Gaussian Wasserstein distance as a fundamental approach to solve the problem. Specifically, the rotated bounding box is converted to a 2-D Gaussian distribution, which enables to approximate the indifferentiable rotational IoU induced loss by the Gaussian Wasserstein distance (GWD) which can be learned efficiently by gradient back-propagation. GWD can still be informative for learning even there is no overlapping between two rotating bounding boxes which is often the case for small object detection. Thanks to its three unique properties, GWD can also elegantly solve the boundary discontinuity and square-like problem regardless how the bounding box is defined. Experiments on five datasets using different detectors show the effectiveness of our approach. Codes are available at https://github.com/yangxue0827/RotationDetection.

## Results and Models

| Method    | angle  | Backbone | Lr schd | Dataset         | preprocess    |  BS  | loss | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| --------- | ------ | -------- | ------- | --------------- | ------------- | :--: | ---- | ---------- | ----------- | ----- |
| RetinaNet | v1(oc) | ResNet50 | 1x      | DOTA-v1.0,train | 1024x1024,512 |  4   | GWD  | 67.24      | 36.25       | 38.04 |