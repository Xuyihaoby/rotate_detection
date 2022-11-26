# The KFIoU Loss for Rotated Object Detection

[The KFIoU Loss for Rotated Object Detection](http://arxiv.org/abs/2201.12558)

## Abstract

Differing from the well-developed horizontal object detection area whereby the computing-friendly IoU based loss is readily adopted and well fits with the detection metrics. In contrast, rotation detectors often involve a more complicated loss based on SkewIoU which is unfriendly to gradient-based training. In this paper, we argue that one effective alternative is to devise an approximate loss who can achieve trend-level alignment with SkewIoU loss instead of the strict value-level identity. Specifically, we model the objects as Gaussian distribution and adopt Kalman filter to inherently mimic the mechanism of SkewIoU by its definition, and show its alignment with the SkewIoU at trend-level. This is in contrast to recent Gaussian modeling based rotation detectors e.g. GWD, KLD that involves a human-specified distribution distance metric which requires additional hyperparameter tuning. The resulting new loss called KFIoU is easier to implement and works better compared with exact SkewIoU, thanks to its full differentiability and ability to handle the non-overlapping cases. We further extend our technique to the 3-D case which also suffers from the same issues as 2-D detection. Extensive results on various public datasets (2-D/3-D, aerial/text/face images) with different base detectors show the effectiveness of our approach.

## Results and Models

| Method | Backbone | Angle  | Loss  | Lr schd | Dataset         | preprocess    | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| ------ | -------- | ------ | ----- | ------- | --------------- | ------------- | ---------- | ----------- | ----- |
| retina | ResNet50 | v1(oc) | KFIoU | 1x      | DOTA-v1.0,train | 1024x1024,512 | 67.05      | 29.09       | 34.53 |

