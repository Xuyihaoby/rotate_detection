# Adaptive Rotated Convolution for Rotated Object Detection

[Adaptive Rotated Convolution for Rotated Object Detection](http://arxiv.org/abs/2303.07820)

## Abstract 

Rotated object detection aims to identify and locate objects in images with arbitrary orientation. In this scenario, the oriented directions of objects vary considerably across different images, while multiple orientations of objects exist within an image. This intrinsic characteristic makes it challenging for standard backbone networks to extract high-quality features of these arbitrarily orientated objects. In this paper, we present Adaptive Rotated Convolution (ARC) module to handle the aforementioned challenges. In our ARC module, the convolution kernels rotate adaptively to extract object features with varying orientations in different images, and an efficient conditional computation mechanism is introduced to accommodate the large orientation variations of objects within an image. The two designs work seamlessly in rotated object detection problem. Moreover, ARC can conveniently serve as a plug-and-play module in various vision backbones to boost their representation ability to detect oriented objects accurately. Experiments on commonly used benchmarks (DOTA and HRSC2016) demonstrate that equipped with our proposed ARC module in the backbone network, the performance of multiple popular oriented object detectors is significantly improved (e.g. +3.03% mAP on Rotated RetinaNet and +4.16% on CFA). Combined with the highly competitive method Oriented R-CNN, the proposed approach achieves state-of-the-art performance on the DOTA dataset with 81.77% mAP.

## Results and Models

| Method     | Backbone  | Angle     | Loss  | Lr schd | Dataset         | bs   | preprocess    | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| ---------- | --------- | --------- | ----- | ------- | --------------- | ---- | ------------- | ---------- | ----------- | ----- |
| gfl_retina | arc_res50 | v2(le135) | R_IoU | 1x      | DOTA-v1.0,train | 2    | 1024x1024,512 | 72.44      | 43.88       | 42.60 |