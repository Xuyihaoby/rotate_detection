# Multi-task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics

[Multi-task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://ieeexplore.ieee.org/document/8578879/)

## Abstract

Numerous deep learning applications beneﬁt from multitask learning with multiple regression and classiﬁcation objectives. In this paper we make the observation that the performance of such systems is strongly dependent on the relative weighting between each task’s loss. Tuning these weights by hand is a difﬁcult and expensive process, making multi-task learning prohibitive in practice. We propose a principled approach to multi-task deep learning which weighs multiple loss functions by considering the homoscedastic uncertainty of each task. This allows us to simultaneously learn various quantities with different units or scales in both classiﬁcation and regression settings. We demonstrate our model learning per-pixel depth regression, semantic and instance segmentation from a monocular input image. Perhaps surprisingly, we show our model can learn multi-task weightings and outperform separate models trained individually on each task.

## Results and Models

| Method | Backbone | Angle  | Loss  | Lr schd | Dataset         | preprocess    | sep  | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| ------ | -------- | ------ | ----- | ------- | --------------- | ------------- | :--: | ---------- | ----------- | ----- |
| retina | ResNet50 | v1(oc) | R_IoU | 1x      | DOTA-v1.0,train | 1024x1024,512 |  Y   | 68.07      | 39.38       | 39.40 |
| retina | ResNet50 | v1(oc) | R_IoU | 1x      | DOTA-v1.0,train | 1024x1024,512 |  N   | 68.30      | 40.29       | 39.93 |

Sep: means have two individual uncertain branch for regress and class