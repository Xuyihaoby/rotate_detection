# R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object

[R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object](http://arxiv.org/abs/1908.05612)

## Abstract

Rotation detection is a challenging task due to the difﬁculties of locating the multi-angle objects and separating them accurately and quickly from the background. Though considerable progress has been made, for practical settings, there still exist challenges for rotating objects with large aspect ratio, dense distribution and category extremely imbalance. In this paper, we propose an end-to-end reﬁned single-stage rotation detector for fast and accurate positioning objects. Considering the shortcoming of feature misalignment in existing reﬁned single-stage detector, we design a feature reﬁnement module to improve detection performance by getting more accurate features. The key idea of feature reﬁnement module is to re-encode the position information of the current reﬁned bounding box to the corresponding feature points through feature interpolation to realize feature reconstruction and alignment. Extensive experiments on two remote sensing public datasets DOTA, HRSC2016 as well as scene text data ICDAR2015 show the state-of-the-art accuracy and speed of our detector. Code is available at https://github.com/Thinklab-SJTU/R3Det_Tensorflow .

## Results and Models

| Method | Backbone | Angle  | Loss     | Lr schd | Dataset         | preprocess    | MS   | Aug  | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| ------ | -------- | ------ | -------- | ------- | --------------- | ------------- | ---- | ---- | ---------- | ----------- | ----- |
| R3Det  | ResNet50 | v1(oc) | smoothl1 | 1x      | DOTA-v1.0,train | 1024x1024,512 | \    | \    | 65.36      | 32.94       | 34.96 |

**note**: the model is original version of the paper.