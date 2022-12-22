# RepPoints: Point Set Representation for Object Detection

[RepPoints: Point Set Representation for Object Detection](https://ieeexplore.ieee.org/document/9009032/)

## Abstract

Modern object detectors rely heavily on rectangular bounding boxes, such as anchors, proposals and the ﬁnal predictions, to represent objects at various recognition stages. The bounding box is convenient to use but provides only a coarse localization of objects and leads to a correspondingly coarse extraction of object features. In this paper, we present RepPoints (representative points), a new ﬁner representation of objects as a set of sample points useful for both localization and recognition. Given ground truth localization and recognition targets for training, RepPoints learn to automatically arrange themselves in a manner that bounds the spatial extent of an object and indicates semantically signiﬁcant local areas. They furthermore do not require the use of anchors to sample a space of bounding boxes. We show that an anchor-free object detector based on RepPoints can be as effective as the state-of-the-art anchor-based detection methods, with 46.5 AP and 67.4 AP50 on the COCO test-dev detection benchmark, using ResNet-101 model. Code is available at https://github.com/microsoft/RepPoints.

## Results and Models

| Method    | Backbone | Lr schd | Dataset         | preprocess    | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| --------- | -------- | ------- | --------------- | ------------- | ---------- | ----------- | ----- |
| reppoints | ResNet50 | 1x      | DOTA-v1.0,train | 1024x1024,512 | 60.29      | 32.51       | 32.87 |