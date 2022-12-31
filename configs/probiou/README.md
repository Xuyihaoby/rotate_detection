# Gaussian Bounding Boxes and Probabilistic Intersection-over-Union for Object Detection

[Gaussian Bounding Boxes and Probabilistic Intersection-over-Union for Object Detection](http://arxiv.org/abs/2106.06072)

## Abstract

Most object detection methods use bounding boxes to encode and represent the object shape and location. In this work, we explore a fuzzy representation of object regions using Gaussian distributions, which provides an implicit binary representation as (potentially rotated) ellipses. We also present a similarity measure for the Gaussian distributions based on the Hellinger Distance, which can be viewed as a Probabilistic Intersection-over-Union (ProbIoU). Our experimental results show that the proposed Gaussian representations are closer to annotated segmentation masks in publicly available datasets, and that loss functions based on ProbIoU can be successfully used to regress the parameters of the Gaussian representation. Furthermore, we present a simple mapping scheme from traditional (or rotated) bounding boxes to Gaussian representations, allowing the proposed ProbIoU-based losses to be seamlessly integrated into any object detector.

## Results and Models

| Method    | angle  | Backbone | Lr schd | Dataset         | preprocess    |  BS  | loss       | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| --------- | ------ | -------- | ------- | --------------- | ------------- | :--: | ---------- | ---------- | ----------- | ----- |
| RetinaNet | v1(oc) | ResNet50 | 1x      | DOTA-v1.0,train | 1024x1024,512 |  4   | PrbIoU(l1) | 67.80      | 36.27       | 37.87 |
| RetinaNet | v1(oc) | ResNet50 | 1x      | DOTA-v1.0,train | 1024x1024,512 |  4   | PrbIoU(l2) | 67.61      | 29.89       | 35.18 |