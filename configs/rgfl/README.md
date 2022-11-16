# Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection

[Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection](http://arxiv.org/abs/2006.04388)

## Abstract

One-stage detector basically formulates object detection as dense classification and localization. The classification is usually optimized by Focal Loss and the box location is commonly learned under Dirac delta distribution. A recent trend for one-stage detectors is to introduce an individual prediction branch to estimate the quality of localization, where the predicted quality facilitates the classification to improve detection performance. This paper delves into the representations of the above three fundamental elements: quality estimation, classification and localization. Two problems are discovered in existing practices, including (1) the inconsistent usage of the quality estimation and classification between training and inference and (2) the inflexible Dirac delta distribution for localization when there is ambiguity and uncertainty in complex scenes. To address the problems, we design new representations for these elements. Specifically, we merge the quality estimation into the class prediction vector to form a joint representation of localization quality and classification, and use a vector to represent arbitrary distribution of box locations. The improved representations eliminate the inconsistency risk and accurately depict the flexible distribution in real data, but contain continuous labels, which is beyond the scope of Focal Loss. We then propose Generalized Focal Loss (GFL) that generalizes Focal Loss from its discrete form to the continuous version for successful optimization. On COCO test-dev, GFL achieves 45.0\% AP using ResNet-101 backbone, surpassing state-of-the-art SAPD (43.5\%) and ATSS (43.6\%) with higher or comparable inference speed, under the same backbone and training settings. Notably, our best model can achieve a single-model single-scale AP of 48.2\%, at 10 FPS on a single 2080Ti GPU. Code and models are available at https://github.com/implus/GFocal.

## Results and Models

| Method | Backbone | Angle     | Loss  | Lr schd | Dataset            | preprocess    | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| ------ | -------- | --------- | ----- | ------- | ------------------ | ------------- | ---------- | ----------- | ----- |
| rgfl   | ResNet50 | v2(le135) | R_IoU | 1x      | DOTA-v1.0,train    | 1024x1024,512 | 72.03      | 42.70       | 41.33 |
| rgfl   | ResNet50 | v2(le135) | R_IoU | 1x      | DOTA-v1.0,trainval | 1024x1024,824 | 73.94      | 43.31       | 42.92 |

**note**:In this part, we just utilize the method of the regression rather than loss function.