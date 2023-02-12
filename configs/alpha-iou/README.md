## Alpha-IoU: A Family of Power Intersection over Union Losses for Bounding Box Regression

[Alpha-IoU: A Family of Power Intersection over Union Losses for Bounding Box Regression](http://arxiv.org/abs/2110.13675)

## Abstract

Bounding box (bbox) regression is a fundamental task in computer vision. So far, the most commonly used loss functions for bbox regression are the Intersection over Union (IoU) loss and its variants. In this paper, we generalize existing IoU-based losses to a new family of power IoU losses that have a power IoU term and an additional power regularization term with a single power parameter $\alpha$. We call this new family of losses the $\alpha$-IoU losses and analyze properties such as order preservingness and loss/gradient reweighting. Experiments on multiple object detection benchmarks and models demonstrate that $\alpha$-IoU losses, 1) can surpass existing IoU-based losses by a noticeable performance margin; 2) offer detectors more flexibility in achieving different levels of bbox regression accuracy by modulating $\alpha$; and 3) are more robust to small datasets and noisy bboxes.

## Results and Models

| Method    | angle  | Backbone | Lr schd | Dataset            | preprocess    |  NA  |  BS  | loss           | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| --------- | ------ | -------- | ------- | ------------------ | ------------- | :--: | :--: | -------------- | ---------- | ----------- | ----- |
| RetinaNet | v1(oc) | ResNet50 | 1x      | DOTA-v1.0,trainval | 1024x1024,200 |  1   |  2   | $\alpha$-RIoU  | 70.16      | 40.69       | 40.36 |
| RetinaNet | v1(oc) | ResNet50 | 1x      | DOTA-v1.0,trainval | 1024x1024,200 |  1   |  2   | $\alpha$-RGIoU | 69.47      | 40.40       | 40.03 |
| RetinaNet | v1(oc) | ResNet50 | 1x      | DOTA-v1.0,trainval | 1024x1024,200 |  1   |  2   | $\alpha$-RDIoU | 69.28      | 40.20       | 39.94 |