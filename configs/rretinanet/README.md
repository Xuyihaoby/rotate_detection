# Focal Loss for Dense Object Detection

[Focal Loss for Dense Object Detection](http://arxiv.org/abs/1708.02002)

## Abstract

The highest accuracy object detectors to date are based on a two-stage approach popularized by R-CNN, where a classifier is applied to a sparse set of candidate object locations. In contrast, one-stage detectors that are applied over a regular, dense sampling of possible object locations have the potential to be faster and simpler, but have trailed the accuracy of two-stage detectors thus far. In this paper, we investigate why this is the case. We discover that the extreme foreground-background class imbalance encountered during training of dense detectors is the central cause. We propose to address this class imbalance by reshaping the standard cross entropy loss such that it down-weights the loss assigned to well-classified examples. Our novel Focal Loss focuses training on a sparse set of hard examples and prevents the vast number of easy negatives from overwhelming the detector during training. To evaluate the effectiveness of our loss, we design and train a simple dense detector we call RetinaNet. Our results show that when trained with the focal loss, RetinaNet is able to match the speed of previous one-stage detectors while surpassing the accuracy of all existing state-of-the-art two-stage detectors. Code is at: https://github.com/facebookresearch/Detectron.

## Results and Models

| Method             | angle     | Backbone    | Lr schd | Dataset            | preprocess    |  NA  |  BS  | loss     | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| ------------------ | --------- | ----------- | ------- | ------------------ | ------------- | :--: | :--: | -------- | ---------- | ----------- | ----- |
| RetinaNet          | v1(oc)    | ResNet18    | 1x      | DOTA-v1.0,trainval | 1024x1024,200 |  1   |  2   | RIoU     | 65.04      | 36.10       | 36.50 |
| RetinaNet          | v1(oc)    | ResNet18    | 2x      | DOTA-v1.0,trainval | 1024x1024,200 |  1   |  2   | RIoU     | 68.79      | 39.21       | 39.01 |
| RetinaNet          | v1(oc)    | ResNet18    | 3x      | DOTA-v1.0,trainval | 1024x1024,200 |  1   |  2   | RIoU     | 69.29      | 39.45       | 39.75 |
| RetinaNet          | v1(oc)    | MobileNetv2 | 1x      | DOTA-v1.0,trainval | 1024x1024,200 |  1   |  2   | RIoU     | 63.54      | 33.00       | 34.21 |
| RetinaNet          | v1(oc)    | ResNet50    | 1x      | DOTA-v1.0,trainval | 1024x1024,200 |  1   |  2   | RIoU     | 69.02      | 37.58       | 38.48 |
| RetinaNet          | v1(oc)    | ResNet50    | 1x      | DOTA-v1.0,trainval | 1024x1024,200 |  9   |  2   | RIoU     | 68.82      | 38.94       | 39.49 |
| RetinaNet          | v1(oc)    | ResNet50    | 1x      | DOTA-v1.0,train    | 1024x1024,512 |  9   |  2   | RIoU     | 67.16      | 36.69       | 37.95 |
| RetinaNet          | v1(oc)    | ResNet50    | 1x      | DOTA-v1.0,train    | 1024x1024,512 |  9   |  2   | RGIoU    | 65.66      | 38.79       | 38.16 |
| RetinaNet          | v1(oc)    | ResNet50    | 1x      | DOTA-v1.0,train    | 1024x1024,512 |  9   |  2   | smoothL1 | 64.11      | /           | /     |
| RetinaNet          | v1(oc)    | ResNet50    | 2x      | DOTA-v1.0,train    | 1024x1024,512 |  9   |  2   | smoothL1 | 66.36      | /           | /     |
| RetinaNet          | v2(le135) | ResNet50    | 1x      | DOTA-v1.0,trainval | 1024x1024,200 |  9   |  2   | L1       | 67.45      | 36.44       | 38.20 |
| RetinaNet          | v3(le90)  | ResNet50    | 1x      | DOTA-v1.0,trainval | 1024x1024,200 |  9   |  2   | L1       | 67.70      | 41.41       | 40.23 |
| RetinaNet-(simOTA) | v1(oc)    | ResNet50    | 1x      | DOTA-v1.0,trainval | 1024x1024,200 |  1   |  2   | RIoU     | 69.07      | 38.01       | 38.55 |
| RetinaNet-(pafpn)  | v1(oc)    | ResNet50    | 1x      | DOTA-v1.0,train    | 1024x1024,512 |  1   |  2   | RIoU     | 68.03      | 40.61       | 39.77 |
| RetinaNet          | v1(oc)    | ResNet50    | 1x      | DOTA-v1.0,trainval | 1024x1024,512 |  9   |  2   | smoothL1 | 71.0       | /           | /     |

NA: num_anchors; BS: batch_size