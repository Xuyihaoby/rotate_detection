# Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection

[ATSS](http://arxiv.org/abs/1912.02424)

## Abstract

Object detection has been dominated by anchor-based detectors for several years. Recently, anchor-free detectors have become popular due to the proposal of FPN and Focal Loss. In this paper, we first point out that the essential difference between anchor-based and anchor-free detection is actually how to define positive and negative training samples, which leads to the performance gap between them. If they adopt the same definition of positive and negative samples during training, there is no obvious difference in the final performance, no matter regressing from a box or a point. This shows that how to select positive and negative training samples is important for current object detectors. Then, we propose an Adaptive Training Sample Selection (ATSS) to automatically select positive and negative samples according to statistical characteristics of object. It significantly improves the performance of anchor-based and anchor-free detectors and bridges the gap between them. Finally, we discuss the necessity of tiling multiple anchors per location on the image to detect objects. Extensive experiments conducted on MS COCO support our aforementioned analysis and conclusions. With the newly introduced ATSS, we improve state-of-the-art detectors by a large margin to $50.7\%$ AP without introducing any overhead. The code is available at https://github.com/sfzhang15/ATSS

## Results and Models

### DOTA

| Method | Backbone | Angle     | Loss  | Lr schd | Dataset            | preprocess    | MS   | RR   | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| ------ | -------- | --------- | ----- | ------- | ------------------ | ------------- | ---- | ---- | ---------- | ----------- | ----- |
| retina | ResNet50 | v1(oc)    | R_IoU | 1x      | DOTA-v1.0,trainval | 1024x1024,824 | \ | \ | 73.15      | 42.26       | 42.06 |
| retina | ResNet50 | v1(oc)    | R_IoU | 2x      | DOTA-v1.0,trainval | 1024x1024,824 | \ | \ | 74.05      | 45.13       | 43.91 |
| retina | ResNet50 | v1(oc)    | R_IoU | 3x      | DOTA-v1.0,trainval | 1024x1024,824 | \ | \ | 73.70      | 45.90       | 44.57 |
| retina | ResNet50 | v2(le135) | R_IoU | 1x      | DOTA-v1.0,trainval | 1024x1024,824 | \ | \ | 73.38      | 42.51       | 42.22 |
| retina | ResNet50 | v2(le135) | R_IoU | 1x | DOTA-v1.0,trainval | 1024x1024,824 | Yes | \ | 77.61 | 47.43 | 46.08 |
| retina | ResNet50 | v2(le135) | R_IoU | 1x | DOTA-v1.0,trainval | 1024x1024,824 | Yes | Yes | 78.91 | 53.63 | 49.63 |
| retina | ResNet101 | v2(le135) | R_IoU | 1x | DOTA-v1.0,trainval | 1024x1024,824 | Yes | Yes | 79.46      | 54.68 | 50.85 |
| retina | ResNet50 | v3(le90)  | R_IoU | 1x      | DOTA-v1.0,trainval | 1024x1024,824 | \ | \ | 73.12      | 42.03       | 42.39 |
| retina | ResNet50 | v3(le90)  | R_IoU | 2x      | DOTA-v1.0,trainval | 1024x1024,824 | \ | \ | 73.28      | 44.95       | 43.67 |
| retina | ResNet50 | v3(le90) | R_IoU | 1x | DOTA-v1.0,trainval | 1024x1024,824 | Yes | \ | 77.56 | 50.75 | 47.95 |
| faster_rcnn | ResNet50 | v1(oc) | l1 | 1x | DOTA-v1.0,train | 1024x1024,512 | \ | \ | 69.72 | 38.09 | 38.48 |

### HRSC2016

| Method | Backbone  | Angle     | Loss  | Lr schd | Dataset  | preprocess | RR   | $AP$(VOC07) | $AP$(VOC12) |
| ------ | --------- | --------- | ----- | ------- | -------- | ---------- | ---- | ----------- | ----------- |
| retina | ResNet101 | v2(le135) | R_IoU | 3x      | HRSC2016 | 1333x800   | Yes  | 89.85       | 94.95       |

### DIOR-R

| Method | Backbone  | Angle     | Loss  | Lr schd | Dataset | preprocess | RR   | $AP$ |
| ------ | --------- | --------- | ----- | ------- | ------- | ---------- | ---- | ---- |
| retina | ResNet101 | v2(le135) | R_IoU | 1x      | DIOR-R  | 800x800    | Yes  | 60.8 |