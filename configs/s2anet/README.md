# Align Deep Features for Oriented Object Detection

[http://arxiv.org/abs/2008.09397](http://arxiv.org/abs/2008.09397)

## Abstract

The past decade has witnessed significant progress on detecting objects in aerial images that are often distributed with large scale variations and arbitrary orientations. However most of existing methods rely on heuristically defined anchors with different scales, angles and aspect ratios and usually suffer from severe misalignment between anchor boxes and axis-aligned convolutional features, which leads to the common inconsistency between the classification score and localization accuracy. To address this issue, we propose a Single-shot Alignment Network (S$^2$A-Net) consisting of two modules: a Feature Alignment Module (FAM) and an Oriented Detection Module (ODM). The FAM can generate high-quality anchors with an Anchor Refinement Network and adaptively align the convolutional features according to the anchor boxes with a novel Alignment Convolution. The ODM first adopts active rotating filters to encode the orientation information and then produces orientation-sensitive and orientation-invariant features to alleviate the inconsistency between classification score and localization accuracy. Besides, we further explore the approach to detect objects in large-size images, which leads to a better trade-off between speed and accuracy. Extensive experiments demonstrate that our method can achieve state-of-the-art performance on two commonly used aerial objects datasets (i.e., DOTA and HRSC2016) while keeping high efficiency. The code is available at https://github.com/csuhan/s2anet.

## Results and Models

| Method | Backbone | Angle     | Loss     | Lr schd | Dataset            | preprocess    | MS   | Aug  | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| ------ | -------- | --------- | -------- | ------- | ------------------ | ------------- | ---- | ---- | ---------- | ----------- | ----- |
| S2ANet | ResNet50 | v2(le135) | smoothl1 | 1x      | DOTA-v1.0,train    | 1024x1024,512 | \    | \    | 71.47      | 37.36       | 39.62 |
| S2ANet | ResNet50 | v2(le135) | smoothl1 | 1x      | DOTA-v1.0,trainval | 1024x1024,824 | \    | \    | 73.83      | 38.17       | 40.56 |
| S2ANet | ResNet50 | v2(le135) | smoothl1 | 1x      | DOTA-v1.0,trainval | 1024x1024,824 | \    | RR   | \          | \           | \     |
| S2ANet | ResNet50 | v2(le135) | smoothl1 | 1x      | DOTA-v1.0,trainval | 1024x1024,512 | Yes  | RR   | \          | \           | \     |