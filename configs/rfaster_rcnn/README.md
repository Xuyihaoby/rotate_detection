# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/abs/1506.01497)

## Abstract

State-of-the-art object detection networks depend on region proposal algorithms to hypothesize object locations. Advances like SPPnet and Fast R-CNN have reduced the running time of these detection networks, exposing region proposal computation as a bottleneck. In this work, we introduce a Region Proposal Network (RPN) that shares full-image convolutional features with the detection network, thus enabling nearly cost-free region proposals. An RPN is a fully convolutional network that simultaneously predicts object bounds and objectness scores at each position. The RPN is trained end-to-end to generate high-quality region proposals, which are used by Fast R-CNN for detection. We further merge RPN and Fast R-CNN into a single network by sharing their convolutional features---using the recently popular terminology of neural networks with 'attention' mechanisms, the RPN component tells the unified network where to look. For the very deep VGG-16 model, our detection system has a frame rate of 5fps (including all steps) on a GPU, while achieving state-of-the-art object detection accuracy on PASCAL VOC 2007, 2012, and MS COCO datasets with only 300 proposals per image. In ILSVRC and COCO 2015 competitions, Faster R-CNN and RPN are the foundations of the 1st-place winning entries in several tracks. Code has been made publicly available.

## Results and Models

| Method       |         Aug          | Backbone | Angle  | Loss      | Lr schd | Dataset         | preprocess    |  MS  | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| ------------ | :------------------: | -------- | ------ | --------- | :-----: | --------------- | ------------- | :--: | ---------- | :---------: | :---: |
| rfaster_rcnn |          \           | ResNet50 | v1(oc) | smooth_l1 |   1x    | DOTA-v1.0,train | 1024x1024,512 |  \   | 70.48      |      \      |   \   |
| rfaster_rcnn |          \           | ResNet50 | v1(oc) | smooth_l1 |   1x    | DOTA-v1.0,train | 1024x1024,512 |  Y   | 76.8       |      \      |   \   |
| rfaster_rcnn |         ohem         | ResNet50 | v1(oc) | smooth_l1 |   1x    | DOTA-v1.0,train | 1024x1024,512 |  \   | 68.43      |      \      |   \   |
| rfaster_rcnn |         ema          | ResNet50 | v1(oc) | smooth_l1 |   1x    | DOTA-v1.0,train | 1024x1024,512 |  \   | 70.87      |    38.72    | 39.21 |
| rfaster_rcnn |          RR          | ResNet50 | v1(oc) | smooth_l1 |   1x    | DOTA-v1.0,train | 1024x1024,512 |  \   | 72.5       |      \      |   \   |
| rfaster_rcnn |          RR          | ResNet50 | v1(oc) | smooth_l1 |   1x    | DOTA-v1.0,train | 1024x1024,512 |  Y   | 79.27      |    49.55    | 47.56 |
| rfaster_rcnn |      grid_mask       | ResNet50 | v1(oc) | smooth_l1 |   1x    | DOTA-v1.0,train | 1024x1024,512 |  \   | 71.09      |      \      |   \   |
| rfaster_rcnn |    hsv-mosaic-RR     | ResNet50 | v1(oc) | smooth_l1 |   1x    | DOTA-v1.0,train | 1024x1024,512 |  \   | 73.10      |      \      |   \   |
| rfaster_rcnn |    hsv- mixup-RR     | ResNet50 | v1(oc) | smooth_l1 |   1x    | DOTA-v1.0,train | 1024x1024,512 |  \   | 72.5       |      \      |   \   |
| rfaster_rcnn |   hsv-mosaic-mixup   | ResNet50 | v1(oc) | smooth_l1 |   1x    | DOTA-v1.0,train | 1024x1024,512 |  \   | 74.21      |      \      |   \   |
| rfaster_rcnn |   hsv-mosaic-mixup   | ResNet50 | v1(oc) | smooth_l1 |   1x    | DOTA-v1.0,train | 1024x1024,512 |  Y   | 79.01      |      \      |   \   |
| rfaster_rcnn | hsv-mosaic-mixup_swa | ResNet50 | v1(oc) | smooth_l1 |   1x    | DOTA-v1.0,train | 1024x1024,512 |  \   | 74.93      |      \      |   \   |