# Learning RoI Transformer for Oriented Object Detection in Aerial Images

[Learning RoI Transformer for Oriented Object Detection in Aerial Images](https://ieeexplore.ieee.org/document/8953881/)

# Abstract

Object detection in aerial images is an active yet challenging task in computer vision because of the bird’s-eye view perspective, the highly complex backgrounds, and the variant appearances of objects. Especially when detecting densely packed objects in aerial images, methods relying on horizontal proposals for common object detection often introduce mismatches between the Region of Interests (RoIs) and objects. This leads to the common misalignment between the ﬁnal object classiﬁcation conﬁdence and localization accuracy. In this paper, we propose a RoI Transformer to address these problems. The core idea of RoI Transformer is to apply spatial transformations on RoIs and learn the transformation parameters under the supervision of oriented bounding box (OBB) annotations. RoI Transformer is with lightweight and can be easily embedded into detectors for oriented object detection. Simply apply the RoI Transformer to light-head RCNN has achieved state-of-the-art performances on two common and challenging aerial datasets, i.e., DOTA and HRSC2016, with a neglectable reduction to detection speed. Our RoI Transformer exceeds the deformable Position Sensitive RoI pooling when oriented bounding-box annotations are available. Extensive experiments have also validated the ﬂexibility and effectiveness of our RoI Transformer.

## Results and Models

| Method          | Backbone | Angle  | Loss     | Lr schd | Dataset         | preprocess    | MS   | RR   | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| --------------- | -------- | ------ | -------- | ------- | --------------- | ------------- | ---- | ---- | ---------- | ----------- | ----- |
| RoI-transformer | ResNet50 | v1(oc) | smoothl1 | 1x      | DOTA-v1.0,train | 1024x1024,512 | \    | \    | 71.87      | 46.24       | 43.73 |
| RoI-transformer | ResNet50 | v1(oc) | smoothl1 | 1x      | DOTA-v1.0,train | 1024x1024,512 | Y    | Y    | 80.51      | 57.73       | 52.52 |

