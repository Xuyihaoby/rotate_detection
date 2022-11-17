# Oriented RepPoints for Aerial Object Detection

[Oriented RepPoints for Aerial Object Detection](http://arxiv.org/abs/2105.11111)

## Abstract

In contrast to the oriented bounding boxes, point set representation has great potential to capture the detailed structure of instances with the arbitrary orientations, large aspect ratios and dense distribution in aerial images. However, the conventional point set-based approaches are handcrafted with the fixed locations using points-to-points supervision, which hurts their flexibility on the fine-grained feature extraction. To address these limitations, in this paper, we propose a novel approach to aerial object detection, named Oriented RepPoints. Specifically, we suggest to employ a set of adaptive points to capture the geometric and spatial information of the arbitrary-oriented objects, which is able to automatically arrange themselves over the object in a spatial and semantic scenario. To facilitate the supervised learning, the oriented conversion function is proposed to explicitly map the adaptive point set into an oriented bounding box. Moreover, we introduce an effective quality assessment measure to select the point set samples for training, which can choose the representative items with respect to their potentials on orientated object detection. Furthermore, we suggest a spatial constraint to penalize the outlier points outside the ground-truth bounding box. In addition to the traditional evaluation metric mAP focusing on overlap ratio, we propose a new metric mAOE to measure the orientation accuracy that is usually neglected in the previous studies on oriented object detection. Experiments on three widely used datasets including DOTA, HRSC2016 and UCAS-AOD demonstrate that our proposed approach is effective.

## Results and Models

| Method      | Backbone | Angle  | Loss      | Lr schd | Dataset         | preprocess    | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| ----------- | -------- | ------ | --------- | ------- | --------------- | ------------- | ---------- | ----------- | ----- |
| orientedrep | ResNet50 | v1(oc) | ConvexIoU | 1x      | DOTA-v1.0,train | 1024x1024,512 | 66.53      | 37.03       | 37.02 |

**note**：this version is the original version of  oriented reppoints and it is not the paper version.