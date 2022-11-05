# Probabilistic Anchor Assignment with IoU Prediction for Object Detection

[Probabilistic Anchor Assignment with IoU Prediction for Object Detection](https://link.springer.com/10.1007/978-3-030-58595-2_22)

## Abstract

In object detection, determining which anchors to assign as positive or negative samples, known as anchor assignment, has been revealed as a core procedure that can signiﬁcantly aﬀect a model’s performance. In this paper we propose a novel anchor assignment strategy that adaptively separates anchors into positive and negative samples for a ground truth bounding box according to the model’s learning status such that it is able to reason about the separation in a probabilistic manner. To do so we ﬁrst calculate the scores of anchors conditioned on the model and ﬁt a probability distribution to these scores. The model is then trained with anchors separated into positive and negative samples according to their probabilities. Moreover, we investigate the gap between the training and testing objectives and propose to predict the Intersection-overUnions of detected boxes as a measure of localization quality to reduce the discrepancy. The combined score of classiﬁcation and localization qualities serving as a box selection metric in non-maximum suppression well aligns with the proposed anchor assignment strategy and leads signiﬁcant performance improvements. The proposed methods only add a single convolutional layer to RetinaNet baseline and does not require multiple anchors per location, so are eﬃcient. Experimental results verify the eﬀectiveness of the proposed methods. Especially, our models set new records for single-stage detectors on MS COCO test-dev dataset with various backbones. Code is available at https://github.com/kkhoot/ PAA.

## Results and Models

| Method             | angle | Backbone | Lr schd | Dataset            | preprocess    | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| ------------------ | ----- | -------- | ------- | ------------------ | ------------- | ---------- | ----------- | ----- |
| RetinaNet-(maxiou) | v2    | ResNet50 | 1x      | DOTA-v1.0,trainval | 1024x1024,200 | 72.50      | 42.10       | 41.97 |
| RetinaNet-(ATSS)   | v2    | ResNet50 | 1x      | DOTA-v1.0,trainval | 1024x1024,200 | 72.75      | 43.46       | 42.46 |

