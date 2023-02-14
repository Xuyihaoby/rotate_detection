# PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions

[PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions](http://arxiv.org/abs/2204.12511)

## Abstract

Cross-entropy loss and focal loss are the most common choices when training deep neural networks for classification problems. Generally speaking, however, a good loss function can take on much more flexible forms, and should be tailored for different tasks and datasets. Motivated by how functions can be approximated via Taylor expansion, we propose a simple framework, named PolyLoss, to view and design loss functions as a linear combination of polynomial functions. Our PolyLoss allows the importance of different polynomial bases to be easily adjusted depending on the targeting tasks and datasets, while naturally subsuming the aforementioned cross-entropy loss and focal loss as special cases. Extensive experimental results show that the optimal choice within the PolyLoss is indeed dependent on the task and dataset. Simply by introducing one extra hyperparameter and adding one line of code, our Poly-1 formulation outperforms the cross-entropy loss and focal loss on 2D image classification, instance segmentation, object detection, and 3D object detection tasks, sometimes by a large margin.

## Results and Models

| Method    | angle  | Backbone | Lr schd | Dataset            | preprocess    |  NA  |  BS  | loss | $\epsilon$ | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| --------- | ------ | -------- | ------- | ------------------ | ------------- | :--: | :--: | ---- | ---------- | ---------- | ----------- | ----- |
| RetinaNet | v1(oc) | ResNet50 | 1x      | DOTA-v1.0,trainval | 1024x1024,200 |  1   |  2   | RIoU | -1         | 68.71      | 38.01       | 38.75 |
| RetinaNet | v1(oc) | ResNet50 | 1x      | DOTA-v1.0,trainval | 1024x1024,200 |  1   |  2   | RIoU | 1          | 70.13      | 39.09       | 39.77 |