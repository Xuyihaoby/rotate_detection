# EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications

[EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications](http://arxiv.org/abs/2206.10589)

## Abstract

In the pursuit of achieving ever-increasing accuracy, large and complex neural networks are usually developed. Such models demand high computational resources and therefore cannot be deployed on edge devices. It is of great interest to build resource-efficient general purpose networks due to their usefulness in several application areas. In this work, we strive to effectively combine the strengths of both CNN and Transformer models and propose a new efficient hybrid architecture EdgeNeXt. Specifically in EdgeNeXt, we introduce split depth-wise transpose attention (SDTA) encoder that splits input tensors into multiple channel groups and utilizes depth-wise convolution along with self-attention across channel dimensions to implicitly increase the receptive field and encode multi-scale features. Our extensive experiments on classification, detection and segmentation tasks, reveal the merits of the proposed approach, outperforming state-of-the-art methods with comparatively lower compute requirements. Our EdgeNeXt model with 1.3M parameters achieves 71.2\% top-1 accuracy on ImageNet-1K, outperforming MobileViT with an absolute gain of 2.2\% with 28\% reduction in FLOPs. Further, our EdgeNeXt model with 5.6M parameters achieves 79.4\% top-1 accuracy on ImageNet-1K. The code and models are publicly available at https://t.ly/_Vu9.

## Results and models

| Method     | Backbone       | Angle     | Loss  | Lr schd | Dataset         | preprocess    | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| ---------- | -------------- | --------- | ----- | ------- | --------------- | ------------- | ---------- | ----------- | ----- |
| gfl_retina | edgenect_small | v2(le135) | R_IoU | 1x      | DOTA-v1.0,train | 1024x1024,512 | 70.75      | 40.86       | 40.57 |