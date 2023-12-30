# Advancing Plain Vision Transformer Towards Remote Sensing Foundation Model

[Advancing Plain Vision Transformer Towards Remote Sensing Foundation Model](http://arxiv.org/abs/2208.03987)

## Abstract

Deep learning has largely reshaped remote sensing research for aerial image understanding. Nevertheless, most of existing deep models are initialized with ImageNet pretrained weights, where the natural images inevitably presents a large domain gap relative to the aerial images, probably limiting the finetuning performance on downstream aerial scene tasks. This issue motivates us to conduct an empirical study of remote sensing pretraining (RSP). To this end, we train different networks from scratch with the help of the largest remote sensing scene recognition dataset up to now-MillionAID, to obtain the remote sensing pretrained backbones, including both convolutional neural networks (CNN) and vision transformers such as Swin and ViTAE, which have shown promising performance on computer vision tasks. Then, we investigate the impact of ImageNet pretraining (IMP) and RSP on a series of downstream tasks including scene recognition, semantic segmentation, object detection, and change detection using the CNN and vision transformers backbones. We have some empirical findings as follows. First, vision transformers generally outperforms CNN backbones, where ViTAE achieves the best performance, owing to its strong representation capacity by introducing intrinsic inductive bias from convolutions to transformers. Second, both IMP and RSP help deliver better performance, where IMP enjoys a versatility by learning more universal representations from diverse images belonging to much more categories while RSP is distinctive in perceiving remote sensing related semantics. Third, RSP mitigates the data discrepancy of IMP for remote sensing but may still suffer from the task discrepancy, where downstream tasks require different representations from the scene recognition task. These findings call for further research efforts on both large-scale pretraining datasets and effective pretraining methods.

## Results and Models

| Method | Backbone                  | Angle     | Loss  | Lr schd | Dataset         | preprocess    | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| ------ | ------------------------- | --------- | ----- | ------- | --------------- | ------------- | ---------- | ----------- | ----- |
| rgfl   | ViTAE_NC_Win_RVSA_V3_WSZ7 | v2(le135) | R_IoU | 1x      | DOTA-v1.0,train | 1024x1024,512 | 60.42      | 32.80       | 33.74 |

