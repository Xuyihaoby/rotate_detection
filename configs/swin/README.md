# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

[Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](http://arxiv.org/abs/2103.14030)

## abstract

This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a general-purpose backbone for computer vision. Challenges in adapting Transformer from language to vision arise from differences between the two domains, such as large variations in the scale of visual entities and the high resolution of pixels in images compared to words in text. To address these differences, we propose a hierarchical Transformer whose representation is computed with shifted windows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection. This hierarchical architecture has the flexibility to model at various scales and has linear computational complexity with respect to image size. These qualities of Swin Transformer make it compatible with a broad range of vision tasks, including image classification (86.4 top-1 accuracy on ImageNet-1K) and dense prediction tasks such as object detection (58.7 box AP and 51.1 mask AP on COCO test-dev) and semantic segmentation (53.5 mIoU on ADE20K val). Its performance surpasses the previous state-of-the-art by a large margin of +2.7 box AP and +2.6 mask AP on COCO, and +3.2 mIoU on ADE20K, demonstrating the potential of Transformer-based models as vision backbones. The code and models will be made publicly available at~\url{https://github.com/microsoft/Swin-Transformer}.

## Results and Models

|         Method         | Backbone  | Angle     | Loss      | Lr Sch. | Dataset         | preprocess    | extra aug | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| :--------------------: | --------- | --------- | :-------- | ------- | :-------------- | ------------- | :-------: | ---------- | ----------- | ----- |
|       gfl_retina       | swin_tiny | v2(le135) | R_IoU     | 1x      | DOTA-v1.0,train | 1024x1024,512 |     \     | 72.69      | 43.97       | 43.10 |
|         retina         | swin_tiny | v2(le135) | R_IoU     | 1x      | DOTA-v1.0,train | 1024x1024,512 |     \     | 5.89       | 0.89        | 1.81  |
| rfaster_rcnn (hbb+obb) | swin_tiny | v1(oc)    | smooth_l1 | 1x      | DOTA-v1.0,train | 1024x1024,512 |     \     | 72.69      | 41.66       | 41.45 |
| rfaster_rcnn (hbb+obb) | swin_tiny | v1(oc)    | smooth_l1 | 1x      | DOTA-v1.0,train | 1024x1024,512 |    RR     | 75.38      | 42.48       | 42.61 |

**note**:If directly regress five parameters and use R_IoU loss, the model can not converge well!!

