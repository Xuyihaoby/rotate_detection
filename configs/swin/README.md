# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

[Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](http://arxiv.org/abs/2103.14030)

## abstract

This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a general-purpose backbone for computer vision. Challenges in adapting Transformer from language to vision arise from differences between the two domains, such as large variations in the scale of visual entities and the high resolution of pixels in images compared to words in text. To address these differences, we propose a hierarchical Transformer whose representation is computed with shifted windows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection. This hierarchical architecture has the flexibility to model at various scales and has linear computational complexity with respect to image size. These qualities of Swin Transformer make it compatible with a broad range of vision tasks, including image classification (86.4 top-1 accuracy on ImageNet-1K) and dense prediction tasks such as object detection (58.7 box AP and 51.1 mask AP on COCO test-dev) and semantic segmentation (53.5 mIoU on ADE20K val). Its performance surpasses the previous state-of-the-art by a large margin of +2.7 box AP and +2.6 mask AP on COCO, and +3.2 mIoU on ADE20K, demonstrating the potential of Transformer-based models as vision backbones. The code and models will be made publicly available at~\url{https://github.com/microsoft/Swin-Transformer}.

---

# Swin Transformer V2: Scaling Up Capacity and Resolution

[Swin Transformer V2: Scaling Up Capacity and Resolution](http://arxiv.org/abs/2111.09883)

We present techniques for scaling Swin Transformer up to 3 billion parameters and making it capable of training with images of up to 1,536$\times$1,536 resolution. By scaling up capacity and resolution, Swin Transformer sets new records on four representative vision benchmarks: 84.0% top-1 accuracy on ImageNet-V2 image classification, 63.1/54.4 box/mask mAP on COCO object detection, 59.9 mIoU on ADE20K semantic segmentation, and 86.8% top-1 accuracy on Kinetics-400 video action classification. Our techniques are generally applicable for scaling up vision models, which has not been widely explored as that of NLP language models, partly due to the following difficulties in training and applications: 1) vision models often face instability issues at scale and 2) many downstream vision tasks require high resolution images or windows and it is not clear how to effectively transfer models pre-trained at low resolutions to higher resolution ones. The GPU memory consumption is also a problem when the image resolution is high. To address these issues, we present several techniques, which are illustrated by using Swin Transformer as a case study: 1) a post normalization technique and a scaled cosine attention approach to improve the stability of large vision models; 2) a log-spaced continuous position bias technique to effectively transfer models pre-trained at low-resolution images and windows to their higher-resolution counterparts. In addition, we share our crucial implementation details that lead to significant savings of GPU memory consumption and thus make it feasible to train large vision models with regular GPUs. Using these techniques and self-supervised pre-training, we successfully train a strong 3B Swin Transformer model and effectively transfer it to various vision tasks involving high-resolution images or windows, achieving the state-of-the-art accuracy on a variety of benchmarks.

---

# Rotary Position Embedding for Vision Transformer

[Rotary Position Embedding for Vision Transformer](http://arxiv.org/abs/2403.13298)

Rotary Position Embedding (RoPE) performs remarkably on language models, especially for length extrapolation of Transformers. However, the impacts of RoPE on computer vision domains have been underexplored, even though RoPE appears capable of enhancing Vision Transformer (ViT) performance in a way similar to the language domain. This study provides a comprehensive analysis of RoPE when applied to ViTs, utilizing practical implementations of RoPE for 2D vision data. The analysis reveals that RoPE demonstrates impressive extrapolation performance, i.e., maintaining precision while increasing image resolution at inference. It eventually leads to performance improvement for ImageNet-1k, COCO detection, and ADE-20k segmentation. We believe this study provides thorough guidelines to apply RoPE into ViT, promising improved backbone performance with minimal extra computational overhead. Our code and pre-trained models are available at https://github.com/naver-ai/rope-vit.

## Results and Models

|         Method          | Backbone         | Angle     | Loss      | Lr Sch. | Dataset         | preprocess    |  MS  | extra aug | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| :---------------------: | ---------------- | --------- | :-------- | ------- | :-------------- | ------------- | :--: | :-------: | ---------- | ----------- | ----- |
|       gfl_retina        | swin_tiny        | v2(le135) | R_IoU     | 1x      | DOTA-v1.0,train | 1024x1024,512 |  \   |     \     | 73.58      | 44.24       | 43.16 |
|       gfl_retina        | swin_tiny_v2     | v2(le135) | R_IoU     | 1x      | DOTA-v1.0,train | 1024x1024,512 |  \   |     \     | 73.01      | 43.49       | 43.06 |
|       gfl_retina        | swin_tiny_rope   | v2(le135) | R_IoU     | 1x      | DOTA-v1.0,train | 1024x1024,512 |  \   |     \     | 73.59      | 45.01       | 43.73 |
|         retina          | swin_tiny        | v2(le135) | R_IoU     | 1x      | DOTA-v1.0,train | 1024x1024,512 |  \   |     \     | 5.89       | 0.89        | 1.81  |
| rfaster_rcnn (hbb+obb)  | swin_tiny        | v1(oc)    | smooth_l1 | 1x      | DOTA-v1.0,train | 1024x1024,512 |  \   |     \     | 72.69      | 41.66       | 41.45 |
| rfaster_rcnn (hbb+obb)  | swin_tiny        | v1(oc)    | smooth_l1 | 1x      | DOTA-v1.0,train | 1024x1024,512 |  \   |    RR     | 75.38      | 42.48       | 42.61 |
| rfaster_rcnn (hbb+obb)  | swin_tiny        | v1(oc)    | smooth_l1 | 1x      | DOTA-v1.0,train | 1024x1024,512 | Yes  |    RR     | 79.58      | 51.88       | 48.34 |
| oriented_rcnn (hbb+obb) | swin_tiny(pafpn) | v1(oc)    | smooth_l1 | 1x      | DOTA-v1.0,train | 1024x1024,512 | Yes  |    RR     | 79.38      | \           | \     |

**note**:If directly regress five parameters and use R_IoU loss, the model can not converge well!!

