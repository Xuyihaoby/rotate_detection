# BOAT: Bilateral Local Attention Vision Transformer

[BOAT: Bilateral Local Attention Vision Transformer]()

## Abstract

Vision Transformers achieved outstanding performance in many computer vision tasks. Early Vision Transformers such as ViT and DeiT adopt global self-attention, which is computationally expensive when the number of patches is large. To improve efficiency, recent Vision Transformers adopt local self-attention mechanisms, where self-attention is computed within local windows. Despite the fact that window-based local self-attention significantly boosts efficiency, it fails to capture the relationships between distant but similar patches in the image plane. To overcome this limitation of image-space local attention, in this paper, we further exploit the locality of patches in the feature space. We group the patches into multiple clusters using their features, and self-attention is computed within every cluster. Such feature-space local attention effectively captures the connections between patches across different local windows but still relevant. We propose a Bilateral lOcal Attention vision Transformer (BOAT), which integrates feature-space local attention with image-space local attention. We further integrate BOAT with both Swin and CSWin models, and extensive experiments on several benchmark datasets demonstrate that our BOAT-CSWin model clearly and consistently outperforms existing state-of-the-art CNN models and vision Transformers.

## Results and Models

| Method     | Backbone       | Angle     | Loss  | Lr schd | Dataset         | bs   | preprocess    | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| ---------- | -------------- | --------- | ----- | ------- | --------------- | ---- | ------------- | ---------- | ----------- | ----- |
| gfl_retina | boat_swin_tiny | v2(le135) | R_IoU | 1x      | DOTA-v1.0,train | 2    | 1024x1024,512 | 73.49      | 44.45       | 43.94 |

**note**: the pretrained model is from [an uofficial repo](https://github.com/mahaoyuHKU/pytorch-boat). 