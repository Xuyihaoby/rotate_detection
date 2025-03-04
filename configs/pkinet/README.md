# Poly Kernel Inception Network for Remote Sensing Detection

[Poly Kernel Inception Network for Remote Sensing Detection](http://arxiv.org/abs/2403.06258)

## abstract

Object detection in remote sensing images (RSIs) often suffers from several increasing challenges, including the large variation in object scales and the diverse-ranging context. Prior methods tried to address these challenges by expanding the spatial receptive field of the backbone, either through large-kernel convolution or dilated convolution. However, the former typically introduces considerable background noise, while the latter risks generating overly sparse feature representations. In this paper, we introduce the Poly Kernel Inception Network (PKINet) to handle the above challenges. PKINet employs multi-scale convolution kernels without dilation to extract object features of varying scales and capture local context. In addition, a Context Anchor Attention (CAA) module is introduced in parallel to capture long-range contextual information. These two components work jointly to advance the performance of PKINet on four challenging remote sensing detection benchmarks, namely DOTA-v1.0, DOTA-v1.5, HRSC2016, and DIOR-R.

## Results and Models

|   Method   | Backbone | Angle     | Loss  | Lr Sch. | Dataset         | preprocess    |  MS  | extra aug | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| :--------: | -------- | --------- | :---- | ------- | :-------------- | ------------- | :--: | :-------: | ---------- | ----------- | ----- |
| gfl_retina | tiny     | v2(le135) | R_IoU | 1x      | DOTA-v1.0,train | 1024x1024,512 |  \   |     \     | 69.78      | 40.32       | 39.91 |



