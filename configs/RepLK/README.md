# Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs

http://arxiv.org/abs/2203.06717

## abstract

In this paper we revisit large kernel design in modern convolutional neural networks (CNNs), which is often neglected in the past few years. Inspired by recent advances of vision transformers (ViTs), we point out that using a few large kernels instead of a stack of small convolutions could be a more powerful paradigm. We therefore summarize 5 guidelines, e.g., applying re-parameterized large depth-wise convolutions, to design efficient high-performance large-kernel CNNs. Following the guidelines, we propose RepLKNet, a pure CNN architecture whose kernel size is as large as 31x31. RepLKNet greatly bridges the performance gap between CNNs and ViTs, e.g., achieving comparable or better results than Swin Transformer on ImageNet and downstream tasks, while the latency of RepLKNet is much lower. Moreover, RepLKNet also shows feasible scalability to big data and large models, obtaining 87.8% top-1 accuracy on ImageNet and 56.0%} mIoU on ADE20K. At last, our study further suggests large-kernel CNNs share several nice properties with ViTs, e.g., much larger effective receptive fields than conventional CNNs, and higher shape bias rather than texture bias. Code & models at https://github.com/megvii-research/RepLKNet.

## Results and Models

| Method | Backbone       | Angle  | Loss  | Lr schd | Dataset         | preprocess    | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| ------ | -------------- | ------ | ----- | ------- | --------------- | ------------- | ---------- | ----------- | ----- |
| retina | ReplLKNet_base | v1(oc) | R_IoU | 1x      | DOTA-v1.0,train | 1024x1024,512 | 73.81      | 43.33       | 43.28 |

**note**: the speed of training and inference is very slow under nn.Conv2d or DepthWiseConv2dImplicitGEMM, when the batch size is small