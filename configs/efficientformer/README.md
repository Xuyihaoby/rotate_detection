# EfficientFormer: Vision Transformers at MobileNet Speed

[EfficientFormer: Vision Transformers at MobileNet Speed](http://arxiv.org/abs/2206.01191)

## Abstract

Vision Transformers (ViT) have shown rapid progress in computer vision tasks, achieving promising results on various benchmarks. However, due to the massive number of parameters and model design, \textit{e.g.}, attention mechanism, ViT-based models are generally times slower than lightweight convolutional networks. Therefore, the deployment of ViT for real-time applications is particularly challenging, especially on resource-constrained hardware such as mobile devices. Recent efforts try to reduce the computation complexity of ViT through network architecture search or hybrid design with MobileNet block, yet the inference speed is still unsatisfactory. This leads to an important question: can transformers run as fast as MobileNet while obtaining high performance? To answer this, we first revisit the network architecture and operators used in ViT-based models and identify inefficient designs. Then we introduce a dimension-consistent pure transformer (without MobileNet blocks) as a design paradigm. Finally, we perform latency-driven slimming to get a series of final models dubbed EfficientFormer. Extensive experiments show the superiority of EfficientFormer in performance and speed on mobile devices. Our fastest model, EfficientFormer-L1, achieves $79.2\%$ top-1 accuracy on ImageNet-1K with only $1.6$ ms inference latency on iPhone 12 (compiled with CoreML), which runs as fast as MobileNetV2$\times 1.4$ ($1.6$ ms, $74.7\%$ top-1), and our largest model, EfficientFormer-L7, obtains $83.3\%$ accuracy with only $7.0$ ms latency. Our work proves that properly designed transformers can reach extremely low latency on mobile devices while maintaining high performance.

## Results and Models

| Method     | Backbone             | Angle     | Loss  | Lr schd | Dataset         | preprocess    | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| ---------- | -------------------- | --------- | ----- | ------- | --------------- | ------------- | ---------- | ----------- | ----- |
| gfl_retina | efficientformerv1-l1 | v2(le135) | R_IoU | 1x      | DOTA-v1.0,train | 1024x1024,512 | 71.12      | 41.37       | 40.72 |

**note**：efficientformer is sensitive to the parameters of optimizer and v2 may not converge well in some downstream tasks 