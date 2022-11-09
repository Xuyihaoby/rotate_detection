# HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions

[HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions](http://arxiv.org/abs/2207.14284)

## abstract

Recent progress in vision Transformers exhibits great success in various tasks driven by the new spatial modeling mechanism based on dot-product self-attention. In this paper, we show that the key ingredients behind the vision Transformers, namely input-adaptive, long-range and high-order spatial interactions, can also be efficiently implemented with a convolution-based framework. We present the Recursive Gated Convolution ($\textit{g}^\textit{n}$Conv) that performs high-order spatial interactions with gated convolutions and recursive designs. The new operation is highly flexible and customizable, which is compatible with various variants of convolution and extends the two-order interactions in self-attention to arbitrary orders without introducing significant extra computation. $\textit{g}^\textit{n}$Conv can serve as a plug-and-play module to improve various vision Transformers and convolution-based models. Based on the operation, we construct a new family of generic vision backbones named HorNet. Extensive experiments on ImageNet classification, COCO object detection and ADE20K semantic segmentation show HorNet outperform Swin Transformers and ConvNeXt by a significant margin with similar overall architecture and training configurations. HorNet also shows favorable scalability to more training data and a larger model size. Apart from the effectiveness in visual encoders, we also show $\textit{g}^\textit{n}$Conv can be applied to task-specific decoders and consistently improve dense prediction performance with less computation. Our results demonstrate that $\textit{g}^\textit{n}$Conv can be a new basic module for visual modeling that effectively combines the merits of both vision Transformers and CNNs. Code is available at https://github.com/raoyongming/HorNet

## Results and Models

| Method | Backbone            | Angle  | Loss  | Lr schd | Dataset         | preprocess    | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| ------ | ------------------- | ------ | ----- | ------- | --------------- | ------------- | ---------- | ----------- | ----- |
| retina | hornet_tiny         | v1(oc) | R_IoU | 1x      | DOTA-v1.0,train | 1024x1024,512 | 70.40      | 40.92       | 40.74 |
| retina | hornet_tiny_gnfpn   | v1(oc) | R_IoU | 1x      | DOTA-v1.0,train | 1024x1024,512 | 67.81      | 36.66       | 38.18 |
| retina | hornet_tiny_with_gf | v1(oc) | R_IoU | 1x      | DOTA-v1.0,train | 1024x1024,512 | 70.02      | 39.24       | 40.03 |

