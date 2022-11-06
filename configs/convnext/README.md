# A ConvNet for the 2020s

[A ConvNet for the 2020s](http://arxiv.org/abs/2201.03545)

## abstract

The "Roaring 20s" of visual recognition began with the introduction of Vision Transformers (ViTs), which quickly superseded ConvNets as the state-of-the-art image classification model. A vanilla ViT, on the other hand, faces difficulties when applied to general computer vision tasks such as object detection and semantic segmentation. It is the hierarchical Transformers (e.g., Swin Transformers) that reintroduced several ConvNet priors, making Transformers practically viable as a generic vision backbone and demonstrating remarkable performance on a wide variety of vision tasks. However, the effectiveness of such hybrid approaches is still largely credited to the intrinsic superiority of Transformers, rather than the inherent inductive biases of convolutions. In this work, we reexamine the design spaces and test the limits of what a pure ConvNet can achieve. We gradually "modernize" a standard ResNet toward the design of a vision Transformer, and discover several key components that contribute to the performance difference along the way. The outcome of this exploration is a family of pure ConvNet models dubbed ConvNeXt. Constructed entirely from standard ConvNet modules, ConvNeXts compete favorably with Transformers in terms of accuracy and scalability, achieving 87.8% ImageNet top-1 accuracy and outperforming Swin Transformers on COCO detection and ADE20K segmentation, while maintaining the simplicity and efficiency of standard ConvNets.

## Results and Models

| Method     | Backbone      | Angle     | Loss  | Lr schd | Dataset         | preprocess    | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| ---------- | ------------- | --------- | ----- | ------- | --------------- | ------------- | ---------- | ----------- | ----- |
| gfl_retina | convnext_tiny | v2(le135) | R_IoU | 1x      | DOTA-v1.0,train | 1024x1024,512 | 75.03      | 46.58       | 45.04 |
| retina     | convnext_tiny | v2(le135) | R_IoU | 1x      | DOTA-v1.0,train | 1024x1024,512 | 5.07       | 0.34        | 1.31  |

**note**:If directly regress five parameters and use R_IoU loss, the model can not converge well!!

