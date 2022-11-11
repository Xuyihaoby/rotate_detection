# RepVGG: Making VGG-style ConvNets Great Again

[RepVGG: Making VGG-style ConvNets Great Again](https://ieeexplore.ieee.org/document/9577516/)

## abstract

We present a simple but powerful architecture of convolutional neural network, which has a VGG-like inferencetime body composed of nothing but a stack of 3 × 3 convolution and ReLU, while the training-time model has a multi-branch topology. Such decoupling of the trainingtime and inference-time architecture is realized by a structural re-parameterization technique so that the model is named RepVGG. On ImageNet, RepVGG reaches over 80% top-1 accuracy, which is the ﬁrst time for a plain model, to the best of our knowledge. On NVIDIA 1080Ti GPU, RepVGG models run 83% faster than ResNet-50 or 101% faster than ResNet-101 with higher accuracy and show favorable accuracy-speed trade-off compared to the stateof-the-art models like EfﬁcientNet and RegNet. The code and trained models are available at https://github.com/megvii-model/RepVGG.

## Results and Models

| Method     | Backbone    | Angle     | Loss  | Lr schd | Dataset         | preprocess    | optimizer | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| ---------- | ----------- | --------- | ----- | ------- | --------------- | ------------- | :-------: | ---------- | ----------- | ----- |
| gfl_retina | repvgg_b1g2 | v2(le135) | R_IoU | 1x      | DOTA-v1.0,train | 1024x1024,512 |    SGD    | 70.95      | 40.28       | 40.53 |
| retina     | repvgg_b1g2 | v1(oc)    | R_IoU | 1x      | DOTA-v1.0,train | 1024x1024,512 |    SGD    | 39.08      | 20.01       | 21.21 |

**note**:If directly regress five parameters and use R_IoU loss, the model can not converge well!!