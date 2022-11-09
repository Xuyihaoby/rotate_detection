# More ConvNets in the 2020s: Scaling up Kernels Beyond 51 × 51 using Sparsity

[More ConvNets in the 2020s: Scaling up Kernels Beyond 51 × 51 using Sparsity]((https://arxiv.org/abs/2207.03620))

## abstract

Transformers have quickly shined in the computer vision world since the emergence of Vision Transformers (ViTs). The dominant role of convolutional neural networks (CNNs) seems to be challenged by increasingly effective transformer-based models. Very recently, a couple of advanced convolutional models strike back with large kernels motivated by the local but large attention mechanism, showing appealing performance and efﬁciency. While one of them, i.e. RepLKNet, impressively manages to scale the kernel size to 31×31 with improved performance, the performance starts to saturate as the kernel size continues growing, compared to the scaling trend of advanced ViTs such as Swin Transformer. In this paper, we explore the possibility of training extreme convolutions larger than 31×31 and test whether the performance gap can be eliminated by strategically enlarging convolutions. This study ends up with a recipe for applying extremely large kernels from the perspective of sparsity, which can smoothly scale up kernels to 61×61 with better performance. Built on this recipe, we propose Sparse Large Kernel Network (SLaK), a pure CNN architecture equipped with 51×51 kernels that can perform on par with or better than state-of-the-art hierarchical Transformers and modern ConvNet architectures like ConvNeXt and RepLKNet, on ImageNet classiﬁcation as well as typical downstream tasks.

## Results and Models

| Method | Backbone  | Angle  | Loss  | Lr Sch. | Dataset         | preprocess    | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| :----: | --------- | ------ | :---- | ------- | :-------------- | ------------- | ---------- | ----------- | ----- |
| retina | slak_tiny | v1(oc) | R_IoU | 1x      | DOTA-v1.0,train | 1024x1024,512 | 71.07      | 42.25       | 41.45 |