# SELF-SUPERVISED LEARNING WITH ROTATION- INVARIANT KERNELS

[SELF-SUPERVISED LEARNING WITH ROTATION- INVARIANT KERNELS](https://arxiv.org/abs/2208.00789)

## Abstract

We introduce a regularization loss based on kernel mean embeddings with rotation-invariant kernels on the hypersphere (also known as dot-product kernels) for self-supervised learning of image representations. Besides being fully competitive with the state of the art, our method signiﬁcantly reduces time and memory complexity for self-supervised training, making it implementable for very large embedding dimensions on existing devices and more easily adjustable than previous methods to settings with limited resources. Our work follows the major paradigm where the model learns to be invariant to some predeﬁned image transformations (cropping, blurring, color jittering, etc.), while avoiding a degenerate solution by regularizing the embedding distribution. Our particular contribution is to propose a loss family promoting the embedding distribution to be close to the uniform distribution on the hypersphere, with respect to the maximum mean discrepancy pseudometric. We demonstrate that this family encompasses several regularizers of former methods, including uniformity-based and information-maximization methods, which are variants of our ﬂexible regularization loss with different kernels. Beyond its practical consequences for state-ofthe-art self-supervised learning with limited resources, the proposed generic regularization approach opens perspectives to leverage more widely the literature on kernel methods in order to improve self-supervised learning methods.


## Results and Models

| Method | Backbone         | Angle     | Loss  | Lr schd | Dataset         | preprocess    | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| ------ | ---------------- | --------- | ----- | ------- | --------------- | ------------- | ---------- | ----------- | ----- |
| rgfl   | ResNet50(vicreg) | v2(le135) | R_IoU | 1x      | DOTA-v1.0,train | 1024x1024,512 | 70.31      | 38.66       | 39.27 |
