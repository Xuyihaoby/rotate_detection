# Image as Set of Points

## Abstract

What is an image and how to extract latent features? Convolutional Networks (ConvNets) consider an image as organized pixels in a rectangular shape and extract features via convolutional operation in local region; Vision Transformers (ViTs) treat an image as a sequence of patches and extract features via attention mechanism in a global range. In this work, we introduce a straightforward and promising paradigm for visual representation, which is called Context Clusters. Context clusters (CoCs) view an image as a set of unorganized points and extract features via simplified clustering algorithm. In detail, each point includes the raw feature (e.g., color) and positional information (e.g. coordinates), and a simplified clustering algorithm is employed to group and extract deep features hierarchically. Our CoCs are convolution- and attention-free, and only rely on clustering algorithm for spatial interaction. Owing to the simple design, we show CoCs endow gratifying interpretability via the visualization of clustering process. Our CoCs aim at providing a new perspective on image and visual representation, which may enjoy broad applications in different domains and exhibit profound insights. Even though we are not targeting SOTA performance, COCs still achieve comparable or even better performance than ConvNets or ViTs on several benchmarks. Codes are made available at: https://anonymous.4open.science/r/ContextCluster.

## Results and models

| Method     | Backbone                    | Angle     | Loss  | Lr schd | Dataset         | bs   | preprocess    | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| ---------- | --------------------------- | --------- | ----- | ------- | --------------- | ---- | ------------- | ---------- | ----------- | ----- |
| gfl_retina | context_cluster_small_feat2 | v2(le135) | R_IoU | 1x      | DOTA-v1.0,train | 2    | 1024x1024,512 | 65.16      | 38.59       | 38.08 |
| gfl_retina | context_cluster_small_feat5 | v2(le135) | R_IoU | 1x      | DOTA-v1.0,train | 2    | 1024x1024,512 | 70.69      | 43.12       | 41.81 |
| gfl_retina | context_cluster_small_feat7 | v2(le135) | R_IoU | 1x      | DOTA-v1.0,train | 1    | 1024x1024,512 | 68.42      | 40.51       | 39.55 |

