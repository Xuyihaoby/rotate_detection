# Focal Modulation Networks

[Focal Modulation Networks](http://arxiv.org/abs/2203.11926)

## abstract

In this work, we propose focal modulation network (FocalNet in short), where self-attention (SA) is completely replaced by a focal modulation module that is more effective and efficient for modeling token interactions. Focal modulation comprises three components: $(i)$ hierarchical contextualization, implemented using a stack of depth-wise convolutional layers, to encode visual contexts from short to long ranges at different granularity levels, $(ii)$ gated aggregation to selectively aggregate context features for each visual token (query) based on its content, and $(iii)$ modulation or element-wise affine transformation to fuse the aggregated features into the query vector. Extensive experiments show that FocalNets outperform the state-of-the-art SA counterparts (e.g., Swin Transformers) with similar time and memory cost on the tasks of image classification, object detection, and semantic segmentation. Specifically, our FocalNets with tiny and base sizes achieve 82.3% and 83.9% top-1 accuracy on ImageNet-1K. After pretrained on ImageNet-22K, it attains 86.5% and 87.3% top-1 accuracy when finetuned with resolution 224x224 and 384x384, respectively. FocalNets exhibit remarkable superiority when transferred to downstream tasks. For object detection with Mask R-CNN, our FocalNet base trained with 1x already surpasses Swin trained with 3x schedule (49.0 v.s. 48.5). For semantic segmentation with UperNet, FocalNet base evaluated at single-scale outperforms Swin evaluated at multi-scale (50.5 v.s. 49.7). These results render focal modulation a favorable alternative to SA for effective and efficient visual modeling in real-world applications. Code is available at https://github.com/microsoft/FocalNet.

## Results and Models

| Method    | Backbone       | Lr schd | Dataset         | preprocess    | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| --------- | -------------- | ------- | --------------- | ------------- | ---------- | ----------- | ----- |
| RetinaNet | focal_tiny_srf | 1x      | DOTA-v1.0,train | 1024x1024,512 | 72.48      | 44.79       | 42.98 |

