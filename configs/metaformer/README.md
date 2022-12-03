# MetaFormer is Actually What You Need for Vision

[MetaFormer is Actually What You Need for Vision](http://arxiv.org/abs/2111.11418)

## Abstract

Transformers have shown great potential in computer vision tasks. A common belief is their attention-based token mixer module contributes most to their competence. However, recent works show the attention-based module in transformers can be replaced by spatial MLPs and the resulted models still perform quite well. Based on this observation, we hypothesize that the general architecture of the transformers, instead of the specific token mixer module, is more essential to the model's performance. To verify this, we deliberately replace the attention module in transformers with an embarrassingly simple spatial pooling operator to conduct only the most basic token mixing. Surprisingly, we observe that the derived model, termed as PoolFormer, achieves competitive performance on multiple computer vision tasks. For example, on ImageNet-1K, PoolFormer achieves 82.1% top-1 accuracy, surpassing well-tuned vision transformer/MLP-like baselines DeiT-B/ResMLP-B24 by 0.3%/1.1% accuracy with 35%/52% fewer parameters and 48%/60% fewer MACs. The effectiveness of PoolFormer verifies our hypothesis and urges us to initiate the concept of "MetaFormer", a general architecture abstracted from transformers without specifying the token mixer. Based on the extensive experiments, we argue that MetaFormer is the key player in achieving superior results for recent transformer and MLP-like models on vision tasks. This work calls for more future research dedicated to improving MetaFormer instead of focusing on the token mixer modules. Additionally, our proposed PoolFormer could serve as a starting baseline for future MetaFormer architecture design. Code is available at https://github.com/sail-sg/poolformer

## Results and Models

| Method | Backbone | angle    | Lr schd | Dataset         | preprocess    | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| ------ | -------- | -------- | ------- | --------------- | ------------- | ---------- | ----------- | ----- |
| gfl    | pppp_s12 | v2(le90) | 1x      | DOTA-v1.0,train | 1024x1024,512 | 72.03      | 43.51       | 42.14 |
| gfl    | pppa_s12 | v2(le90) | 1x      | DOTA-v1.0,train | 1024x1024,512 | 70.30      | 41.51       | 40.95 |
| gfl    | id_s12   | v2(le90) | 1x      | DOTA-v1.0,train | 1024x1024,512 | 67.42      | 38.47       | 38.61 |