# AdaMixer: A Fast-Converging Query-Based Object Detector

[AdaMixer: A Fast-Converging Query-Based Object Detector](http://arxiv.org/abs/2203.16507)

## Abstract

Traditional object detectors employ the dense paradigm of scanning over locations and scales in an image. The recent query-based object detectors break this convention by decoding image features with a set of learnable queries. However, this paradigm still suffers from slow convergence, limited performance, and design complexity of extra networks between backbone and decoder. In this paper, we find that the key to these issues is the adaptability of decoders for casting queries to varying objects. Accordingly, we propose a fast-converging query-based detector, named AdaMixer, by improving the adaptability of query-based decoding processes in two aspects. First, each query adaptively samples features over space and scales based on estimated offsets, which allows AdaMixer to efficiently attend to the coherent regions of objects. Then, we dynamically decode these sampled features with an adaptive MLP-Mixer under the guidance of each query. Thanks to these two critical designs, AdaMixer enjoys architectural simplicity without requiring dense attentional encoders or explicit pyramid networks. On the challenging MS COCO benchmark, AdaMixer with ResNet-50 as the backbone, with 12 training epochs, reaches up to 45.0 AP on the validation set along with 27.9 APs in detecting small objects. With the longer training scheme, AdaMixer with ResNeXt-101-DCN and Swin-S reaches 49.5 and 51.3 AP. Our work sheds light on a simple, accurate, and fast converging architecture for query-based object detectors. The code is made available at https://github.com/MCG-NJU/AdaMixer

## Results and Models

| Method   | angle  | Backbone | Lr schd | Dataset            | preprocess    |  BS  | loss    | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| -------- | ------ | -------- | ------- | ------------------ | ------------- | :--: | ------- | ---------- | ----------- | ----- |
| Adamixer | v1(oc) | ResNet50 | 2x      | DOTA-v1.0,trainval | 1024x1024,200 |  2   | RIoU+L1 | 65.58      | 38.94       | 37.62 |

