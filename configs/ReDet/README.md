# ReDet: A Rotation-equivariant Detector for Aerial Object Detection

[ReDet: A Rotation-equivariant Detector for Aerial Object Detection](http://arxiv.org/abs/2103.07733)

## Abstract

Recently, object detection in aerial images has gained much attention in computer vision. Different from objects in natural images, aerial objects are often distributed with arbitrary orientation. Therefore, the detector requires more parameters to encode the orientation information, which are often highly redundant and inefficient. Moreover, as ordinary CNNs do not explicitly model the orientation variation, large amounts of rotation augmented data is needed to train an accurate object detector. In this paper, we propose a Rotation-equivariant Detector (ReDet) to address these issues, which explicitly encodes rotation equivariance and rotation invariance. More precisely, we incorporate rotation-equivariant networks into the detector to extract rotation-equivariant features, which can accurately predict the orientation and lead to a huge reduction of model size. Based on the rotation-equivariant features, we also present Rotation-invariant RoI Align (RiRoI Align), which adaptively extracts rotation-invariant features from equivariant features according to the orientation of RoI. Extensive experiments on several challenging aerial image datasets DOTA-v1.0, DOTA-v1.5 and HRSC2016, show that our method can achieve state-of-the-art performance on the task of aerial object detection. Compared with previous best results, our ReDet gains 1.2, 3.5 and 2.6 mAP on DOTA-v1.0, DOTA-v1.5 and HRSC2016 respectively while reducing the number of parameters by 60\% (313 Mb vs. 121 Mb). The code is available at: \url{https://github.com/csuhan/ReDet}.

## Results and Models

| Method | angle  | Backbone   | Lr schd | Dataset            | preprocess    | loss     | MS   | RR   | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| ------ | ------ | ---------- | ------- | ------------------ | ------------- | -------- | ---- | ---- | ---------- | ----------- | ----- |
| ReDet  | v1(oc) | ReResNet50 | 1x      | DOTA-v1.0,train    | 1024x1024,512 | smoothl1 | \    | \    | 73.25      | 45.08       | 43.65 |
| ReDet  | v1(oc) | ReResNet50 | 1x      | DOTA-v1.0,trainval | 1024x1024,512 | smoothl1 | Y    | Y    | 80.23      | 56.29       | 51.49 |