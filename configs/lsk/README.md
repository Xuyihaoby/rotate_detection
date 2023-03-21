# Large Selective Kernel Network for Remote Sensing Object Detection

[Large Selective Kernel Network for Remote Sensing Object Detection](http://arxiv.org/abs/2303.09030)

## Abstract 

Recent research on remote sensing object detection has largely focused on improving the representation of oriented bounding boxes but has overlooked the unique prior knowledge presented in remote sensing scenarios. Such prior knowledge can be useful because tiny remote sensing objects may be mistakenly detected without referencing a sufficiently long-range context, and the long-range context required by different types of objects can vary. In this paper, we take these priors into account and propose the Large Selective Kernel Network (LSKNet). LSKNet can dynamically adjust its large spatial receptive field to better model the ranging context of various objects in remote sensing scenarios. To the best of our knowledge, this is the first time that large and selective kernel mechanisms have been explored in the field of remote sensing object detection. Without bells and whistles, LSKNet sets new state-of-the-art scores on standard benchmarks, i.e., HRSC2016 (98.46\% mAP), DOTA-v1.0 (81.64\% mAP) and FAIR1M-v1.0 (47.87\% mAP). Based on a similar technique, we rank 2nd place in 2022 the Greater Bay Area International Algorithm Competition. Code is available at https://github.com/zcablii/Large-Selective-Kernel-Network.

## Results and Models

| Method     | Backbone | Angle     | Loss  | Lr schd | Dataset         | bs   | preprocess    | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| ---------- | -------- | --------- | ----- | ------- | --------------- | ---- | ------------- | ---------- | ----------- | ----- |
| gfl_retina | lsk_tiny | v2(le135) | R_IoU | 1x      | DOTA-v1.0,train | 2    | 1024x1024,512 | 71.79      | 41.31       | 41.45 |