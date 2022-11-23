## You Only Learn One Representation: Unified Network for Multiple Tasks

[You Only Learn One Representation: Unified Network for Multiple Tasks](http://arxiv.org/abs/2105.04206)

## Abstract

People ``understand'' the world via vision, hearing, tactile, and also the past experience. Human experience can be learned through normal learning (we call it explicit knowledge), or subconsciously (we call it implicit knowledge). These experiences learned through normal learning or subconsciously will be encoded and stored in the brain. Using these abundant experience as a huge database, human beings can effectively process data, even they were unseen beforehand. In this paper, we propose a unified network to encode implicit knowledge and explicit knowledge together, just like the human brain can learn knowledge from normal learning as well as subconsciousness learning. The unified network can generate a unified representation to simultaneously serve various tasks. We can perform kernel space alignment, prediction refinement, and multi-task learning in a convolutional neural network. The results demonstrate that when implicit knowledge is introduced into the neural network, it benefits the performance of all tasks. We further analyze the implicit representation learnt from the proposed unified network, and it shows great capability on catching the physical meaning of different tasks. The source code of this work is at : https://github.com/WongKinYiu/yolor.

## Results and Models

| Method    | Backbone | Lr schd | Dataset         | preprocess    | $AP_{0.5}$ | $AP_{0.75}$ | $mAP$ |
| --------- | -------- | ------- | --------------- | ------------- | ---------- | ----------- | ----- |
| RetinaNet | ResNet50 | 1x      | DOTA-v1.0,train | 1024x1024,512 | 67.53      | 38.65       | 38.77 |