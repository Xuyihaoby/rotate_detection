## Introduction

MMDetection is an open source object detection toolbox based on PyTorch.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/model_zoo.md).

Supported backbones:

- [x] ResNet
- [x] ResNeXt
- [x] VGG
- [x] HRNet
- [x] RegNet
- [x] Res2Net
- [x] ResNeSt
- [x] swin
- [x] reresnet

Supported methods（rotate）:

- [x] RFaster R-CNN(hbb+obb)
- [x] HSP
- [x] RCascade RCNN(hbb proposal&hbb+obb)
- [x] RRetinanet(obb)
- [x] RHTC(hbb proposal&hbb+obb)
- [x] Oriented RCNN（obb/obb+hbb）
- [x] RoITrans（1st hbb+obb 2nd obb）
- [x] Double head（obb+hbb rf&oriented）
- [x] ReDet
- [x] s2anet
- [x] GWD
- [x] KLD
- [x] r3det
- [x] rreppoints(vanilla, OrientedReppoints)
- [x] RFCOS
- [x] RDETR(not converge)
- [ ] GRep(vanilla)
- [ ] centernet



Support different coder

$v1:cv2.minAreaRect$

$v2:-\pi/4\rightarrow\pi 3/4$

$v3:-\pi/2\rightarrow\pi/2$

Some other methods are also supported in [projects using MMDetection](./docs/projects.md).

## Installation

create virtual environment

```
conda create -n open-mmlab python=3.8 -y
conda activate open-mmlab
```

install pytorch

```
pip3 install torch==1.7.0+cu110{cuda_version} torchvision==0.8.1+cu110{cuda_version} torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```

manually install mmdetection/mmcv

```
pip install mmcv-full==1.2.5 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
```

```
cd rotate_detection
pip install -r requirements/build.txt
pip install -v -e . # or "python setup.py develop"
```

Install DOTA_devkit

```
cd DOTA_devkit
sudo apt-get install swig
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```
## Parse result
```
python tools/parse_results.py {configs} {pkl} {nms}
                              --[type]
                              --[eval]
```
* configs:the model config you design
* pkl:model inference result
* nms:whether to merge result [Y/N]
* type:if you want to merge, merge rotate or horizon [HBB/OBB/ALL]
* eval: whether to eval result

## Contributing

We appreciate all contributions to improve MMDetection. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMDetection is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new detectors.

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```
