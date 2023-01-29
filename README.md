## Introduction

Rotate Detection is an open source rotated object detection toolbox based on PyTorch.

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
- [x] Swin
- [x] ReResNet
- [x] MobileNet_v2
- [x] RepVGG
- [x] ConvNeXt(v1\v2)
- [x] RepLK
- [x] SLaK
- [x] HorNet
- [x] FocalNet
- [x] PVT(v1\v2)
- [x] metaformer(v1\v2)
- [x] efficientformer

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
- [x] rreppoints(vanilla, OrientedReppoints, cfa)
- [x] RFCOS
- [x] RDETR(not converge)
- [x] GRep(vanilla)
- [x] KFIoU
- [x] RGFL(vanilla)
- [x] radamixer(vanilla)
- [x] rsparse_rcnn(vanilla)
- [x] rpaa(vanilla,atss)
- [ ] centernet

## Angle version

$v1(oc):cv2.minAreaRect$

$v2(le135):-\pi/4\rightarrow\pi 3/4$

$v3(le90):-\pi/2\rightarrow\pi/2$

You can find more details in [About angle definition](docs/angle/About_angle_definition.md)

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

## Reference

[open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)

[open-mmlab/mmrotate](https://github.com/open-mmlab/mmrotate)

[jbwang1997/OBBDetection](https://github.com/jbwang1997/OBBDetection)
