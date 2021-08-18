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

Supported methods（rotate）:

- [x] [Faster R-CNN](configs/faster_rcnn)
- [x] HSP

Some other methods are also supported in [projects using MMDetection](./docs/projects.md).

## Installation

create virtual environment

```
conda create -n openmmlab python=3.8 -y
conda activate openmmlab
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

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
