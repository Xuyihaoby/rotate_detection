创建虚拟环境

```
conda create -n openmmlab python=3.8 -y
conda activate openmmlab
```

安装pytorch

```
pip3 install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```

手动安装mmdetection/mmcv

```
pip install mmcv-full==1.2.5 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
```

```
cd rotate_detection
pip install -r requirements/build.txt
pip install -v -e . # or "python setup.py develop"
```

