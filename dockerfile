FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

ENV TORCH_CUDA_ARCH_LIST="3.5 3.7 5.0 5.2 6.0 6.1 7.0 7.5+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install mmcv-full==1.2.5 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
RUN pip install opencv-contrib-python==4.2.0.32


# 拷贝文件
COPY . /work

#环境变量设置
# 设置编码方式
ENV LANG=C.UTF-8

ENV FORCE_CUDA="1"

WORKDIR /
RUN cd /work && pip install -r requirements/build.txt && pip install -v -e .

CMD ["python3", "-u", "/work/main.py"]
