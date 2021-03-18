#!/usr/bin/env python
import os
import subprocess
import time
from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)


def make_cuda_ext(name, module, sources, sources_cuda=[]):

    define_macros = []
    extra_compile_args = {'cxx': []}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension
        # raise EnvironmentError('CUDA is required to compile MMDetection!')

    return extension(
        name=f'{name}',
        sources=[p for p in sources],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)



if __name__ == '__main__':
    setup(
        name='rnms_ext',  # 模块名称
        ext_modules=[
            make_cuda_ext(
                name='rnms_ext',
                module='',
                sources=['src/rnms_ext.cpp', 'src/rcpu/rnms_cpu.cpp'],
                sources_cuda=[
                    'src/rcuda/rnms_cuda.cpp', 'src/rcuda/rnms_kernel.cu'
                ]),
        ],
        cmdclass={'build_ext': BuildExtension})
