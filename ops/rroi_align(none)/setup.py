from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import torch
import os


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
        name=f'roi_align_rotated_cuda',
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)


if __name__ == "__main__":
    extra_compile_args = {'cxx': []}
    extra_compile_args['nvcc'] = [
        '-D__CUDA_NO_HALF_OPERATORS__',
        '-D__CUDA_NO_HALF_CONVERSIONS__',
        '-D__CUDA_NO_HALF2_OPERATORS__',
    ]
    setup(
        name='roi_align_rotated_cuda',
        ext_modules=[
            CUDAExtension('roi_align_rotated_cuda',
                          sources=['src/roi_align_rotated_cuda.cpp',
                                   'src/roi_align_rotated_kernel.cu'],
                          define_macros=[('WITH_CUDA', None)],
                          extra_compile_args=extra_compile_args
                          )
        ],
        cmdclass={'build_ext': BuildExtension})
