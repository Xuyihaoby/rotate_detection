from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


setup(
    name='rbbox_geo_cuda',  # 模块名称
    ext_modules=[CUDAExtension('rbbox_geo_cuda', sources=['src/rbbox_geo_cuda.cpp','src/rbbox_geo_kernel.cu'])],
    cmdclass={
        'build_ext': BuildExtension
    }
)