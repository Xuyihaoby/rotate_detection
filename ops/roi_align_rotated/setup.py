from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


setup(
    name='rroialign',  # 模块名称
    ext_modules=[CUDAExtension('rroialign', sources=['src/ROIAlignRotated_cpu.cpp','ROIAlignRotated_cpu_cuda.cu'])],
    cmdclass={
        'build_ext': BuildExtension
    }
)