from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name='polygon_geo_cpu',  # 模块名称
    ext_modules=[CppExtension('polygon_geo_cpu', sources=['src/polygon_geo_cpu.cpp'])],
    cmdclass={
        'build_ext': BuildExtension
    }
)