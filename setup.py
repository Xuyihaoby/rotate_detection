#!/usr/bin/env python
import os
from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


version_file = 'mmdet/version.py'


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


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

    return extension(
        name=f'{module}.{name}',
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)


def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.
    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs
    Returns:
        List[str]: list of requirements items
    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import sys
    from os.path import exists
    import re
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            elif '@git+' in line:
                info['package'] = line
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


if __name__ == '__main__':
    setup(
        name='mmdet',
        version=get_version(),
        description='OpenMMLab Detection Toolbox and Benchmark',
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='OpenMMLab',
        author_email='openmmlab@gmail.com',
        keywords='computer vision, object detection',
        url='https://github.com/open-mmlab/mmdetection',
        packages=find_packages(exclude=('configs', 'tools', 'demo')),
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        license='Apache License 2.0',
        setup_requires=parse_requirements('requirements/build.txt'),
        tests_require=parse_requirements('requirements/tests.txt'),
        install_requires=parse_requirements('requirements/runtime.txt'),
        extras_require={
            'all': parse_requirements('requirements.txt'),
            'tests': parse_requirements('requirements/tests.txt'),
            'build': parse_requirements('requirements/build.txt'),
            'optional': parse_requirements('requirements/optional.txt'),
        },
        ext_modules=[
            make_cuda_ext(
                name='rnms_ext',
                module='mmdet.ops.nms',
                sources=['src/rnms_ext.cpp', 'src/rcpu/rnms_cpu.cpp'],
                sources_cuda=[
                    'src/rcuda/rnms_cuda.cpp', 'src/rcuda/rnms_kernel.cu'
                ]),
            make_cuda_ext(
                name='rbbox_geo_cuda',
                module='mmdet.ops.rbbox_geo',
                sources=[],
                sources_cuda=[
                    'src/rbbox_geo_cuda.cpp', 'src/rbbox_geo_kernel.cu'
                ]),
            make_cuda_ext(
                name='polygon_geo_cpu',
                module='mmdet.ops.polygon_geo',
                sources=['src/polygon_geo_cpu.cpp']),
            make_cuda_ext(
                name='roi_align_rotated_cuda',
                module='mmdet.ops.roi_align_rotated',
                sources=['src/ROIAlignRotated_cpu.cpp', 'src/ROIAlignRotated_cuda.cu']),
            make_cuda_ext(
                name='orn_cuda',
                module='mmdet.ops.orn',
                sources=['src/vision.cpp',
                         'src/cpu/ActiveRotatingFilter_cpu.cpp', 'src/cpu/RotationInvariantEncoding_cpu.cpp',
                         'src/cuda/ActiveRotatingFilter_cuda.cu', 'src/cuda/RotationInvariantEncoding_cuda.cu',
                         ]),
            make_cuda_ext(
                name='riroi_align_cuda',
                module='mmdet.ops.riroi_align',
                sources=[],
                sources_cuda=['src/riroi_align_cuda.cpp',
                              'src/riroi_align_kernel.cu',
                              ]),
            make_cuda_ext(
                name='feature_refine_cuda',
                module='mmdet.ops.fr',
                sources=[],
                sources_cuda=[
                    'src/feature_refine_cuda.cpp', 'src/feature_refine_kernel.cu'
                ]),
            make_cuda_ext(
                name='box_iou_rotated_ext',
                module='mmdet.ops.box_iou_rotated',
                sources=[
                    'src/box_iou_rotated_cpu.cpp',
                    'src/box_iou_rotated_ext.cpp'
                ],
                sources_cuda=['src/box_iou_rotated_cuda.cu']),
            make_cuda_ext(
                name='ml_nms_rotated_cuda',
                module='mmdet.ops.ml_nms_rotated',
                sources=['src/nms_rotated_cpu.cpp'],
                sources_cuda=['src/nms_rotated_cuda.cu']),
            make_cuda_ext(
                name='nms_rotated_ext',
                module='mmdet.ops.nms_rotated',
                sources=['src/nms_rotated_cpu.cpp', 'src/nms_rotated_ext.cpp'],
                sources_cuda=[
                    'src/nms_rotated_cuda.cu',
                    'src/poly_nms_cuda.cu',
                ]),
            make_cuda_ext(
                name='convex_ext',
                module='mmdet.ops.convex',
                sources=['src/convex_cpu.cpp', 'src/convex_ext.cpp'],
                sources_cuda=['src/convex_cuda.cu']),
            make_cuda_ext(
                name='sort_vertices_cuda',
                module='mmdet.ops.box_iou_rotated_diff',
                sources=['src/sort_vert.cpp', 'src/sort_vert_kernel.cu', ]),
            make_cuda_ext(
                name='convex_giou_cuda',
                module='mmdet.ops.iou',
                sources=['src/convex_giou_cuda.cpp', 'src/convex_giou_kernel.cu']),
            make_cuda_ext(
                name='convex_iou_cuda',
                module='mmdet.ops.iou',
                sources=['src/convex_iou_cuda.cpp', 'src/convex_iou_kernel.cu'])
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
