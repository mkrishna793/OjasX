from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='scratch_ext',
    ext_modules=[
        CppExtension('scratch_ext', ['scratch_ext.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
