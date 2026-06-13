from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="ojasx",
    ext_modules=[
        CppExtension(
            name="torchcl._C",
            sources=["torchcl/csrc/torchcl_extension.cpp"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
