from setuptools import setup
# Pytorch's C++ tools
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

# Set OpenMP flags which compiler to use
os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"

setup(
    # Name of the python package
    name='arap_cpp',
    ext_modules=[
        CppExtension(
            # Name of the module to import in python, used as TORCH_EXTENSION_NAME
            name='arap_cpp',
            sources=['arap.cpp'],
            # Extra compiler flags for OpenMP
            extra_compile_args=[
                '-fopenmp', 
                '-O3',
                '-I/usr/include/eigen3'
            ],
            extra_link_args=['-fopenmp'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)