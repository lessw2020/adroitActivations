from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='cuda_tensor_loader',
      ext_modules=[cpp_extension.CppExtension('cuda_tensor_loader', ['cuda_tensor_loader.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      install_requires=['torch>=1.0'])
