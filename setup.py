from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(r"D:\Heng_shared\AOI-Project\tools\cocoapi-master\PythonAPI\pycocotools\_mask.pyx"),
    zip_safe=False,
)