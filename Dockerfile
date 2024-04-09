# pytorch, cuda, cudnn
# py3.10
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel 

# for AOI-Project package
# ADD requirements.txt requirements.txt
# RUN pip install -r requirements.txt

# mmdetection
RUN pip install -U openmim  && \
    mim install mmengine && \
    mim install "mmcv>=2.0.0"

# pycocotools
RUN apt-get update && apt-get install -y git
RUN pip install cython
RUN pip install git+https://github.com/Zhong-Zi-Zeng/cocoapi.git#subdirectory=PythonAPI

# numpy
RUN pip install numpy==1.23.0

# Pillow
RUN pip install update Pillow==9.5

# faster-coco
RUN pip install git+https://github.com/Zhong-Zi-Zeng/faster_coco_eval.git

# OpenGL
# RUN apt-get update && apt-get install -y libgl1-mesa-glx
# RUN apt-get update && apt-get install -y libglib2.0-0

# opencv
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
# RUN pip uninstall opencv-python
RUN pip uninstall opencv-python-headless
RUN pip install "opencv-python-headless<4.3"

WORKDIR /WORKDIR