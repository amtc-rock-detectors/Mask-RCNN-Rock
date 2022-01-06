FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04 
LABEL description="Conda 3 / nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04" \
      maintainer="https://github.com/rlan/docker" 
ARG CONDA_INSTALLER="https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh"
# Build-time metadata as defined at http://label-schema.org
ARG BUILD_DATE 
ARG VCS_REF 
ARG VERSION 
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/rlan/docker" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0" 

RUN apt-get -qq update \
      && apt-get -qq install -y --no-install-recommends \
        bzip2 \
        wget \
      && apt-get -qq clean \
    && wget $CONDA_INSTALLER -O /tmp/miniconda.sh --quiet --no-check-certificate \
      && chmod +x /tmp/miniconda.sh \
      && /tmp/miniconda.sh -b -p /opt/conda \
      && /opt/conda/bin/conda clean -ya --quiet \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* 
ENV PATH /opt/conda/bin:$PATH 
RUN conda install --quiet \
      keras-gpu==2.1.5 \
      tensorflow-gpu=1.14.0 \
    && conda clean -aqy
# TensorBoard
EXPOSE 6006
# Science libraries and other common packages
RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6 gcc -y 
RUN pip --no-cache-dir install \ 
numpy scipy sklearn scikit-image pandas matplotlib Cython requests pycocotools imgaug
# Opencv
RUN pip --no-cache-dir install opencv-python
