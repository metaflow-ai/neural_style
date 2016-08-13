FROM morgangiraud/tensorflow-cuda7.5-cudnn5

MAINTAINER Morgan Giraud <morgan@explee.com>

RUN apt-get update && apt-get install -y \
        libjpeg8-dev \
        vim \
        imagemagick \
        wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install \
        cycler \
        graphviz \
        h5py \
        Keras \
        matplotlib \
        numpy \
        Pillow \
        protobuf \
        pydot \
        pyparsing \
        pypng \
        python-dateutil \
        pytz \
        PyYAML \
        scandir \
        scipy \
        six
        
COPY keras.json /root/.keras/