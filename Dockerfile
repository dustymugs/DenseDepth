#
# docker build -f Dockerfile -t dustymugs/densedepth .
#

FROM dustymugs/tensorflow/gpu

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y libglib2.0 libfontconfig.so.1 libxkbcommon-x11-0 libdbus-1-3 libxcursor1 meshlab && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN conda install -y scikit-image pydot && \
    pip install PyGLM PySide2 PyQt5 PyOpenGL click ipdb && \
    conda clean -yt

# purely for running GUI apps that require not being root
useradd -g users toor

WORKDIR /projects
CMD ["/bin/bash"]
