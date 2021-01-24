FROM balenalib/raspberrypi3-debian:buster
RUN [ "cross-build-start" ]

ARG MAYOR_REVISION
ARG MINOR_REVISION
ARG BUILD_ID

#labeling
LABEL org.label-schema.name="raspbian-edgetpu" \
    org.label-schema.description="Docker running Raspbian including Coral Edge-TPU libraries" \
    org.label-schema.url="https://lemariva.com" \
    org.label-schema.vcs-url="https://github.com/lemariva/raspbian-EdgeTPU" \
    org.label-schema.vendor="LeMaRiva|Tech info@lemariva.com" \
    org.label-schema.version="${MAYOR_REVISION}.${MINOR_REVISION}.${BUILD_ID}"

ENV READTHEDOCS True
ENV NODE_OPTIONS "--max-old-space-size=2048"
ENV CONFIG_PATH "/root/.jupyter/jupyter_notebook_config.py"

#do installation
RUN apt-get update \
	&& apt-get install -y --no-install-recommends openssh-server \
#do users
    && echo 'root:root' | chpasswd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd \
    && mkdir /var/run/sshd

#copy files
RUN mkdir /notebooks
COPY ./examples/ /notebooks/
COPY ./conf/jupyter_notebook_config.py ${CONFIG_PATH}
COPY ./conf/requirements.txt /tmp/requirements.txt

#nodejs for notebooks
RUN curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash - \
    && apt-get install -y nodejs

#install libraries for camera and opencv2
RUN apt-get install -y --no-install-recommends build-essential git wget feh pkg-config libjpeg-dev zlib1g-dev libssl-dev libffi-dev \
    libraspberrypi0 libraspberrypi-dev libraspberrypi-doc libraspberrypi-bin libfreetype6-dev libxml2 libopenjp2-7 \
    libatlas-base-dev libjasper-dev libqtgui4 libqt4-test libavformat-dev libswscale-dev  \
    python3-dev python3-pip python3-setuptools python3-wheel python3-pil python3-zmq python3-opencv \ 
    python3-matplotlib python3-numpy python3-scipy 

#python libraries
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --ignore-installed -r /tmp/requirements.txt

#jupyter packages
#RUN NODE_OPTIONS=--max_old_space_size=2048 jupyter nbextension enable --py widgetsnbextension \
#    && NODE_OPTIONS=--max_old_space_size=2048 jupyter labextension install --minimize=False @jupyter-widgets/jupyterlab-manager \
#    && NODE_OPTIONS=--max_old_space_size=2048 jupyter labextension install --minimize=False jupyter-webrtc

#install live camera libraries
RUN apt-get install libgstreamer1.0-0 gstreamer1.0-tools \ 
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good  \ 
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly v4l-utils

#downloading library file 
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list \
    && echo "deb https://packages.cloud.google.com/apt coral-cloud-stable main" | sudo tee /etc/apt/sources.list.d/coral-cloud.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add - \
    && apt-get update \
    && apt-get install -y python3-pycoral python3-tflite-runtime

#SSL for jupyter    
RUN mkdir /root/certs/ \
    && cd /root/certs/ \
    && openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout mykey.key -out mycert.pem -subj '/O=LeMaRiva|Tech/C=DE'

#loading pretrained models
WORKDIR /notebooks

RUN apt-get autoremove \
    && rm -rf /tmp/* \
    && rm -rf /var/lib/apt/lists/*

#copy supervisord files
COPY "./conf/supervisord.conf" /etc/supervisor/conf.d/supervisord.conf
RUN mkdir /var/log/supervisord/

#supervisord
CMD ["/usr/local/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

#opencv2 bug on arm (Undefined reference to __atomic)
ENV LD_PRELOAD "/usr/lib/arm-linux-gnueabihf/libatomic.so.1.2.0"

#SSH Port
EXPOSE 22 8888 8080

#set stop signal
STOPSIGNAL SIGTERM

#stop processing ARM emulation (comment out next line if built on Raspberry)
RUN [ "cross-build-end" ]
