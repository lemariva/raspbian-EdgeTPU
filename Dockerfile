from balenalib/raspberrypi3-debian

RUN [ "cross-build-start" ]

#labeling
LABEL mantainer="Muro Riva <lemariva@gmail.com>" \
    org.label-schema.build-date=$BUILD_DATE \
    org.label-schema.name="raspbian-edgetpu" \
    org.label-schema.description="Docker running Raspbian including Coral Edge-TPU libraries" \
    org.label-schema.url="https://lemariva.com" \
    org.label-schema.vcs-ref=$VCS_REF \
    org.label-schema.vcs-url="https://github.com/lemariva/raspbian-EdgeTPU" \
    org.label-schema.vendor="Mauro Riva" \
    org.label-schema.version=$VERSION \
    org.label-schema.schema-version="1.0"

ENV READTHEDOCS True
ENV CONFIG_PATH="/root/.jupyter/jupyter_notebook_config.py"

#copy files
RUN mkdir /notebooks
COPY "./lib/*" /notebooks
COPY "./conf/jupyter_notebook_config.py" ${CONFIG_PATH}

#setting execute flags
#
#do installation
RUN apt-get update \
    && apt-get install -y git openssh-server supervisor \
#do users
    && echo 'root:root' | chpasswd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd \
    && mkdir /var/run/sshd 

#install libraries
RUN apt-get install -y wget build-essential python3-dev python3-pip feh pkg-config python-tk \
    && apt-get install libraspberrypi0 libraspberrypi-dev libraspberrypi-doc libraspberrypi-bin libfreetype6-dev libxml2

#python libraries
RUN python3 -m pip install setuptools wheel \
    && python3 -m pip install picamera numpy \
    && python3 -m pip install pillow jupyter matplotlib cython 

#installing edge-tpu library
WORKDIR /opt

#downloading library file 
RUN wget http://storage.googleapis.com/cloud-iot-edge-pretrained-models/edgetpu_api.tar.gz \
    && tar xzf edgetpu_api.tar.gz \
    && rm edgetpu_api.tar.gz
#trick platform recognizer 
COPY "./conf/platform_recognizer.sh" /opt/python-tflite-source/platform_recognizer.sh
#installing library
RUN cd python-tflite-source/ \
    && bash install.sh -y

#loading pretrained models
WORKDIR /notebooks
RUN wget -P test_data/ https://storage.googleapis.com/cloud-iot-edge-pretrained-models/canned_models/mobilenet_v2_1.0_224_quant_edgetpu.tflite \
    && wget -P test_data/ http://storage.googleapis.com/cloud-iot-edge-pretrained-models/canned_models/imagenet_labels.txt

#copy supervisord files
COPY "./conf/supervisord.conf" /etc/supervisor/conf.d/supervisord.conf
RUN mkdir /var/log/supervisord/

#supervisord
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

#SSH Port
EXPOSE 22 8888 8080

#set stop signal
STOPSIGNAL SIGTERM

#stop processing ARM emulation (comment out next line if built on Raspberry)
RUN [ "cross-build-end" ]