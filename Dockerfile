from balenalib/raspberry-pi-debian

RUN [ "cross-build-start" ]


#labeling
LABEL maintainer="lemariva@gmail.com" \
      version="V0.1.2" \
      description="Docker raspbian & Coral USB accelerator"

#copy files
COPY "./lib/*" /root/

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
RUN apt-get install -y wget build-essential python3-dev python3-pip \
    && apt-get install libraspberrypi0 libraspberrypi-dev libraspberrypi-doc libraspberrypi-bin
#camera libraries
RUN python3 -m pip install setuptools \
    && python3 -m pip install wheel \
    && python3 -m pip install picamera numpy  

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
WORKDIR /root
RUN wget -P test_data/ https://storage.googleapis.com/cloud-iot-edge-pretrained-models/canned_models/mobilenet_v2_1.0_224_quant_edgetpu.tflite \
    && wget -P test_data/ http://storage.googleapis.com/cloud-iot-edge-pretrained-models/canned_models/imagenet_labels.txt

#clean up
#RUN apt-get -yqq autoremove \
#    && apt-get -y clean \
#    && rm -rf /var/lib/apt/lists/*

#copy supervisord files
COPY "./conf/supervisord.conf" /etc/supervisor/conf.d/supervisord.conf
RUN mkdir /var/log/supervisord/

#supervisord
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

#SSH Port
EXPOSE 22 8000

#set stop signal
STOPSIGNAL SIGTERM

#stop processing ARM emulation (comment out next line if built on Raspberry)
RUN [ "cross-build-end" ]