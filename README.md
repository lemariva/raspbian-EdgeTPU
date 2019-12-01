# raspbian-EdgeTPU
This Docker image includes the libraries and packages to work with the Coral USB Accelerator (Google). The based image is `balenalib/raspberrypi3-debian:buster`. 

The image contains the following packages/libraries:
* Python 3.7.3
* NodeJS 12.13.1
* Python Packages:
    * numpy, matplotlib, pil, zmq
    * supervisor, tornado, picamera, python-periphery
    * jupyter, cython, jupyterlab, ipywebrtc, opencv-python
    * google-auth, oauthlib, imutils
* Other libraries included (check the `Dockerfile`)

A Jupyterlab is available (`https://<ip-address>:8888`) in which you can write code to process images obtained e.g. from the Pi camera. Moreover, two examples using the Coral USB Accelerator are included:
* `webcam_obj_detector.py`: detects and classifies objects on the fly processing the images taken with the Pi camera. The streaming images are available over http (`<ip address>:8080`).
* `teachable_jupyter.ipynb`: a notebook to re-train the tflite models to include your objects for image classification. 

To run the container type the following on a Raspberry Pi terminal:
```
docker run -d --privileged -p 25:22 -p 8080:8080 -p 8888:8888 -e PASSWORD=<<JUPYTER_PASSWORD>> --restart unless-stopped -v /dev/bus/usb:/dev/bus/usb  lemariva/raspbian-edgetpu
```
You need to activate the camera interface using `sudo raspi-config` to use live images of the Raspberry Pi camera.

## More Information
More information about the repository can be found on the following links:
* [#Edge-TPU: Coral USB Accelerator + rPI + Docker](https://lemariva.com/blog/2019/03/edge-tpu-coral-usb-accelerator-rpi-docker)
* [#Edge-TPU: Hands-On with Google's Coral USB accelerator](https://lemariva.com/blog/2019/04/edge-tpu-coral-usb-accelerator-dockerized)

## Licence
* Apache 2.0