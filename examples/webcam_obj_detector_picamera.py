import os
import io
import time
import base64
import argparse
import logging
import socketserver
from http import server
from threading import Condition

import picamera
import cv2
from PIL import Image

import numpy as np
from edgetpu.detection.engine import DetectionEngine

# Parameters
AUTH_USERNAME = os.environ.get('AUTH_USERNAME', 'pi')
AUTH_PASSWORD = os.environ.get('AUTH_PASSWORD', 'picamera')
AUTH_BASE64 = base64.b64encode('{}:{}'.format(AUTH_USERNAME, AUTH_PASSWORD).encode('utf-8'))
BASIC_AUTH = 'Basic {}'.format(AUTH_BASE64.decode('utf-8'))
RESOLUTION = os.environ.get('RESOLUTION', '800x600').split('x')
RESOLUTION_X = int(RESOLUTION[0])
RESOLUTION_Y = int(RESOLUTION[1])
FRAMERATE = int(os.environ.get('FRAMERATE', '30'))
ROTATION = int(os.environ.get('ROTATE', 0))
HFLIP = os.environ.get('HFLIP', 'false').lower() == 'true'
VFLIP = os.environ.get('VFLIP', 'true').lower() == 'true'

PAGE = """\
<html>
<head>
<title>edgeTPU Object Detection</title>
</head>
<body>
<img src="stream.mjpg" width="{}" height="{}" />
</body>
</html>
""".format(RESOLUTION_X, RESOLUTION_Y)


class StreamingOutput(object):
    def __init__(self):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()
        self.engine = None

    def set_engine(self, engine):
        self.engine = engine

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame, copy the existing buffer's content and notify all
            # clients it's available
            self.buffer.truncate()
            with self.condition:
                self.frame = self.buffer.getvalue()
                self.condition.notify_all()
            self.buffer.seek(0)
        return self.buffer.write(buf)


class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.headers.get('Authorization') is None:
            self.do_AUTHHEAD()
            self.wfile.write(b'no auth header received')
        elif self.headers.get('Authorization') == BASIC_AUTH:
            self.authorized_get()
        else:
            self.do_AUTHHEAD()
            self.wfile.write(b'not authenticated')

    def do_AUTHHEAD(self):
        self.send_response(401)
        self.send_header('WWW-Authenticate', 'Basic realm=\"picamera\"')
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def append_objs_to_img(self, cv2_im, objs, labels):
        height, width, channels = cv2_im.shape
        for obj in objs:
            x0, y0, x1, y1 = obj.bounding_box.flatten().tolist()
            x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
            percent = int(100 * obj.score)
            label = '%d%% %s' % (percent, labels[obj.label_id])

            cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        return cv2_im

    def authorized_get(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                stream_video = io.BytesIO()
                _, width, height, channels = engine.get_input_tensor_shape()

                while True:
                    # getting image
                    camera.capture(stream_video, 
                                format='jpeg', 
                                use_video_port=True)
                    stream_video.truncate()
                    stream_video.seek(0)
                    
                    # cv2 / PIL coding
                    cv2_im = np.frombuffer(stream_video.getvalue(), dtype=np.uint8)
                    cv2_im = cv2.imdecode(cv2_im, 1)
                    pil_im = Image.fromarray(cv2_im)
                    
                    # object detection
                    start_ms = time.time()
                    objs = engine.detect_with_image(pil_im, threshold=args.threshold,
                                    keep_aspect_ratio=True, relative_coord=True,
                                    top_k=args.top_k)
                    elapsed_ms = time.time() - start_ms

                    cv2_im = self.append_objs_to_img(cv2_im, objs, labels)

                    r, buf = cv2.imencode(".jpg", cv2_im)

                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-type','image/jpeg')
                    self.send_header('Content-length',str(len(buf)))
                    self.end_headers()
                    self.wfile.write(bytearray(buf))
                    self.wfile.write(b'\r\n')
                    

            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='File path of Tflite model.', required=True)
    parser.add_argument('--label', help='File path of label file.', required=True)
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of classes with highest score to display')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='class score threshold')

    args = parser.parse_args()
    res = '{}x{}'.format(RESOLUTION_X, RESOLUTION_Y)

    with open(args.label, 'r') as f:
        pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
        labels = dict((int(k), v) for k, v in pairs)

    engine = DetectionEngine(args.model)

    with picamera.PiCamera(resolution=res, framerate=FRAMERATE, sensor_mode=2) as camera:
        camera.hflip = HFLIP
        camera.vflip = VFLIP
        camera.rotation = ROTATION

        try:
            address = ('', 8080)
            server = StreamingServer(address, StreamingHandler)
            server.serve_forever()
        except:
            print("error on the server!")