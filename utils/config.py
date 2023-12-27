from collections import namedtuple

CameraParams = namedtuple('Camera', ['coords', 'yaw', 'pitch'])
ImageParams = namedtuple('ImageParams', ['horizontal_angle', 'vertical_angle', 'size'])