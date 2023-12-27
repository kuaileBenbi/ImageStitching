from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
from osgeo import gdal
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename
import traceback
from threading import Thread

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# 假定所有的GeoTIFF图像都位于这个文件夹中
IMAGE_FOLDER = 'results/5'
STITCHED_FOLDER = 'static/stitched_images'

@app.route('/get_stitched_image/<filename>')
def get_stitched_image(filename):
    return send_from_directory(STITCHED_FOLDER, filename)

def geotiff2rgb(arr):
    if arr.dtype != np.uint8:
        # 假设arr的值域是0到最大值
        arr_normalized = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
        arr_uint8 = np.uint8(arr_normalized)
    else:
        arr_uint8 = arr
    # 然后，将单波段数组复制三次以创建一个三通道数组
    arr_3_channel = np.stack((arr_uint8,) * 3, axis=-1)
    return arr_3_channel

def stitch_images_task(image_folder, callback):
    try:
        # 执行长时间运行的拼接任务
        stitched_image_path, bounds = stitch_images(image_folder)
        if stitched_image_path:
            # 拼接成功，执行回调函数
            callback('success', stitched_image_path, bounds)
        else:
            # 拼接失败，执行回调函数
            callback('error', '拼接图像失败', None)
    except Exception as e:
        # 出现异常，执行回调函数
        callback('error', str(e), None)

def start_stitch_images_thread(image_folder):
    # 创建一个线程来运行长时间运行的拼接任务
    thread = Thread(target=stitch_images_task, args=(image_folder, handle_stitch_result))
    thread.start()

def handle_stitch_result(status, image_path, bounds):
    if status == 'success':
        # 获取安全的文件名并发送到前端
        # image_name = secure_filename(os.path.basename(image_path))
        socketio.emit('stitched_image_data', {'image_name': image_path, 'bounds': bounds})
    else:
        # 发送错误消息到前端
        socketio.emit('error', {'message': image_path})

def stitch_images(image_folder):
    try:
        # 找到所有的.tif文件
        tif_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.tif')]

        # 存储图像和对应的变换矩阵
        images = []
        transforms = []

        for tif in tif_files:
            ds = gdal.Open(tif)
            transform = ds.GetGeoTransform()
            print("transform: ", transform)
            band = ds.GetRasterBand(1)
            arr = band.ReadAsArray()
            images.append(geotiff2rgb(arr))
            transforms.append(transform)
        
        # 使用OpenCV进行拼接（简化示例）
        print("i am stitching...wait for a few minutes...")
        stitcher = cv2.createStitcher() if cv2.__version__.startswith('3.') else cv2.Stitcher_create()
        status, stitched = stitcher.stitch(images)
        print("i have finished the stitch job!")

        if status != cv2.Stitcher_OK:
            print('Can\'t stitch images, error code = %d' % status)
            return None, None
        
        # 保存拼接后的图像
        stitched_image_path = os.path.join(STITCHED_FOLDER, 'stitched_image.jpg')
        cv2.imwrite(stitched_image_path, stitched)

        # 假设所有图像在同一坐标系下，计算边界（这需要根据实际情况来确定）
        # 以下计算是非常简化的，可能并不适用于您的实际情况
        min_x = min(transform[0] for transform in transforms)
        max_y = min(transform[3] for transform in transforms)
        pixel_width = min(transform[1] for transform in transforms)
        pixel_height = -max(transform[5] for transform in transforms)
        max_x = min_x + pixel_width * stitched.shape[1]
        min_y = max_y - pixel_height * stitched.shape[0]
        bounds = [[min_y, min_x], [max_y, max_x]]
        print("stitched_image_path, bounds: ", stitched_image_path, bounds)
        return stitched_image_path, bounds
    except Exception as e:
        print(f'An error occurred in stitch_images: {e}')
        traceback.print_exc()  # 这将打印堆栈跟踪，有助于确定错误位置
        return None, None

@socketio.on('stitch_and_send')
def handle_stitch_request():
    print('Stitch and send request received.')
    # 开始后台线程来处理拼接任务
    start_stitch_images_thread(IMAGE_FOLDER)

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=500, debug=True)
