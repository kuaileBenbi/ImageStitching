from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import pika
import threading
from osgeo import gdal, osr

app = Flask(__name__)
socketio = SocketIO(app)

previous_image = None
stitched_image_path = "static/stitched_image.jpg"


@app.route('/')
def index():
    return render_template('index.html')


def get_single_image_bounds(current_image):
    """ 
    Helper function to compute bounds for a single image.
    """
    # Convert the OpenCV image back to GeoTIFF byte data
    success, encoded_image = cv2.imencode(".tif", current_image)
    if not success:
        raise ValueError("Failed to encode the image back to GeoTIFF format.")
    
    mem_file = "/vsimem/in_memory"
    gdal.FileFromMemBuffer(mem_file, encoded_image.tobytes())

    # Open the virtual file using GDAL
    ds = gdal.Open(mem_file)

    geotransform = ds.GetGeoTransform()
    print("geotransform: ", geotransform)
    img_width = ds.RasterXSize
    img_height = ds.RasterYSize
    top_left_lon = geotransform[0]
    top_left_lat = geotransform[3]
    pixel_width_geo = geotransform[1]
    pixel_height_geo = geotransform[5]
    bottom_right_lon = top_left_lon + (img_width * pixel_width_geo)
    bottom_right_lat = top_left_lat + (img_height * pixel_height_geo)
    print("top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon: ", top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon)

    # Close and delete the virtual file to release memory
    ds = None
    gdal.Unlink(mem_file)

    return top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon

def compute_image_bounds(prev_image, current_image):
    """
    Compute the bounds for the stitched image using bounds of previous and current image.
    
    :param prev_image_path: Path to the previous stitched image.
    :param current_image_path: Path to the current image to be stitched.
    :return: Tuple of coordinates for the stitched image's bounds.
    """
    # Get bounds for each image
    prev_bounds = get_single_image_bounds(prev_image)
    current_bounds = get_single_image_bounds(current_image)

    # Compute bounds for stitched image
    top_left_lat = max(prev_bounds[0], current_bounds[0])
    top_left_lon = min(prev_bounds[1], current_bounds[1])
    bottom_right_lat = min(prev_bounds[2], current_bounds[2])
    bottom_right_lon = max(prev_bounds[3], current_bounds[3])

    return top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon


def image_callback(ch, method, properties, body):
    global previous_image

    nparr = np.frombuffer(body, np.uint8)
    current_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if previous_image is None:
        previous_image = current_image
        cv2.imwrite(stitched_image_path, current_image)
    else:
        print("stitching...")
        stitcher = cv2.Stitcher_create()
        ret, stitched_image = stitcher.stitch([previous_image, current_image])

        if ret == cv2.Stitcher_OK:
            print("stitched done!")
            cv2.imwrite(stitched_image_path, stitched_image)
            new_lat1, new_lon1, new_lat2, new_lon2 = compute_image_bounds(previous_image, current_image)
            print("new_lat1, new_lon1, new_lat2, new_lon2: ",new_lat1, new_lon1, new_lat2, new_lon2)
            socketio.emit('new_image', {'image_path': stitched_image_path, 'bounds': [
                          [new_lat1, new_lon1], [new_lat2, new_lon2]]})
        previous_image = current_image


def start_rabbitmq_consumer():
    connection = pika.BlockingConnection(
        pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='image_queue', durable=True)

    channel.basic_consume(queue='image_queue',
                          on_message_callback=image_callback, auto_ack=True)
    channel.start_consuming()


# Start RabbitMQ consumer in a different thread
rabbitmq_thread = threading.Thread(target=start_rabbitmq_consumer)
rabbitmq_thread.start()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=500)
