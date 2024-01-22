import pika
import time
import cv2
import numpy as np

"""
client.py
客户端 - 以一秒一帧的频率接收图像
"""

def receive_and_display_images(queue_name="image_queue"):
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()

    channel.queue_declare(queue=queue_name, durable=True)
    cv2.namedWindow("Received Image", cv2.WINDOW_NORMAL)

    def callback(ch, method, properties, body):
        # Convert the byte data to a numpy array
        nparr = np.frombuffer(body, np.uint8)

        # Decode the numpy array as image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Display the image
        cv2.imshow("Received Image", img)
        cv2.waitKey(1000)  # Display the image for 1 second

    channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)
    channel.start_consuming()

    # When done, destroy the OpenCV window
    cv2.destroyAllWindows()

if __name__ == "__main__":
    receive_and_display_images()
