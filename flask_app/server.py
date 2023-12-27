import pika
import os

"""
server.py
起一个服务器把某个路径下的文件夹图像推到一个队列中
"""

def push_images_to_queue(folder_path, queue_name="image_queue"):
    # Set up a connection and channel to RabbitMQ server
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()

    # Declare a queue (create if not exists)
    channel.queue_declare(queue=queue_name, durable=True)

    # Push each image from the folder to the queue
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".jpg") or filename.endswith(".tif"):
            with open(os.path.join(folder_path, filename), 'rb') as f:
                image_data = f.read()
                channel.basic_publish(exchange='', routing_key=queue_name, body=image_data)

    # Close the connection
    connection.close()

if __name__ == "__main__":
    folder_path = 'results/5'
    push_images_to_queue(folder_path)
