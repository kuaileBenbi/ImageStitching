from PIL import Image
import glob, os

"""
去黑边，上下左右各减5个像素
"""

def crop_image(input_path, output_path, pixels=5):
    with Image.open(input_path) as img:
        width, height = img.size
        left = pixels
        top = pixels
        right = width - pixels
        bottom = height - pixels
        cropped_img = img.crop((left, top, right, bottom))
        cropped_img.save(output_path)

img_path = 'datasets/5/*.jpg'
imgnames = glob.glob(img_path)
save_path = 'cut_datasets/5'

if not os.path.exists(save_path):
    os.makedirs(save_path)

for read_name in imgnames:
    _, save_name = os.path.split(read_name)
    crop_image(read_name, os.path.join(save_path, save_name), 5)
    print("cut done!")

