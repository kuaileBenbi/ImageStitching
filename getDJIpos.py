# from PIL import Image
# from PIL.ExifTags import TAGS, GPSTAGS

# def get_exif_data(image_path):
#     """Extract EXIF data from an image file."""
#     image = Image.open(image_path)
#     exif_data = {}

#     # Extract EXIF data
#     exif_info = image._getexif()
#     if exif_info:
#         for tag, value in exif_info.items():
#             decoded_tag = TAGS.get(tag, tag)
#             if decoded_tag == "GPSInfo":
#                 gps_data = {}
#                 for gps_tag in value:
#                     decoded_gps_tag = GPSTAGS.get(gps_tag, gps_tag)
#                     gps_data[decoded_gps_tag] = value[gps_tag]
#                 exif_data[decoded_tag] = gps_data
#             else:
#                 exif_data[decoded_tag] = value
#     return exif_data

# # Use the function
# image_path = '/Users/lixiwang/Desktop/二维-2D/5cm-正射影像-2D-53pic/DJI_20230403143804_0017_V.JPG'
# exif_data = get_exif_data(image_path)
# print(exif_data)

import os
import json
import re

def rename_images(json_path, images_directory):
    # 读取 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    renamed_images = {}

    for item in data:
        # 提取日期和时间
        origin_path = item['origin_path']
        match = re.search(r'(\d{4})(\d{2})(\d{2})\D*(\d{2})(\d{2})(\d{2})', origin_path)
        if match:
            new_name = f"vis_{match.group(2)}{match.group(3)}_{match.group(4)}{match.group(5)}{match.group(6)}.JPG"
            old_path = os.path.join(images_directory, item['path'].lstrip('./'))
            new_path = os.path.join(images_directory, new_name)

            # 重命名文件
            os.rename(old_path, new_path)

            renamed_images[item['id']] = new_name

    return renamed_images

def create_posture_info_json(renamed_images, json_path, output_json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    posture_info = []

    for item in data:
        if item['id'] in renamed_images:
            posture_info.append({
                'id': renamed_images[item['id']],
                'pos_info': item['pos_info']
            })

    with open(output_json_path, 'w', encoding='utf-8') as file:
        json.dump(posture_info, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # renamed_images = rename_images(json_path="survey/image_list.json", images_directory="survey/images")
    # create_posture_info_json(renamed_images, "survey/image_list.json", "posture_info.json")

    def modify_posture_json(json_path, output_json_path):
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        modified_data = {}

        for item in data:
            # 删除 pos_info 中的 id
            if 'id' in item['pos_info']:
                del item['pos_info']['id']
                del item['pos_info']['pos_sigma']
            item['pos_info']['aircraft_angles'] = [0.0, 0.0, 0.0]

            # 以图片 id 作为键，更新后的 pos_info 作为值
            modified_data[item['id']] = item['pos_info']

        with open(output_json_path, 'w', encoding='utf-8') as file:
            json.dump(modified_data, file, ensure_ascii=False, indent=4)

    modify_posture_json('posture_info.json', 'modified_posture_info.json')


