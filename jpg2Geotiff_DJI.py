import json, os, re, cv2
import pandas as pd
from utils import GeometricCorrect_cv_DJI
import numpy as np

"""
step1: 对每幅图像进行几何校正
step2: 给校正后的图像加上地理信息
"""
def parse_intrinsic_matrix(camera_info_file):
    """
    :param: camera_info_file: 相机内参和畸变校正系数

    :return: intrinsic_matrix: 相机内参和畸变校正系数的dict
    """
    with open(camera_info_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

    camera_info={}
    camera_info["intrinsic_matrix"] = np.array([[data["fx"],0,data["cx"]],
                                                  [0,data["fy"],data["cy"]],
                                                  [0,0,1]], dtype="float32")
    camera_info["size_wh"] = [data["photo_width"], data["photo_height"]]
    camera_info["distortion_k12p12k3"] = [data["k1"], data["k2"], data["p1"], data["p2"], data["k3"]]
    camera_info["focal_length"] = data["focal_length"]
    camera_info["horizontal_fov"] = data["horizontal_fov"]
    camera_info["vertical_fov"] = data["vertical_fov"]
    return camera_info

def parse_extrinsic_matrix(posture_info_file):
    """
    :param: posture_info_file: 相机外参

    :return: 所有图片的外参dict, key:xx.jpg value:"pos":llh "orientation":ypr
    """
    with open(posture_info_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    return data

def prepare_original_images(input_directory):
    """
    :param: input_directory: 原始图像路径

    :return: 图像名为key,图像路径为value的字典
    """
    def sort_key(filename):
        # 使用正则表达式提取日期和时间部分
        match = re.match(r"vis_(\d{4})_(\d{6})\.JPG", filename)
        if match:
            # 返回 "mmddhhmmss" 格式的字符串，用于排序
            return match.group(1) + match.group(2)
        else:
            # 如果文件名不符合预期的格式，返回原始文件名
            return filename
    imgnames = os.listdir(input_directory)
    imgnames = sorted(imgnames, key=sort_key)
    original_images = {imgname: os.path.join(input_directory, imgname) for imgname in imgnames}
    return original_images
    

def main(camera_info_file, posture_info_file, input_directory, output_directory):
    """
    :param: camera_info_file: 相机内参和畸变校正系数, 若没有畸变校正参数json文件可设为0
    :param: posture_info_file: 相机外参
    :param: input_directory: 原始图像
    :param: output_directory: 几何校正+地理信息后的Geotiff保存路径

    :return: None
    """
    original_images = prepare_original_images(input_directory)
    posture_info = parse_extrinsic_matrix(posture_info_file)
    camera_info = parse_intrinsic_matrix(camera_info_file)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for imgname, imgpath in original_images.items():
        original_image = cv2.imread(imgpath)
        savepath = os.path.join(output_directory, imgname[:-4]+".tif")
        GeometricCorrect_cv_DJI.reprojection(original_image, camera_info, posture_info[imgname], savepath)

    
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Convert DJI images to GeoTIFF format.")

    parser.add_argument("--camera_info_file", default="camera_info.json", help="Path to the camera info JSON file.")
    parser.add_argument("--posture_info_file", default="modified_posture_info.json", help="Path to the posture info JSON file.")
    parser.add_argument("--input_directory", default="survey/DJI_images", help="Directory containing DJI images.")
    parser.add_argument("--output_directory", default="corrected_DJI_images", help="Directory to save the corrected DJI images.")

    # 解析命令行参数
    args = parser.parse_args()

    main(args.camera_info_file, args.posture_info_file, args.input_directory, args.output_directory)
