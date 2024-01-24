import json, os, re, cv2
import pandas as pd
from utils import GeometricCorrect_cv
from utils import Geometric_addgeoinfo


"""
step1: 对每幅图像进行几何校正
step2: 给校正后的图像加上地理信息
"""

def getpath(configname):
    with open(configname) as json_file:
        config_data = json.load(json_file)
    return config_data['geoinfo'], config_data['image'], config_data['corrected']

def getgeoinfo(geoinfo_path):
    df = pd.read_csv(geoinfo_path)
    return df.set_index('时间').T.to_dict('dict')

def getimg(img_path):
    try:
        image_filenames = os.listdir(img_path)
    except FileNotFoundError:
        return "Directory not found"
    
    def extract_timestamp(filename):
        # Use regular expression to extract the date and time part from the filename
        match = re.search(r'date(\d+_\d+h\d+m\d+s)', filename)
        return match.group(1) if match else None

    sorted_images = sorted(
        (filename for filename in image_filenames if filename.endswith('.jpg')),
        key=extract_timestamp
    )
    return sorted_images


# 读取图像进行几何校正, 增加地理信息并存储为GeoTiff
def correctimage(current_geoinfo, current_img, current_savename):
    current_camera_position = (current_geoinfo['相机经度'], current_geoinfo['相机纬度'], current_geoinfo['飞艇高度'])
    pod_angles = (0, -90)
    aircraft_angles = (current_geoinfo['飞艇翻滚角'], current_geoinfo['飞艇俯仰角'], current_geoinfo['飞艇航向角'])
    focal_length_mm, sensor_size_mm = 40, (16, 12.8)
    corrected_image = GeometricCorrect_cv.reprojection(current_img,
                                                       current_camera_position,
                                                       pod_angles,
                                                       aircraft_angles,
                                                       focal_length_mm,
                                                       sensor_size_mm)
    coord_string = current_geoinfo['短波视场坐标']
    # 使用正则表达式提取数字
    coord_numbers = re.findall(r'\d+\.\d+', coord_string)
    # 将字符串数字转换为浮点数
    coordinates = [float(num) for num in coord_numbers]
    # 将提取的浮点数分组为坐标点（经度，纬度）
    points = [(coordinates[i], coordinates[i + 1]) for i in range(0, len(coordinates), 2)]

    # 中心点的坐标是所有顶点坐标的平均值
    img_center_lon = sum(point[0] for point in points) / len(points)
    img_center_lat = sum(point[1] for point in points) / len(points)


    horizontal_angle, vertical_angle = current_geoinfo['短波视场角宽（角度）'], current_geoinfo['短波视场角高（角度）']
    Geometric_addgeoinfo.georeference_image_in_local(corrected_image,
                                            img_center_lon,
                                            img_center_lat,
                                            current_camera_position[2],
                                            horizontal_angle,
                                            vertical_angle,
                                            current_savename)

    
if __name__ == "__main__":
    geopath, imgpath, savepath = getpath('path_config.json')
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # print(geopath, imgpath)
    geoinfo = getgeoinfo(geopath)
    # print(geoinfo[0])
    imgnames = getimg(imgpath)
    # print(imgnames)
    for imgname in imgnames:
        current_imgnname = os.path.join(imgpath, imgname)
        current_img = cv2.imread(current_imgnname)
        current_geoinfo = geoinfo[imgname[6:-4]]
        current_savename = os.path.join(savepath, imgname[:-4]+'.tif')
        correctimage(current_geoinfo, current_img, current_savename)

