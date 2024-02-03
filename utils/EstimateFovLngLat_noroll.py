import math
import pyproj

from typing import Dict
from config import CameraParams, ImageParams


def check_pitch(pitch: float) -> bool:
    """
    通过俯仰角值范围 -> 判断是否能正常拍摄到地面

    参数:
        pitch: 俯仰角 (单位: 度) 低头为负，抬头为正，水平为0

    返回:
        pitch>=0 or <-90: 返回false, 认为相机此时只能看到天空或者后面，属于特殊情况都不进行经纬度计算
        -90 < pitch < -5: 返回ture, 认为相机低头朝下看, 可以看到前、下方画面, 进行正常计算
        85.4659997355438 = math.degree(math.arcsin(6371/(6371+20))) => 认为相机低头角度小于5度看不到地球表面
    """
    if -90 <= pitch < -5:
        # -90度 <= 转塔俯仰角 < -5 相机低头朝下
        return True
    else:
        # 转塔俯仰角 > -5 相机无法投影到地面
        # 转塔俯仰角 < -90 相机朝后看 情况太难 本人决定放弃计算
        return False


def calculate_coordinates(camera_params: CameraParams, image_params: ImageParams) -> Dict[str, float]:
    """
    计算视场的中心点、四角的地理坐标。

    参数:
        camera_param:
            lng, lat, alt: 相机地理坐标 (经度、纬度、高度)
            yaw: 方位角 (单位: 度)
            pitch: 俯仰角 (单位: 度)低头为负 抬头为正
        image_param: 
            vertical_angle: 垂直视场角a (单位: 度)
            horizontal_angle: 水平视场角β (单位: 度)

    返回:
        fov_lnglat_coords{
            fov_center_lnglat: 中心点C的地理坐标 (经度、纬度)
            fov_topleft_lnglat: 左上角的地理坐标 (经度、纬度)
            fov_topright_lnglat: 右上角的地理坐标 (经度、纬度)
            fov_bottomright_lnglat: 右下角的地理坐标 (经度、纬度)
            fov_bottomleft_lnglat: 左下角的地理坐标 (经度、纬度)
        }
    """

    if not check_pitch(camera_params.pitch):
        raise ValueError("Sorry! Pitch angle is out of the calculation range!")

    (lng, lat, alt), pitch, yaw = camera_params.coords, camera_params.pitch, camera_params.yaw
    h_fov, v_fov = image_params.horizontal_angle, image_params.vertical_angle
    h_fov_half = h_fov / 2
    v_fov_half = v_fov / 2

    # 地球参考系
    geod = pyproj.Geod(ellps="WGS84")

    # 计算中心点C的地理坐标
    distance_to_center = (
        alt * math.tan(math.radians(v_fov_half + (pitch + 90)))) if pitch != -90.0 else 0
    img_center_lng, img_center_lat = geod.fwd(
        lng, lat, yaw, distance_to_center)[:2]

   # 计算四个角相对于相机中心的方位角和俯仰角
    if pitch == -90:
        angles = [
            (yaw - h_fov_half, v_fov_half),  # 左上
            (yaw + h_fov_half, v_fov_half),  # 右上
            (yaw + 180 - h_fov_half, -v_fov_half),  # 右下
            (yaw + 180 + h_fov_half, -v_fov_half),  # 左下
        ]
    else:
        angles = [
            (yaw - h_fov_half, v_fov + (pitch + 90)),  # 左上
            (yaw + h_fov_half, v_fov + (pitch + 90)),  # 右上
            (yaw - h_fov_half, (pitch + 90)),  # 右下
            (yaw + h_fov_half, (pitch + 90)),  # 左下
        ]

    # 计算四个角的地理坐标
    corner_coords = []
    for angle in angles:
        yaw_corner, pitch_corner = angle
        # 计算对应方向上地面点的距离
        distance_to_corner = alt * math.tan(math.radians(pitch_corner))
        # 计算地理坐标
        corner_lng, corner_lat = geod.fwd(
            lng, lat, yaw_corner, distance_to_corner)[:2]
        corner_coords.append((corner_lng, corner_lat))

    fov_lnglat_coords = {"fov_center_lnglat": (img_center_lng, img_center_lat),
                         "fov_topleft_lnglat": corner_coords[0],
                         "fov_topright_lnglat": corner_coords[1],
                         "fov_bottomleft_lnglat": corner_coords[2],
                         "fov_bottomright_lnglat": corner_coords[3]
                         }

    return fov_lnglat_coords


if __name__ == "__main__":
    camera_params = CameraParams(
        (110, 19, 20000), 0, -90)
    image_params = ImageParams(0, 0, (640, 512))

    fov_geo_coords = calculate_coordinates(
        camera_params, image_params)
    print("==============='-90'=================")
    print("Center (C) coordinates:", fov_geo_coords["fov_center_lnglat"])
    print("Top-left corner coordinates:", fov_geo_coords["fov_topleft_lnglat"])
    print("Top-right corner coordinates:",
          fov_geo_coords["fov_topright_lnglat"])
    print("Bottom-right corner coordinates:",
          fov_geo_coords["fov_bottomright_lnglat"])
    print("Bottom-left corner coordinates:",
          fov_geo_coords["fov_bottomleft_lnglat"])

    camera_params = CameraParams(
        (110, 19, 20000), 0, -80)
    image_params = ImageParams(28, 17, (640, 512))

    fov_geo_coords = calculate_coordinates(
        camera_params, image_params)
    print("==============='-80'=================")
    print("Center (C) coordinates:", fov_geo_coords["fov_center_lnglat"])
    print("Top-left corner coordinates:", fov_geo_coords["fov_topleft_lnglat"])
    print("Top-right corner coordinates:",
          fov_geo_coords["fov_topright_lnglat"])
    print("Bottom-right corner coordinates:",
          fov_geo_coords["fov_bottomright_lnglat"])
    print("Bottom-left corner coordinates:",
          fov_geo_coords["fov_bottomleft_lnglat"])

    camera_params = CameraParams(
        (110, 19, 20000), 0, -45)
    image_params = ImageParams(28, 17, (640, 512))

    fov_geo_coords = calculate_coordinates(
        camera_params, image_params)
    print("==============='-45'=================")
    print("Center (C) coordinates:", fov_geo_coords["fov_center_lnglat"])
    print("Top-left corner coordinates:", fov_geo_coords["fov_topleft_lnglat"])
    print("Top-right corner coordinates:",
          fov_geo_coords["fov_topright_lnglat"])
    print("Bottom-right corner coordinates:",
          fov_geo_coords["fov_bottomright_lnglat"])
    print("Bottom-left corner coordinates:",
          fov_geo_coords["fov_bottomleft_lnglat"])

    camera_params = CameraParams(
        (110.4851, 19.1833, 19038), 15.64, -87)
    image_params = ImageParams(21.8, 17.44, (640, 512))

    fov_geo_coords = calculate_coordinates(
        camera_params, image_params)
    print("=============='short_date0405_07h22m29s'=================")
    print("Center (C) coordinates:", fov_geo_coords["fov_center_lnglat"])
    print("Top-left corner coordinates:", fov_geo_coords["fov_topleft_lnglat"])
    print("Top-right corner coordinates:",
          fov_geo_coords["fov_topright_lnglat"])
    print("Bottom-right corner coordinates:",
          fov_geo_coords["fov_bottomright_lnglat"])
    print("Bottom-left corner coordinates:",
          fov_geo_coords["fov_bottomleft_lnglat"])
