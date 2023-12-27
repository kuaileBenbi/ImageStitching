import cv2
import numpy as np
from typing import Tuple, Optional

"""
基于针孔相机模型的几何校正：
step1: 根据传感器参数-焦距、探测器尺寸、图像尺寸，计算内参矩阵K
step2: 根据相机和载机的姿态、位置，计算外参矩阵P=[R|t]（位置用转换到笛卡尔坐标系）
step3: 根据像素坐标系-相机坐标系、相机坐标系-世界坐标系关系，利用内外参数矩阵计算出畸变图像中四个角点的像素坐标对应的世界坐标（笛卡尔坐标系）
step4: 对世界坐标进行平移、缩放变换使其范围与原始图像分辨率一致，同时保证长宽比、长宽关系
step5: 利用opencv的cv2.findHomography求出投影变换矩阵，cv2.warpProjective进行投影变换得到校正后的图像
"""


def geodetic_to_ecef(lon: float, lat: float, height: float) -> np.ndarray:
    """
    由载机经纬高计算载机在WGS-84地心空间直角坐标系的坐标

    :param: lon: 经度
    :param: lat: 纬度
    :param: height: 高度

    :return: 地心空间直角坐标系坐标-(X,Y,Z)
    """
    # WGS 84 椭球参数
    a = 6378137.0  # 赤道半径
    b = 6356752.3142  # 极半径
    e_sq = 1 - (b**2 / a**2)  # 第一偏心率的平方

    # 将经纬度转换为弧度
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)

    # 计算卯酉圈半径
    N = a / np.sqrt(1 - e_sq * np.sin(lat_rad)**2)

    # 计算笛卡尔坐标
    X = (N + height) * np.cos(lat_rad) * np.cos(lon_rad)
    Y = (N + height) * np.cos(lat_rad) * np.sin(lon_rad)
    Z = ((1 - e_sq) * N + height) * np.sin(lat_rad)

    return X, Y, Z


def intrinsic_matrix(focal_length: float, sensor_size: Tuple[float, float], image_size: Tuple[int, int]) -> np.ndarray:
    """
    根据相机参数计算相机的内参矩阵，表示图像坐标系到像素坐标系的转换

    :param: focal_length: 相机焦距 （单位：mm)
    :param: sensor_size: 传感器尺寸（单位：mm）
    :param: image_size: 图像分辨率（单位：像素）

    :return: K -> 3*3
    """
    # 将焦距从毫米转换为像素单位
    focal_length_px = (focal_length / sensor_size[0]) * image_size[0]
    focal_length_py = (focal_length / sensor_size[1]) * image_size[1]

    # 构建相机内参矩阵
    K = np.array([[focal_length_px, 0, image_size[0] / 2],
                  [0, focal_length_py, image_size[1] / 2],
                  [0, 0, 1]])
    return K


def external_matrix(pod_azimuth_deg: float,
                    pod_elevation_deg: float,
                    aircraft_yaw_deg: float,
                    aircraft_pitch_deg: float,
                    aircraft_roll_deg: float) -> np.ndarray:
    """
    将地图坐标转换为相机坐标系下的图像坐标（外参矩阵）

    :param pod_azimuth_deg: 吊舱方位角    (单位：度)
    :param pod_elevation_deg: 吊舱俯仰角  (单位：度)
    :param aircraft_yaw_deg: 平台航向角   (单位：度)
    :param aircraft_pitch_deg: 平台俯仰角 (单位：度)
    :param aircraft_roll_deg: 平台横滚角  (单位：度)

    :return: C_cm -> （3*4 · 4*4）= 3*4
    """

    pod_azimuth_rad, pod_elevation_rad, aircraft_yaw_rad, aircraft_pitch_rad, aircraft_roll_rad = map(
        np.radians, [pod_azimuth_deg, pod_elevation_deg, aircraft_yaw_deg, aircraft_pitch_deg, aircraft_roll_deg])

    c_p = np.array([[np.cos(pod_azimuth_rad) * np.sin(pod_elevation_rad), np.sin(pod_elevation_rad) * np.sin(pod_azimuth_rad), -np.cos(pod_elevation_rad), 0],
                    [-np.sin(pod_azimuth_rad), np.cos(pod_azimuth_rad), 0, 0,],
                    [np.cos(pod_azimuth_rad) * np.cos(pod_elevation_rad), np.sin(pod_azimuth_rad) * np.cos(pod_elevation_rad), np.sin(pod_elevation_rad), 0]])

    c_a = np.array([[np.cos(aircraft_pitch_rad)*np.cos(aircraft_yaw_rad), np.cos(aircraft_pitch_rad)*np.sin(aircraft_yaw_rad), -np.sin(aircraft_pitch_rad), 0],
                    [np.cos(aircraft_yaw_rad)*np.sin(aircraft_roll_rad)*np.sin(aircraft_pitch_rad)-np.cos(aircraft_roll_rad)*np.sin(aircraft_yaw_rad),
                     np.cos(aircraft_roll_rad)*np.cos(aircraft_yaw_rad)+np.sin(
                         aircraft_roll_rad)*np.sin(aircraft_pitch_rad)*np.sin(aircraft_yaw_rad),
                     np.cos(aircraft_pitch_rad)*np.sin(aircraft_roll_rad), 0],
                    [np.sin(aircraft_roll_rad)*np.sin(aircraft_yaw_rad)+np.cos(aircraft_roll_rad)*np.cos(aircraft_yaw_rad)*np.cos(aircraft_pitch_rad),
                     np.cos(aircraft_roll_rad)*np.sin(aircraft_pitch_rad)*np.sin(
                         aircraft_yaw_rad)-np.cos(aircraft_yaw_rad)*np.sin(aircraft_roll_rad),
                     np.cos(aircraft_roll_rad)*np.cos(aircraft_pitch_rad), 0],
                    [0, 0, 0, 1]])

    return c_p @ c_a


def transform_pixel2world(current_pixel_coordinates: np.ndarray, K: np.ndarray, C_cm: np.ndarray) -> np.ndarray:
    """
    由原始图像每个像素计算出对应物点位置在世界中的坐标

    :param: current_pixel_coordinates: 原始图像中的像素索引（u, v）
    :param: c_cm: 相机的外参矩阵
    :param: K: 相机的内参矩阵

    :return: 地图坐标 -> 3*1
    """
    C = K @ C_cm   # 3*3 · 3*4 = 3*4
    C = np.delete(C, -2, axis=1)  # 在Z=0平面, 3*3
    current_world_coordinate = np.linalg.pinv(C) @ current_pixel_coordinates
    current_world_coordinate /= current_world_coordinate[2]
    return current_world_coordinate


def normalize_coordinates(originate_crd: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """
    规范化世界坐标-平移原点到图像中心、归一化到-1 ～ 1范围、缩放到合适原始图像的范围

    :param: originate_crd: 原始坐标
    :param: image_size: 图像分辨率

    :return: 规范化后的坐标
    """
    originate_crd_ = originate_crd[:, 0:2]
    centroid = np.mean(originate_crd_, axis=0)
    translated_coords = originate_crd_ - centroid
    max_abs_xy = np.max(np.abs(translated_coords), axis=0)[:2]
    scale_factor = 1 / max_abs_xy
    normalized_coords = translated_coords * scale_factor
    normalized_coords[:, 0] *= image_size[0] - 1  # 缩放到图像宽度范围
    normalized_coords[:, 1] *= image_size[1] - 1  # 缩放到图像高度范围

    return normalized_coords


def reprojection(image_path: str,
                 current_camera_position: Tuple[float, float, float],
                 pod_angles: Tuple[float, float],
                 aircraft_angles: Tuple[float, float, float],
                 focal_length_mm: float,
                 sensor_size_mm: Tuple[float, float],
                 save_path: Optional[str] = None,) -> np.ndarray:
    """
    利用opencv的cv2.findHomography求出投影变换矩阵，cv2.warpProjective进行投影变换得到校正后的图像

    :param: image_path: 待处理畸变图像的路径
    :param: save_path: 保存路径
    :param: current_camera_position: 相机当前地理位置（经、纬、高，单位：度，米）
    :param: pod_angles: 载机姿态（横滚roll, 俯仰pitch, 航向yaw）
    :param: aircraft_angles: 吊舱姿态（方位azimuth, 高低elevation）
    :param: focal_length_mm: 焦距（单位：毫米）
    :param: sensor_size_mm: 传感器尺寸（长、宽，单位：毫米）

    :return: corrected_image_rgba: 校正后的图像
    """
    # 加载图像
    image = cv2.imread(image_path)
    image_width, image_height = image.shape[1], image.shape[0]

    # 图像四角的像素坐 (齐次坐标)
    corner_pixel_coordinates = np.array([[0, 0, 1],
                                         [image_width - 1, 0, 1],
                                         [image_width - 1, image_height - 1, 1],
                                         [0, image_height - 1, 1]], dtype="float32")
    # 内参矩阵
    k = intrinsic_matrix(focal_length_mm, sensor_size_mm,
                         (image_width, image_height))

    # 外参矩阵
    cam_center_in_world = geodetic_to_ecef(*current_camera_position)
    R = external_matrix(pod_azimuth_deg=pod_angles[0],
                        pod_elevation_deg=pod_angles[1],
                        aircraft_yaw_deg=aircraft_angles[0],
                        aircraft_pitch_deg=aircraft_angles[1],
                        aircraft_roll_deg=aircraft_angles[2])
    R[:, 3] = cam_center_in_world

    # 原始世界坐标
    corner_world_coordinates = np.zeros_like(corner_pixel_coordinates)

    # =============================法1===================================#
    # inv_k = np.linalg.inv(k)
    # R_c = np.delete(R, -2, axis=1)  # 在z=0平面生成
    # inv_R_c = np.linalg.inv(R_c)
    # for i, crd_pixel in enumerate(corner_pixel_coordinates):
    #     crd_cam = inv_k @ crd_pixel
    #     crd_world = inv_R_c @ crd_cam
    #     crd_world /= crd_world[2]
    #     corner_world_coordinates[i] = crd_world
    #     # print(f"像素坐标系到世界坐标系：{crd_pixel} -> {crd_world}")
    # =================================================================#

    # ==============================法2==================================#
    for i, crd_pixel in enumerate(corner_pixel_coordinates):
        crd_world = transform_pixel2world(crd_pixel, k, R)
        corner_world_coordinates[i] = crd_world
    # ================================================================#

    # 规范化世界坐标
    normalized_coords = normalize_coordinates(
        corner_world_coordinates, (image_width, image_height))
    # print(f"规范化世界坐标: {normalized_coords}")

    # 计算单应性矩阵
    H, _ = cv2.findHomography(
        corner_pixel_coordinates[:, 0:2], normalized_coords)
    # print(f"单应性矩阵H: {H}")

    # 应用单应性变换
    corrected_image = cv2.warpPerspective(
        image, H, (image.shape[1], image.shape[0]))

    # 将BGR图像转换为RGBA
    corrected_image_rgba = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2BGRA)

    # 假设校正过程中产生的空白区域为黑色（值为[0, 0, 0]）
    # 创建一个掩码，标记所有黑色像素
    corrected_image_hsv = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0])
    upper = np.array([0, 0, 0])
    mask = cv2.inRange(corrected_image_hsv, lower, upper)

    # 将掩码中的黑色像素设置为透明
    corrected_image_rgba[mask == 255] = (0, 0, 0, 0)

    # 保存图像为PNG格式，保留透明度
    # cv2.imshow('Corrected Image', corrected_image_rgba)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if save_path:
        cv2.imwrite(save_path, corrected_image_rgba)
    return corrected_image_rgba


if __name__ == "__main__":
    current_camera_position = (
        110.49602631, 19.19247651, 19038)  # lng，lat，altitude
    aircraft_angles = (3.93, -2.69, 15.64)  # roll, pitch, yaw -> 飞艇
    pod_angles = (0, -90)                   # azimuth, elevation -> 吊舱
    focal_length_mm = 40                   # 焦距，毫米
    sensor_size_mm = (16, 12.8)            # 传感器尺寸, 长宽，mm

    image_path = 'datasets/1/short_date0405_07h22m29s.jpg'
    save_path = 'corrected_image_transparent.png'
    corrected_image = reprojection(image_path, current_camera_position,
                                   pod_angles, aircraft_angles, focal_length_mm, sensor_size_mm, save_path)
