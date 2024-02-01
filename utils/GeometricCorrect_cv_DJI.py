import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
from osgeo import gdal, osr
import pyproj
import math

"""
基于针孔相机模型的几何校正：
step1: 根据传感器参数-焦距、探测器尺寸、图像尺寸，计算内参矩阵K
step2: 根据相机和载机的姿态、位置，计算外参矩阵P=[R|t]（位置用转换到笛卡尔坐标系）
step3: 根据像素坐标系-相机坐标系、相机坐标系-世界坐标系关系，利用内外参数矩阵计算出畸变图像中四个角点的像素坐标对应的世界坐标（笛卡尔坐标系）
step4: 对世界坐标进行平移、缩放变换使其范围与原始图像分辨率一致，同时保证长宽比、长宽关系
step5: 利用opencv的cv2.findHomography求出投影变换矩阵，cv2.warpProjective进行投影变换得到校正后的图像
"""


def geodetic_to_ned(lon: float, lat: float, height: float) -> np.ndarray:
    """
    由相机经纬高计算相机在NED坐标系下的坐标

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
    N = a / np.sqrt(1 - e_sq * math.sin(lat_rad)**2)

    # 计算笛卡尔坐标
    X = (N + height) * math.cos(lat_rad) * math.cos(lon_rad)
    Y = (N + height) * math.cos(lat_rad) * math.sin(lon_rad)
    Z = ((1 - e_sq) * N + height) * math.sin(lat_rad)

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    ecef_to_ned_matrix = np.array([
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [-sin_lon, cos_lon, 0],
        [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat]
    ])

    ecef_coords = np.array([X, Y, Z])

    # 应用转换矩阵
    ned_coords = ecef_to_ned_matrix @ ecef_coords


    return ned_coords


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

    # c_p = np.array([[math.cos(pod_azimuth_rad) * math.sin(pod_elevation_rad), math.sin(pod_elevation_rad) * math.sin(pod_azimuth_rad), -math.cos(pod_elevation_rad), 0],
    #                 [-math.sin(pod_azimuth_rad), math.cos(pod_azimuth_rad), 0, 0,],
    #                 [math.cos(pod_azimuth_rad) * math.cos(pod_elevation_rad), math.sin(pod_azimuth_rad) * math.cos(pod_elevation_rad), math.sin(pod_elevation_rad), 0]])

    # c_a = np.array([[math.cos(aircraft_pitch_rad)*math.cos(aircraft_yaw_rad), math.cos(aircraft_pitch_rad)*math.sin(aircraft_yaw_rad), -math.sin(aircraft_pitch_rad), 0],
    #                 [math.cos(aircraft_yaw_rad)*math.sin(aircraft_roll_rad)*math.sin(aircraft_pitch_rad)-math.cos(aircraft_roll_rad)*math.sin(aircraft_yaw_rad),
    #                  math.cos(aircraft_roll_rad)*math.cos(aircraft_yaw_rad)+math.sin(
    #                      aircraft_roll_rad)*math.sin(aircraft_pitch_rad)*math.sin(aircraft_yaw_rad),
    #                  math.cos(aircraft_pitch_rad)*math.sin(aircraft_roll_rad), 0],
    #                 [math.sin(aircraft_roll_rad)*math.sin(aircraft_yaw_rad)+math.cos(aircraft_roll_rad)*math.cos(aircraft_yaw_rad)*math.cos(aircraft_pitch_rad),
    #                  math.cos(aircraft_roll_rad)*math.sin(aircraft_pitch_rad)*math.sin(
    #                      aircraft_yaw_rad)-math.cos(aircraft_yaw_rad)*math.sin(aircraft_roll_rad),
    #                  math.cos(aircraft_roll_rad)*math.cos(aircraft_pitch_rad), 0],
    #                 [0, 0, 0, 1]])
    c_p = euler_to_rotation_matrix(pod_azimuth_deg, pod_elevation_deg, 0)
    c_a = euler_to_rotation_matrix(aircraft_yaw_deg, aircraft_pitch_deg,aircraft_roll_deg)

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
    规范化世界坐标-平移原点到图像中心、归一化到0-1范围、缩放到合适原始图像的范围

    :param: originate_crd: 原始坐标
    :param: image_size: 图像分辨率（宽、高）

    :return: 规范化后的坐标
    """
    originate_crd_ = originate_crd[:, 0:2]
    centroid = np.mean(originate_crd_, axis=0)
    translated_coords = originate_crd_ - centroid
    # 找到平移后坐标的最大和最小值
    min_coords = np.min(translated_coords, axis=0)
    max_coords = np.max(translated_coords, axis=0)
    # 将坐标映射到 0-1范围
    normalized_world_coords = (translated_coords - min_coords) / (max_coords - min_coords)

    normalized_world_coords[:, 0] *= image_size[0] - 1  # 缩放到图像宽度范围
    normalized_world_coords[:, 1] *= image_size[1] - 1  # 缩放到图像高度范围

    return normalized_world_coords


def calculate_pixel_coverage(alt: float, fov: float, size: float) -> float:
    """
    计算每个像素水平覆盖的范围 单位：米

    参数:
        alt: 相机高度
        fov: 视场角 (水平/垂直，单位: 度)
        size: 图像尺寸 水平/垂直

    返回:
        每个像素水平覆盖的范围 单位：米
    """
    D = 2 * alt * math.tan(math.radians(fov) / 2)

    return D / size


def compute_pixel_size_ellipsoid(pixel_distance_meters: float) -> Tuple[float, float]:
    """
    根据地球的椭球模型来计算像素大小对应的经纬度差值

    参数:
        pixel_distance_meters: 每个像素水平、垂直代表的米数

    返回:
        每个像素水平、垂直代表的米数对应的经、纬度数
    """
    geod = pyproj.Geod(ellps="WGS84")

    # 计算东边一定距离的点的经度
    delta_lon, _, _ = geod.fwd(
        0, 0, 90, pixel_distance_meters[0])

    # 计算北边一定距离的点的纬度
    _, delta_lat, _ = geod.fwd(
        0, 0, 0, pixel_distance_meters[1])
    return delta_lon, delta_lat


def compute_GeoTransform(img_center_llh:Tuple[float],
                         fov:Tuple[float],
                         image_size:Tuple[int]):
    """
    计算地理信息

    :param: img_center_llh[lat, lng, altitude]
    :param: fov[horizontal_fov, vertical_fov]: 水平视场角, 垂直视场角
    :param: image_size[width, height]


    :return: List[float]
        1.左上角 X 坐标: 栅格的左上角的 X 坐标。
        2.Pixel Width: 一个像素在 X 方向上的尺寸。
        3.Rotation (about Y-axis): 通常为 0，除非存在旋转。
        4.左上角 Y 坐标: 栅格的左上角的 Y 坐标。
        5.Rotation (about X-axis): 通常为 0，除非存在旋转。
        6.Pixel Height: 一个像素在 Y 方向上的尺寸。
        note: 为统一单位，函数中全部采用经纬度的度作为单位
    """
    pixel_width_deg, pixel_height_deg = compute_pixel_size_ellipsoid([calculate_pixel_coverage(img_center_llh[2], fov[0], image_size[0]),
                                                                      calculate_pixel_coverage(img_center_llh[2], fov[1], image_size[1])])

    ULx = img_center_llh[1] - (pixel_width_deg * img_center_llh[2]) / 2
    ULy = img_center_llh[0] + (pixel_height_deg * img_center_llh[2]) / 2

    return [ULx, pixel_width_deg, 0, ULy, 0, -pixel_height_deg]


def georeference_image_in_local(corrected_image: np.ndarray, 
                                image_size:Tuple[int], 
                                img_center_llh:Tuple[float], 
                                fov:Tuple[float],
                                save_path:str):
    """
    输入一副原始图像，根据图像中心经纬度进行地理校正，保存在本地
    :param: corrected_image: 几何校正后的图像，numpy类型
    :param: image_size[width, height]
    :param: img_center_llh[lat, lng, altitude]
    :param: fov[horizontal_fov, vertical_fov]: 水平视场角, 垂直视场角
    :param: save_path: 保存路径

    :return: None
    """
    img_width, img_height = image_size
    mem_driver = gdal.GetDriverByName('MEM')
    mem_ds = mem_driver.Create('', img_width, img_height, 4, gdal.GDT_Byte)

    band_count = corrected_image.shape[2]

    # 设置背景为透明
    if band_count == 1:
        corrected_image = cv2.cvtColor(corrected_image, cv2.COLOR_GRAY2RGB)
    for i in range(band_count):
        band = mem_ds.GetRasterBand(i + 1)
        band.WriteArray(corrected_image[:,:,i])

    alpha_band = mem_ds.GetRasterBand(4)
    alpha_band.Fill(255) 

    geotransform = compute_GeoTransform(img_center_llh, fov, image_size)
    mem_ds.SetGeoTransform(geotransform)

    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS('WGS84')
    mem_ds.SetProjection(srs.ExportToWkt())
    driver = gdal.GetDriverByName('GTiff')
    driver.CreateCopy(save_path, mem_ds, 0)
    mem_ds = None

    print("Correct done for local!")


def euler_to_rotation_matrix(yaw, pitch, roll):
    # 将角度转换为弧度
    yaw, pitch, roll = map(math.radians, [yaw, pitch, roll])
    
    # 定义各轴旋转矩阵
    Rz_yaw = np.array([
        [math.cos(yaw), math.sin(yaw), 0],
        [-math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    Ry_pitch = np.array([
        [math.cos(pitch), 0, -math.sin(pitch)],
        [0, 1, 0],
        [math.sin(pitch), 0, math.cos(pitch)]
    ], dtype=np.float32)

    Rx_roll = np.array([
        [1, 0, 0],
        [0, math.cos(roll), math.sin(roll)],
        [0, -math.sin(roll), math.cos(roll)]
    ], dtype=np.float32)

    return Rz_yaw @ Ry_pitch @ Rx_roll

def calculate_distance_to_center(alt, v_fov_half, R_total):
    """
    考虑平台姿态和吊舱姿态，重新计算相机视场中心点的距离。
    """
    # 视场中心方向向量（相机坐标系中，朝向正前方）
    center_vector = np.array([1, 0, 0])
    # 应用旋转矩阵，将方向向量转换到全球坐标系
    center_vector_global =  R_total @ center_vector
    # print(f"center_vector_global :{center_vector_global}")
    # 计算俯仰角（相对于水平面）
    pitch_global = np.arcsin(center_vector_global[2])
    # print(f"俯仰角: {np.degrees(pitch_global)}")
    # 计算距离
    distance_to_center = alt * math.tan(v_fov_half + (np.pi / 2 + pitch_global) ) if pitch_global != -np.pi / 2 else 0
    # print(f"distance_to_center: {distance_to_center}")

    return distance_to_center


def calculate_fov_corner_vectors(horizontal_fov, vertical_fov):
    """
    根据相机光轴中心的方向向量和视场角度，计算视场四个角点的方向向量。
    在NED坐标系中，相机光轴中心的方向向量 [1, 0, 0] 意味着相机指向北方。
    """
    # 将角度转换为弧度
    h_fov_half = np.radians(horizontal_fov / 2)
    v_fov_half = np.radians(vertical_fov / 2)

    # 四个角点的方向向量
    top_left = np.array([
        math.cos(v_fov_half) * math.cos(-h_fov_half),
        math.cos(v_fov_half) * math.sin(-h_fov_half),
        -math.sin(v_fov_half)
    ])
    top_right = np.array([
        math.cos(v_fov_half) * math.cos(h_fov_half),
        math.cos(v_fov_half) * math.sin(h_fov_half),
        -math.sin(v_fov_half)
    ])
    bottom_left = np.array([
        math.cos(-v_fov_half) * math.cos(-h_fov_half),
        math.cos(-v_fov_half) * math.sin(-h_fov_half),
        -math.sin(-v_fov_half)
    ])
    bottom_right = np.array([
        math.cos(-v_fov_half) * math.cos(h_fov_half),
        math.cos(-v_fov_half) * math.sin(h_fov_half),
        -math.sin(-v_fov_half)
    ])

    return top_left, top_right, bottom_left, bottom_right


def calculate_view_vectors(rotation_mat, vectors):
    center_vector = rotation_mat @ vectors["center"]
    top_right = rotation_mat @ vectors["topright"]
    top_left = rotation_mat @ vectors["topleft"]
    bottom_right = rotation_mat @ vectors["bottomright"]
    bottom_left = rotation_mat @ vectors["bottomleft"]

    return center_vector, top_right, top_left, bottom_right, bottom_left

def calculate_azimuth_and_elevation(vector):
    azimuth = math.atan2(vector[1], vector[0])
    horizontal_distance = math.sqrt(vector[0]**2 + vector[1]**2)
    elevation = math.atan2(vector[2], horizontal_distance)
    return math.degrees(azimuth), math.degrees(elevation)


def calculate_distance_to_ground(camera_altitude, elevation_angle):
    return camera_altitude / math.tan(math.radians(elevation_angle))


def calculate_coordinates(camera_info:Dict[str, float], posture_info:Dict[str, List[float]])-> Dict[str, float]:
    """
    计算视场的中心点、四角的地理坐标。
    :param: camera_info: 相机内参（畸变系数，如果有）
    :param: posture_info: 地理位置及姿态
                        （纬、经、高，单位：度，米/平台航向角、俯仰角、横滚角，单位：度/吊舱方位, 高低，单位：度）

    :return: fov_lnglat_coords{
            fov_center_lnglat: 中心点C的地理坐标 (经度、纬度)
            fov_topleft_lnglat: 左上角的地理坐标 (经度、纬度)
            fov_topright_lnglat: 右上角的地理坐标 (经度、纬度)
            fov_bottomright_lnglat: 右下角的地理坐标 (经度、纬度)
            fov_bottomleft_lnglat: 左下角的地理坐标 (经度、纬度)}
    """
    pod_azimuth_deg=posture_info["orientation"][0]
    pod_elevation_deg=posture_info["orientation"][1]

    aircraft_yaw_deg=posture_info["aircraft_angles"][0]
    aircraft_pitch_deg=posture_info["aircraft_angles"][1]
    aircraft_roll_deg = posture_info["aircraft_angles"][2]

    camera_lng = posture_info["pos"][1]
    camera_lat = posture_info["pos"][0]
    camera_alatitude = posture_info["pos"][2]
    horizontal_fov = camera_info["horizontal_fov"]
    vertical_fov = camera_info["vertical_fov"]
    
    R_aircraft = euler_to_rotation_matrix(aircraft_yaw_deg, aircraft_pitch_deg, aircraft_roll_deg)
    R_pod = euler_to_rotation_matrix(pod_azimuth_deg, pod_elevation_deg, 0)
    R_total = R_aircraft @ R_pod
    # v_fov_half = np.radians(vertical_fov / 2)
    # distance_to_center = calculate_distance_to_center(camera_alatitude, v_fov_half, R_total)
    geod = pyproj.Geod(ellps="WGS84")
    # 计算视场中心点的经纬度
    # total_yaw = aircraft_yaw_deg + pod_azimuth_deg
    # center_lng, center_lat, _ = geod.fwd(camera_lng, camera_lat, total_yaw, distance_to_center)
    # print(f"center_lng, center_lat: {center_lng}, {center_lat}")
    top_left, top_right, bottom_left, bottom_right = calculate_fov_corner_vectors(horizontal_fov, vertical_fov)
    # 计算四个角点的经纬度
    vectors = {
        "center":np.array([1,0,0]),
        "topleft":top_left,
        "topright":top_right,
        "bottomleft":bottom_left,
        "bottomright":bottom_right
    }
    # print(f"vectors: {vectors}")
    center_vector, top_right, top_left, bottom_right, bottom_left = calculate_view_vectors(R_total, vectors)
    rotated_vectors = {
        "center":center_vector,
        "topleft":top_left,
        "topright":top_right,
        "bottomleft":bottom_left,
        "bottomright":bottom_right
    }
    # print(f"rotated_vectors: {rotated_vectors}")
    coordinates = []
    for _, rotated_vector in rotated_vectors.items():
        azimuth, elevation = calculate_azimuth_and_elevation(rotated_vector)
        distance = calculate_distance_to_ground(camera_alatitude, elevation)
        # 计算经纬度
        new_lng, new_lat, _ = geod.fwd(camera_lng, camera_lat, azimuth, distance)
        coordinates.append((new_lng, new_lat))

    fov_lnglat_coords = {"fov_center_lnglat": coordinates[0],
                         "fov_topright_lnglat": coordinates[1],
                         "fov_topleft_lnglat": coordinates[2],
                         "fov_bottomright_lnglat": coordinates[3],
                         "fov_bottomleft_lnglat": coordinates[4]
                         }
    return fov_lnglat_coords


def reprojection(image: np.ndarray,
                 camera_info:Dict[str, float],
                 posture_info:Dict[str, Dict[str, List[float]]],
                 save_path: str):
    """
    利用opencv的cv2.findHomography求出投影变换矩阵，cv2.warpProjective进行投影变换得到校正后的图像
    利用gdal添加地理信息并保存为Geotiff格式

    :param: image_path: 待处理畸变图像的路径
    :param: camera_info: 相机内参（畸变系数，如果有）
    :param: posture_info: 地理位置及姿态
                        （纬、经、高，单位：度，米/平台航向角、俯仰角、横滚角，单位：度/吊舱方位, 高低，单位：度）
    :param: save_path: 保存路径

    :return: None
    """
    image_width, image_height = camera_info["size_wh"][0], camera_info["size_wh"][1]

    # 图像四角的像素坐 (齐次坐标)
    corner_pixel_coordinates = np.array([[0, 0, 1],
                                         [image_width - 1, 0, 1],
                                         [image_width - 1, image_height - 1, 1],
                                         [0, image_height - 1, 1]], dtype="float32")

    # 内参矩阵
    k = camera_info["intrinsic_matrix"]
    # 畸变系数，如果不为零
    distortion_k12p12k3 = camera_info["distortion_k12p12k3"]

    # 外参矩阵
    # cam_center_in_world = geodetic_to_ned(posture_info["pos"][1], posture_info["pos"][0], posture_info["pos"][2])
    R = external_matrix(pod_azimuth_deg=posture_info["orientation"][0],
                        pod_elevation_deg=posture_info["orientation"][1],
                        aircraft_yaw_deg=posture_info["aircraft_angles"][0],
                        aircraft_pitch_deg=posture_info["aircraft_angles"][1],
                        aircraft_roll_deg=posture_info["aircraft_angles"][2])
    # R[:, 3] = cam_center_in_world

    # ==============================计算相机坐标====================================#
    corner_cam_coordinates = np.zeros_like(corner_pixel_coordinates)
    for i, crd_pixel in enumerate(corner_pixel_coordinates):
        crd_cam = np.linalg.pinv(k) @ crd_pixel
        corner_cam_coordinates[i] = crd_cam
   
    # ==============================计算世界坐标====================================#
    corner_world_coordinates = np.zeros_like(corner_pixel_coordinates)
    # for i, crd_pixel in enumerate(corner_pixel_coordinates):
    #     crd_world = transform_pixel2world(crd_pixel, k, R)
    #     corner_world_coordinates[i] = crd_world
    # normalized_world_coords = normalize_coordinates(corner_world_coordinates, (image_width, image_height))
    # print(f"规范化世界坐标: {normalized_world_coords}")
    
    for i, crd_cam in enumerate(corner_cam_coordinates):
        crd_world = np.linalg.pinv(R) @ crd_cam
        crd_world /= crd_world[2]
        corner_world_coordinates[i] = crd_world

    normalized_world_coords = normalize_coordinates(corner_world_coordinates, (image_width, image_height))

    #================#
    fov_coords= calculate_coordinates(camera_info, posture_info)
    # import matplotlib.pyplot as plt

    # # 提取经纬度坐标
    # lngs, lats = zip(*fov_coords.values())

    # # 创建图形和轴
    # plt.figure(figsize=(10, 8))
    # plt.plot(lngs, lats, 'o', color='blue')  # 绘制点和连接线
    # plt.scatter([fov_coords["fov_center_lnglat"][0]], [fov_coords["fov_center_lnglat"][1]], color='red')  # 绘制中心点

    # # 标注点
    # for key, (lng, lat) in fov_coords.items():
    #     plt.text(lng, lat, key)

    # # 设置标题和坐标轴标签
    # plt.title("Field of View Coordinates Visualization")
    # plt.xlabel("Longitude")
    # plt.ylabel("Latitude")
    # plt.grid(True)  # 显示网格

    # # 显示图形
    # plt.show()
    #================#
    fov_lnglat_coords = fov_coords
    
    # corner_world_coordinates[0] = np.array([fov_lnglat_coords["fov_topleft_lnglat"][0], fov_lnglat_coords["fov_topleft_lnglat"][1], 1], dtype=np.float32)
    # corner_world_coordinates[3] = np.array([fov_lnglat_coords["fov_topright_lnglat"][0], fov_lnglat_coords["fov_topright_lnglat"][1], 1], dtype=np.float32)
    # corner_world_coordinates[2] = np.array([fov_lnglat_coords["fov_bottomright_lnglat"][0], fov_lnglat_coords["fov_bottomright_lnglat"][1], 1], dtype=np.float32)
    # corner_world_coordinates[1] = np.array([fov_lnglat_coords["fov_bottomleft_lnglat"][0], fov_lnglat_coords["fov_bottomleft_lnglat"][1], 1], dtype=np.float32)
    # normalized_world_coords = normalize_coordinates(corner_world_coordinates, (image_width, image_height))
    
    # ==============================计算单应性矩阵==================================#
    H, _ = cv2.findHomography(corner_pixel_coordinates[:, 0:2], normalized_world_coords)
    # print(f"单应性矩阵H: {H}")

    # =============================应用单应性变换====================================#
    # cv2.imshow("distort_image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    distortion_coeffs = (lambda lst: [x for x in lst if x != 0])(distortion_k12p12k3)
    # print(f"distortion_coeffs: {np.array(distortion_coeffs)}") 
    if len(distortion_coeffs) > 0:
        image = cv2.undistort(image, k, np.array(distortion_coeffs))
    # cv2.imshow("undistort_image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    corrected_image = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]))
    # cv2.imwrite(save_path, corrected_image)
    # cv2.imshow("corrected_image", corrected_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # ========================== 增加地理信息并保存====================================#
    
    img_center_lon, img_center_lat = fov_lnglat_coords["fov_center_lnglat"]
    georeference_image_in_local(corrected_image, 
                                (image_width, image_height), 
                                (img_center_lon, img_center_lat, posture_info["pos"][2]), 
                                (camera_info["horizontal_fov"], camera_info["vertical_fov"]), 
                                save_path)




if __name__ == "__main__":
    camera_info = {
    "photo_width":5280,
    "photo_height":3956,
    "focal_length":12.29,
    "horizontal_fov":70.82,
    "vertical_fov":56.09,
    "fx":3713.29,
    "fy":3713.29,
    "cx":2647.02,
    "cy":1969.28,
    "k1":-0.11257524,
    "k2":0.01487443,
    "k3":-0.02706411,
    "p1":-0.00008572,
    "p2":1e-7
}
    posture_info =  {
        "pos": [
            40.190980916,
            116.954533119,
            231.237
        ],
        "orientation": [
            -90,
            -89.9,
            0
        ],
        "aircraft_angles": [
            0.0,
            0.0,
            0.0
        ]
    }
    # image = cv2.imread("survey/DJI_images/vis_0403_143804.JPG")
    # reprojection(image, camera_info, posture_info, "geocorrected_vis_0403_143804.tif")



    # fov_coords = calculate_coordinates(camera_info, posture_info)
    # print(fov_coords)

    # import matplotlib.pyplot as plt

    # # 提取经纬度坐标
    # lngs, lats = zip(*fov_coords.values())

    # # 创建图形和轴
    # plt.figure(figsize=(10, 8))
    # plt.plot(lngs, lats, 'o', color='blue')  # 绘制点和连接线
    # plt.scatter([fov_coords["fov_center_lnglat"][0]], [fov_coords["fov_center_lnglat"][1]], color='red')  # 绘制中心点

    # # 标注点
    # for key, (lng, lat) in fov_coords.items():
    #     plt.text(lng, lat, key)

    # # 设置标题和坐标轴标签
    # plt.title("Field of View Coordinates Visualization")
    # plt.xlabel("Longitude")
    # plt.ylabel("Latitude")
    # plt.grid(True)  # 显示网格

    # # 显示图形
    # plt.show()
    