import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
# from osgeo import gdal, osr
import pyproj
import math

from matplotlib import pyplot as plt

"""
基于针孔相机模型的几何校正：
step1: 根据传感器参数-焦距、探测器尺寸、图像尺寸，计算内参矩阵K
step2: 根据相机和载机的姿态、位置，计算外参矩阵P=[R|t]（位置用转换到笛卡尔坐标系）
step3: 根据像素坐标系-相机坐标系、相机坐标系-世界坐标系关系，利用内外参数矩阵计算出畸变图像中四个角点的像素坐标对应的世界坐标
step4: 对世界坐标进行平移、缩放变换使其范围与原始图像分辨率一致，同时保证长宽比、长宽关系
step5: 利用opencv的cv2.findHomography求出投影变换矩阵，cv2.warpProjective进行投影变换得到校正后的图像
"""

DEBUG = False

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
    # pitch+为抬头，roll+为右旋转，yaw+为右偏航
    # 将角度转换为弧度
    yaw, pitch, roll = np.deg2rad(yaw),np.deg2rad(pitch),np.deg2rad(roll)
    
    # 定义各轴旋转矩阵
    Rz_yaw = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    Ry_pitch = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ], dtype=np.float32)

    Rx_roll = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
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


def define_initial_vectors(horizontal_fov, vertical_fov):
    """
    计算没有姿态角时相机光轴中心、视场四个角点的方向向量。

    机体坐标系：
    以无人机的重心为原点，无人机机头前进的方向为X轴，机头前进方向的右侧为Y轴，Z轴与X轴、Y轴相互垂直交于重心且指向无人机下方
    横滚（无人机仅绕X轴旋转）、俯仰（无人机仅绕Y轴旋转）和偏航（无人机仅绕Z轴旋转）
    ==>假设初始时相机光轴中心方向向量[1,0,0], top_left向量在x轴正半轴、y轴负半轴、z轴负半轴
    大地坐标系：“北-东-地（N-E-D）坐标系”
    无人机指向地球正北的方向为X轴，正东的方向为Y轴，X轴与Y轴相互垂直，Z轴竖直指向无人机下方

    """
    # 将角度转换为弧度
    h_fov_half = np.radians(horizontal_fov / 2)
    v_fov_half = np.radians(vertical_fov / 2)

    # 四个角点的方向向量
    top_left = np.array([
        math.cos(-v_fov_half) * math.cos(-h_fov_half),
        math.cos(-v_fov_half) * math.sin(-h_fov_half),
        -math.sin(-v_fov_half)
    ])
    top_right = np.array([
        math.cos(-v_fov_half) * math.cos(h_fov_half),
        math.cos(-v_fov_half) * math.sin(h_fov_half),
        -math.sin(-v_fov_half)
    ])
    bottom_left = np.array([
        math.cos(v_fov_half) * math.cos(-h_fov_half),
        math.cos(v_fov_half) * math.sin(-h_fov_half),
        -math.sin(v_fov_half)
    ])
    bottom_right = np.array([
        math.cos(v_fov_half) * math.cos(h_fov_half),
        math.cos(v_fov_half) * math.sin(h_fov_half),
        -math.sin(v_fov_half)
    ])
    vectors = {
        "center":np.array([1,0,0]),
        "topleft":top_left,
        "topright":top_right,
        "bottomleft":bottom_left,
        "bottomright":bottom_right
    }

    return vectors


def rotate_vectors(rotation_mat, vectors):
    """
    机体坐标系：
    以无人机的重心为原点，无人机机头前进的方向为X轴，机头前进方向的右侧为Y轴，Z轴与X轴、Y轴相互垂直交于重心且指向无人机下方
    横滚（无人机仅绕X轴旋转）、俯仰（无人机仅绕Y轴旋转）和偏航（无人机仅绕Z轴旋转）
    pitch+为抬头，roll+为右旋转，yaw+为右偏航

    """
    center_vector = rotation_mat @ vectors["center"]
    top_left = rotation_mat @ vectors["topleft"]
    top_right = rotation_mat @ vectors["topright"]
    bottom_left = rotation_mat @ vectors["bottomleft"]
    bottom_right = rotation_mat @ vectors["bottomright"]

    rotated_vectors = {
    "center":center_vector,
    "topleft":top_left,
    "topright":top_right,
    "bottomleft":bottom_left,
    "bottomright":bottom_right}

    return rotated_vectors


def calculate_azimuth_elevation(vector):
    """
    计算向量与水平面的夹角elevation, 用来计算视场向量与地球表面交点 到 相机在地球表面投影点的距离
    计算向量与垂直面的夹角azimuth, 用来计算视场向量与地球表面交点 到 相机在地球表面投影点的方位
    """
    azimuth = math.atan2(vector[1], vector[0])
    horizontal_distance = math.sqrt(vector[0]**2 + vector[1]**2)
    elevation = math.atan2(vector[2], horizontal_distance)
    return math.degrees(azimuth), math.degrees(elevation)


def calculate_distance_to_ground(camera_altitude, elevation_angle):  
    """
    计算视场向量与地球表面交点 到 相机在地球表面投影点的距离
    """
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
    
    # ==============由载机YPR、吊舱AE决定的旋转矩阵================= #
    R_aircraft = euler_to_rotation_matrix(aircraft_yaw_deg, aircraft_pitch_deg, aircraft_roll_deg)
    R_pod = euler_to_rotation_matrix(pod_azimuth_deg, pod_elevation_deg, 0)
    R_total = R_aircraft @ R_pod

    geod = pyproj.Geod(ellps="WGS84")
    # ==============视场中心、四个角方向向量================= #
    vectors = define_initial_vectors(horizontal_fov, vertical_fov)
    
    # ==============旋转后的视场中心、四个角方向向量================= #
    rotated_vectors = rotate_vectors(R_total, vectors)

    # ==============旋转后的视场中心、四个角方向向量与地面交点的经纬度================= #
    coordinates = []
    for _, rotated_vector in rotated_vectors.items():
        azimuth, elevation = calculate_azimuth_elevation(rotated_vector)
        distance = calculate_distance_to_ground(camera_alatitude, elevation)
        # 计算经纬度
        new_lng, new_lat, _ = geod.fwd(camera_lng, camera_lat, azimuth, distance)
        coordinates.append((new_lng, new_lat))

    fov_lnglat_coords = {"fov_center_lnglat": coordinates[0],
                         "fov_topleft_lnglat": coordinates[1],
                         "fov_topright_lnglat": coordinates[2],
                         "fov_bottomleft_lnglat": coordinates[3],
                         "fov_bottomright_lnglat": coordinates[4]
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
    image_width, image_height = image.shape[1], image.shape[0]

    # 图像四角的像素坐 (齐次坐标)
    corner_pixel_coordinates = np.array([[0, 0, 1],
                                         [image_width - 1, 0, 1],
                                         [image_width - 1, image_height - 1, 1],
                                         [0, image_height - 1, 1]], dtype="float32")
    print(f"理想P-uv: {corner_pixel_coordinates}")

    # 内参矩阵
    k = camera_info["intrinsic_matrix"]
    # 畸变系数，如果不为零
    distortion_k12p12k3 = camera_info["distortion_k12p12k3"]

    # fix
    posture_info["orientation"][1] += 90

    # 外参矩阵
    R = external_matrix(pod_azimuth_deg=posture_info["orientation"][0],
                        pod_elevation_deg=posture_info["orientation"][1],
                        aircraft_yaw_deg=posture_info["aircraft_angles"][0],
                        aircraft_pitch_deg=posture_info["aircraft_angles"][1],
                        aircraft_roll_deg=posture_info["aircraft_angles"][2])

    # ============================== 计算相机坐标 ==================================== #
    # 假如没有旋转姿态 求对应的相机坐标
    corner_cam_coordinates = np.zeros_like(corner_pixel_coordinates)
    for i, crd_pixel in enumerate(corner_pixel_coordinates):
        crd_cam = np.linalg.pinv(k) @ crd_pixel
        # crd_cam /= crd_cam[2]
        corner_cam_coordinates[i] = crd_cam
    
    print(f"由理想理想P-uv计算得到P-cam: {corner_cam_coordinates}")

    show_2d(corner_cam_coordinates, 'cam coord')
    show_3d(corner_cam_coordinates, 'cam coord')
   
    # ============================== 计算世界坐标 ==================================== #
    # 假设没有姿态旋转：pcam=pw
    corner_world_coordinates = corner_cam_coordinates
    print(f"在没有姿态角时, 旋转矩阵R-0=单位矩阵, 由此推理得世界坐标系下该点坐标P-w为: {corner_world_coordinates}")
    # 当有旋转时：p`cam = R·pw
    corner_cam_coordinates_rotated = np.zeros_like(corner_world_coordinates) 
    for i, crd_world in enumerate(corner_world_coordinates):
        crd_cam_rotated = R @ crd_world
        # crd_world_rotated /= crd_world_rotated[2]
        corner_cam_coordinates_rotated[i] = crd_cam_rotated
    print(f"当发生姿态旋转时, 旋转矩阵为R={R}, 此时相机坐标系下该点坐标P`-cam为: {corner_cam_coordinates_rotated}")
    show_2d(corner_cam_coordinates_rotated, 'world coord')
    show_3d(corner_cam_coordinates_rotated, 'world coord')

    show_3d_drift(corner_cam_coordinates, corner_cam_coordinates_rotated, 'cam2world')
    
    # ============================== 反推像素坐标 ==================================== #
    corner_pixel_coordinates_raotatd = np.zeros_like(corner_cam_coordinates)
    for i, crd_cam_rotated in enumerate(corner_cam_coordinates_rotated):
        crd_pixel_rotated = k @ crd_cam_rotated
        # crd_pixel_rotated /= crd_pixel_rotated[2]
        corner_pixel_coordinates_raotatd[i] = crd_pixel_rotated
    print(f"由相机坐标系反推回该点的真实像素坐标P`-uv为: {corner_pixel_coordinates_raotatd}")
    
    corner_pixel_coordinates_raotatd
    # normalized_world_coords = normalize_coordinates(corner_world_coordinates, (image_width, image_height))

    # ==============================计算单应性矩阵==================================#
    H, _ = cv2.findHomography(corner_pixel_coordinates[:, 0:2], corner_pixel_coordinates_raotatd[:, 0:2])
    # print(f"单应性矩阵H: {H}")

    # =============================畸变校正，如果有====================================#
    distortion_coeffs = (lambda lst: [x for x in lst if x != 0])(distortion_k12p12k3)
    # print(f"distortion_coeffs: {np.array(distortion_coeffs)}") 
    if len(distortion_coeffs) > 0:
        image = cv2.undistort(image, k, np.array(distortion_coeffs))
    # cv2.imshow("undistort_image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # =============================对畸变校正后的图像应用单应性变换====================================#
    corrected_image = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]))
    cv2.imwrite(save_path, corrected_image)
    # cv2.imshow("corrected_image", corrected_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # ========================== 对几何校正后的图像编码地理信息并保存====================================#
    # fov_lnglat_coords = calculate_coordinates(camera_info, posture_info)
    # img_center_lon, img_center_lat = fov_lnglat_coords["fov_center_lnglat"]
    # georeference_image_in_local(corrected_image, 
    #                             (image_width, image_height), 
    #                             (img_center_lon, img_center_lat, posture_info["pos"][2]), 
    #                             (camera_info["horizontal_fov"], camera_info["vertical_fov"]), 
    #                             save_path)

def show_2d(points, title='Default'):
    if not DEBUG:
        return
    # 创建图形和轴
    plt.figure(figsize=(10, 8))
    colors = ('blue', 'red', 'orange', 'pink')
    texts = ('A', 'B', 'C', 'D')
    for i, point in enumerate(points):
        plt.plot(point[0], point[1], 'o', color=colors[i])  # 绘制点和连接线
        plt.text(point[0], point[1], texts[i])

    # 设置标题和坐标轴标签
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)  # 显示网格

    # 显示图形
    plt.show()

def show_3d(points, title='Default'):
    if not DEBUG:
        return
    # 假设的四个坐标点数据
    colors = ('blue', 'red', 'orange', 'pink')
    texts = ('A', 'B', 'C', 'D')


    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制散点图并标注
    for i, (x, y, z) in enumerate(points):
        ax.scatter(x, y, z, color=colors[i], s=100)  # s是点的大小
        ax.text(x, y, z, '%s' % (texts[i]), size=20, zorder=1, color='k')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title(title)
    plt.show()

def show_3d_drift(points, new_points, title='Default'):
    if not DEBUG:
        return
    # 初始化颜色和文本
    color = 'blue'  # 使用同一种颜色标记所有点
    texts = ('A', 'B', 'C', 'D')

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制原始点并标注
    for i, (x, y, z) in enumerate(points):
        ax.scatter(x, y, z, color=color, s=100, alpha=0.5)  # 原始点半透明
        ax.text(x, y, z, '%s' % (texts[i]), size=20, zorder=1, color='k')

    # 绘制新的点并标注
    for i, (x, y, z) in enumerate(new_points):
        ax.scatter(x, y, z, color=color, s=100)  # 新点不透明
        ax.text(x, y, z, '%s\'' % (texts[i]), size=20, zorder=1, color='k')  # 使用带撇的文本标注新位置

    # 绘制移动方向的线
    for (x_old, y_old, z_old), (x_new, y_new, z_new) in zip(points, new_points):
        ax.plot([x_old, x_new], [y_old, y_new], [z_old, z_new], color='gray', linestyle='--')  # 灰色虚线表示移动方向

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title(title)
    plt.show()

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
            0,
            -90,
            0
        ],
        "aircraft_angles": [
            0.0,
            0.0,
            0.0
        ]
    }

    fov_coords = calculate_coordinates(camera_info, posture_info)
    print(fov_coords)

    import matplotlib.pyplot as plt

    # 提取经纬度坐标
    lngs, lats = zip(*fov_coords.values())

    # 创建图形和轴
    plt.figure(figsize=(10, 8))
    plt.plot(lngs, lats, 'o', color='blue')  # 绘制点和连接线
    plt.scatter([fov_coords["fov_center_lnglat"][0]], [fov_coords["fov_center_lnglat"][1]], color='red')  # 绘制中心点

    # 标注点
    for key, (lng, lat) in fov_coords.items():
        plt.text(lng, lat, key)

    # 设置标题和坐标轴标签
    plt.title("Field of View Coordinates Visualization")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)  # 显示网格

    # 显示图形
    plt.show()
    