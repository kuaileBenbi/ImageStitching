import math
import pyproj
import numpy as np

from typing import Dict, List


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


def euler_to_rotation_matrix(yaw, pitch, roll):
    # pitch+为抬头，roll+为右旋转，yaw+为右偏航
    # 将角度转换为弧度
    yaw, pitch, roll = map(math.radians, [yaw, pitch, roll])
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

    if not check_pitch(pod_elevation_deg.pitch):
        raise ValueError("Sorry! Pitch angle is out of the calculation range!")

    # 地球参考系
    geod = pyproj.Geod(ellps="WGS84")

    # ==============由载机YPR、吊舱AE决定的旋转矩阵================= #
    R_aircraft = euler_to_rotation_matrix(aircraft_yaw_deg, aircraft_pitch_deg, aircraft_roll_deg)
    R_pod = euler_to_rotation_matrix(pod_azimuth_deg, pod_elevation_deg, 0)
    R_total = R_aircraft @ R_pod

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
                         "fov_bottomright_lnglat": coordinates[4]}
    return fov_lnglat_coords