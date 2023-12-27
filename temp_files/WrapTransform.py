import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import radians, cos, sin, sqrt, atan, tan

# 构建平移矩阵
def lla_to_translation_matrix(lon, lat, alt):
    # WGS 84 ellipsiod constants
    a = 6378137
    e = 8.1819190842622e-2

    # 转换为弧度
    lon_rad = radians(lon)
    lat_rad = radians(lat)

    # 辅助值
    N = a / sqrt(1 - e**2 * sin(lat_rad)**2)

    # 计算ECEF坐标
    x = (N + alt) * cos(lat_rad) * cos(lon_rad)
    y = (N + alt) * cos(lat_rad) * sin(lon_rad)
    z = ((1 - e**2) * N + alt) * sin(lat_rad)

    return np.array([x, y, z]).reshape(3, 1)


# 构建旋转矩阵
def angles_to_rotation_matrix(roll, pitch, yaw):
    roll_rad = radians(roll)
    pitch_rad = radians(pitch)
    yaw_rad = radians(yaw)

    R_x = np.array([[1, 0, 0],
                    [0, cos(roll_rad), -sin(roll_rad)],
                    [0, sin(roll_rad), cos(roll_rad)]])

    R_y = np.array([[cos(pitch_rad), 0, sin(pitch_rad)],
                    [0, 1, 0],
                    [-sin(pitch_rad), 0, cos(pitch_rad)]])

    R_z = np.array([[cos(yaw_rad), -sin(yaw_rad), 0],
                    [sin(yaw_rad), cos(yaw_rad), 0],
                    [0, 0, 1]])

    # 组合旋转矩阵
    R = R_z @ R_y @ R_x
    return R

# 构建内参矩阵
def cameraparams_to_intrinsic_matrix(focal_length, sensor_size, image_size):
    # 将焦距从毫米转换为像素单位
    focal_length_px = (focal_length / sensor_size[0]) * image_size[0]
    focal_length_py = (focal_length / sensor_size[1]) * image_size[1]

    # 构建相机内参矩阵
    K = np.array([[focal_length_px, 0, image_size[0] / 2],
                [0, focal_length_py, image_size[1] / 2],
                [0, 0, 1]])
    return K


# 加载上传的斜视图像
# image_path = 'datasets/1/short_date0405_07h22m29s.jpg'
# oblique_image = cv2.imread(image_path)

# 设定相机参数
camera_position = (110.49602631, 19.19247651, 19038)  # 经度，纬度，高度
camera_angles = (-45, 15, 3)  # 俯仰角182.69，方位角15.64, 滚转角3.93
image_size = (640, 512)  # 图像宽度，图像高度
focal_length_mm = 40  # 焦距，毫米
sensor_size_mm = (16, 12.8) # 传感器尺寸, mm

# 视场角
# h_half_fov = atan(sensor_size_mm[0]/(2*focal_length_mm))
# v_half_fov = atan(sensor_size_mm[1]/(2*focal_length_mm))

# GSD-m
# h_gsd = 2*camera_position[2]*tan(h_half_fov)/image_size[0]
# v_gsd = 2*camera_position[2]*tan(v_half_fov)/image_size[1]
# print(f"h_gsd: {h_gsd}, v_gsd: {v_gsd}")

# 内参矩阵
# K = cameraparams_to_intrinsic_matrix(focal_length_mm, sensor_size_mm, image_size)
# K = np.hstack((K, np.transpose(np.array([[0, 0, 1]]))))
# 旋转矩阵
# R = np.transpose(angles_to_rotation_matrix(*camera_angles))
# T = lla_to_translation_matrix(*camera_position)
# T = np.array([camera_position]).reshape(3, 1)
# RT = np.hstack((R, T))
# RT = np.vstack((RT, np.array([0,0,0,1])))
# C = lla_to_translation_matrix(*camera_position)
# C = np.array([camera_position]).reshape(3, 1)
# t = - np.dot(R, C)
# Rt = np.hstack((R, t))
# Rt = np.vstack((Rt, np.array([[0, 0, 0, 1]])))

# inv_K = np.linalg.inv(K)
# inv_K = np.hstack((inv_K, np.transpose(np.array([[0, 0, 1]]))))
# inv_Rt = np.linalg.inv(Rt)
# P = np.dot(K, Rt)
# inv_P = np.linalg.pinv(P)

# P = np.dot(K, RT)
# P = np.delete(P, 2, axis=1)
# oblique_uv = np.zeros(image_size)

# 计算映射
# for y in range(image_size[1]):
#     for x in range(image_size[0]):
#         point = np.array([x, y, 1])
#         normal_point = np.dot(P, point)
#         oblique_uv[x, y] = normal_point[0], normal_point[1]
# normal_00 = np.dot(inv_P, np.array([0, 0, 1]))
# normal_10 = np.dot(inv_P, np.array([639, 0, 1]))
# normal_11 = np.dot(inv_P, np.array([639, 511, 1]))
# normal_01 = np.dot(inv_P, np.array([0, 511, 1]))
# print("校正后图像的四点坐标：", normal_00/normal_00[2], normal_01/normal_01[2], normal_11/normal_11[2], normal_10/normal_10[2])

# xmin = min(normal_00[0], normal_10[0], normal_11[0], normal_01[0])
# xmax = max(normal_00[0], normal_10[0], normal_11[0], normal_01[0])

# ymin = min(normal_00[1], normal_10[1], normal_11[1], normal_01[1])
# ymax = max(normal_00[1], normal_10[1], normal_11[1], normal_01[1])
# print(f"xmin: {xmin}, xmax: {xmax}, ymin: {ymin}, ymax: {ymax}")

# 输出校正图像的总行数、总列数
# correct_col = int((xmax - xmin)/h_gsd) + 1
# correct_row = int((ymax - ymin)/v_gsd) + 1
# print(f"correct_col: {correct_col}, correct_row: {correct_row}")

# pts1 = np.float32([[-3.0743722, -11.4300523], [-79.80502841, 21.6873707], [0.03331912, 0.076485951]])
# pts2 = np.float32([[0, 0], [639, 0], [0, 511]])

# M = cv2.getAffineTransform(pts2, pts1)
# dst = cv2.warpAffine(oblique_image, M, (640, 512))

# cv2.imshow('corrected_image', dst)
# # cv2.imshow('src_image', oblique_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import math

def transform_to_map_coordinates(u, v, P):
    # 将图像坐标转换为齐次坐标
    u_v_1 = np.array([u, v, 1])

    # 使用逆投影矩阵，从图像坐标到地面坐标
    world_coords = np.linalg.pinv(P) @ u_v_1
    world_coords /= world_coords[2]

    # 返回地面坐标（x, y）
    return world_coords[0], world_coords[1]

def transform_to_image_coordinates(x_m, y_m, P):
    # 将地面坐标转换为齐次坐标
    xy_1 = np.array([x_m, y_m, 0, 1])

    # 使用投影矩阵，从地面坐标到图像坐标
    image_coords = P @ xy_1
    image_coords /= image_coords[2]

    # 返回图像坐标（u, v）
    return image_coords[0], image_coords[1]

# 将角度从度转换为弧度
def degrees_to_radians(degrees):
    return degrees * math.pi / 180

# 计算旋转矩阵
def calculate_rotation_matrix(angles):
    pitch, yaw, roll = [degrees_to_radians(angle) for angle in angles]

    R_x = np.array([[1, 0, 0],
                    [0, math.cos(roll), -math.sin(roll)],
                    [0, math.sin(roll), math.cos(roll)]])

    R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                    [0, 1, 0],
                    [-math.sin(pitch), 0, math.cos(pitch)]])

    R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                    [math.sin(yaw), math.cos(yaw), 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

# 计算内参矩阵
def calculate_intrinsic_matrix(focal_length, sensor_size, image_size):
    fx = focal_length / sensor_size[0] * image_size[0]
    fy = focal_length / sensor_size[1] * image_size[1]
    cx = image_size[0] / 2
    cy = image_size[1] / 2
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    return K

# 计算外参矩阵
def calculate_extrinsic_matrix(rotation_matrix, camera_position):
    t = -rotation_matrix @ np.array(camera_position[:3]).reshape((3, 1))
    return np.hstack((rotation_matrix, t))

# 计算投影矩阵
K = calculate_intrinsic_matrix(focal_length_mm, sensor_size_mm, image_size)
R = calculate_rotation_matrix(camera_angles)
R = np.transpose(R)
_camera_position = lla_to_translation_matrix(*camera_position)
T = calculate_extrinsic_matrix(R, _camera_position)
P = K @ T

# 示例：转换一个点的坐标
u, v = 320, 256  # 图像中的一个点
x_m, y_m = transform_to_map_coordinates(u, v, P)
u_transformed, v_transformed = transform_to_image_coordinates(x_m, y_m, P)

print("原始图像坐标:", u, v)
print("转换后的地面坐标:", x_m, y_m)
print("从地面坐标转换回的图像坐标:", u_transformed, v_transformed)
