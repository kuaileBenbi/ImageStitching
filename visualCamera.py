# 
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np

# def llh_to_xyz(lat, lon, alt):
#     """将经纬高坐标转换为笛卡尔坐标系中的点"""
#     # 假设地球是一个完美的球体
#     # 这里仅为示例，实际应用中可能需要更精确的模型
#     R = 6371 + alt  # 地球半径加上高度
#     x = R * np.cos(lat) * np.cos(lon)
#     y = R * np.cos(lat) * np.sin(lon)
#     z = R * np.sin(lat)
#     return x, y, z

# def camera_cone(camera_pos, pitch, yaw, length=1, cone_length=0.2, cone_radius=0.05):
#     """创建表示相机的锥体"""
#     dir_x = np.cos(pitch) * np.cos(yaw)
#     dir_y = np.cos(pitch) * np.sin(yaw)
#     dir_z = np.sin(pitch)
#     return camera_pos, camera_pos + length * np.array([dir_x, dir_y, dir_z]), cone_length, cone_radius

# # 相机的经纬高和姿态
# lat, lon, alt = np.radians(40.7128), np.radians(-74.0060), 0.1  # 纽约的经纬度和示例高度
# pitch, yaw = np.radians(45), np.radians(45)  # 45度俯仰角和45度方位角

# # 转换为笛卡尔坐标
# camera_position = np.array(llh_to_xyz(lat, lon, alt))

# # 创建 3D 图
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 绘制相机位置
# ax.scatter(camera_position[0], camera_position[1], camera_position[2], c='r', marker='o')

# # 获取相机锥体的参数
# camera_pos, camera_tip, cone_length, cone_radius = camera_cone(camera_position, pitch, yaw)

# # 绘制相机方向
# ax.quiver(camera_pos[0], camera_pos[1], camera_pos[2],
#           camera_tip[0], camera_tip[1], camera_tip[2],
#           length=cone_length, normalize=True)

# # 绘制相机锥体
# cone_base = camera_tip - cone_length * np.array([camera_tip[0], camera_tip[1], camera_tip[2]])
# ax.plot([camera_pos[0], cone_base[0]], [camera_pos[1], cone_base[1]], [camera_pos[2], cone_base[2]], 'r-')

# # 设置坐标轴标签
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Z axis')

# # 显示图形
# plt.show()

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np

# def llh_to_xyz(lat, lon, alt):
#     """将经纬高坐标转换为笛卡尔坐标系中的点"""
#     R = 6371 + alt  # 地球半径加上高度
#     x = R * np.cos(lat) * np.cos(lon)
#     y = R * np.cos(lat) * np.sin(lon)
#     z = R * np.sin(lat)
#     return x, y, z

# def calculate_fov(focal_length, sensor_size):
#     """计算相机的视场角"""
#     return 2 * np.arctan(sensor_size / (2 * focal_length))

# def draw_camera_fov(ax, camera_pos, pitch, yaw, fov_h, fov_v, length=1):
#     """在3D图中绘制相机的视场"""
#     # 计算视场方向
#     dir_x = np.cos(pitch) * np.cos(yaw)
#     dir_y = np.cos(pitch) * np.sin(yaw)
#     dir_z = np.sin(pitch)

#     # 绘制视场边界
#     half_fov_h = fov_h / 2
#     half_fov_v = fov_v / 2

#     # 四个角点的方向
#     directions = []
#     for h in [-half_fov_h, half_fov_h]:
#         for v in [-half_fov_v, half_fov_v]:
#             dx = np.cos(pitch + v) * np.cos(yaw + h)
#             dy = np.cos(pitch + v) * np.sin(yaw + h)
#             dz = np.sin(pitch + v)
#             directions.append((dx, dy, dz))

#     # 绘制视场线
#     for dx, dy, dz in directions:
#         ax.plot([camera_pos[0], camera_pos[0] + length * dx],
#                 [camera_pos[1], camera_pos[1] + length * dy],
#                 [camera_pos[2], camera_pos[2] + length * dz], color='r', linestyle='--')

# # 相机参数
# focal_length = 40  # 焦距，单位毫米
# sensor_size = (16, 12.8)   # 探测器尺寸，单位毫米

# # 相机的经纬高和姿态
# lat, lon, alt = np.radians(19.19247651), np.radians(110.49602631), 19.038
# pitch, yaw = np.radians(-90), np.radians(0)

# # 转换为笛卡尔坐标
# camera_position = np.array(llh_to_xyz(lat, lon, alt))

# # 计算相机的水平和垂直视场角
# fov_h = calculate_fov(focal_length, sensor_size[0])
# fov_v = calculate_fov(focal_length, sensor_size[1])

# # 创建 3D 图
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 绘制相机位置
# ax.scatter(camera_position[0], camera_position[1], camera_position[2], c='r', marker='o')

# # 绘制相机视场
# draw_camera_fov(ax, camera_position, pitch, yaw, fov_h, fov_v)

# # 设置坐标轴标签
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Z axis')

# # 显示图形
# plt.show()


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def llh_to_xyz(lat, lon, alt):
    """将经纬高坐标转换为笛卡尔坐标系中的点"""
    R = 6371 + alt  # 地球半径加上高度
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return x, y, z

def rotation_matrix(roll, pitch, yaw):
    """计算由横滚（roll）、俯仰（pitch）和航向（yaw）角定义的旋转矩阵"""
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx

def draw_camera_fov(ax, camera_pos, rotation_mat, fov_h, fov_v, length=1):
    """在3D图中绘制相机的视场，考虑旋转矩阵"""
    # 定义相机朝向的初始方向（假设相机初始朝向是Z轴正方向）
    direction = np.array([0, 0, 1])

    # 应用旋转矩阵以获取正确的相机方向
    direction = rotation_mat @ direction

    # ...（此处省略了视场绘制的其余部分，与之前相同）

# 相机参数
focal_length = 40  # 焦距，单位毫米
sensor_size = (16, 12.8)  # 传感器尺寸，单位毫米（长，宽）

# 相机的经纬高
lat, lon, alt = np.radians(19.19247651), np.radians(110.49602631), 19.038

# 浮空平台的横滚、俯仰和航向角
roll, pitch, yaw = np.radians(3.93), np.radians(-2.69), np.radians(15.64)  # 示例角度

# 计算旋转矩阵
rotation_mat = rotation_matrix(roll, pitch, yaw)

# 转换为笛卡尔坐标
camera_position = np.array(llh_to_xyz(lat, lon, alt))

# 计算相机的视场角
fov_h = 2 * np.arctan2(sensor_size[0] / 2, focal_length)
fov_v = 2 * np.arctan2(sensor_size[1] / 2, focal_length)

# 创建 3D 图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制相机位置
ax.scatter(camera_position[0], camera_position[1], camera_position[2], c='r', marker='o')

# 绘制相机视场
draw_camera_fov(ax, camera_position, rotation_mat, fov_h, fov_v)

# 设置坐标轴标签
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# 显示图形
plt.show()
