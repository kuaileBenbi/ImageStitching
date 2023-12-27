from osgeo import gdal, gdal_array, osr
import glob


# raster = "datasets/UAVTIRSImages/DJI_0482.jpg"
# srcImage = gdal.Open(raster)
# geoTrans = srcImage.GetGeoTransform()
# print("geoTrans: ", geoTrans)

# raster2 = "results/5/short_date0405_10h11m11s_georef.tif"
# srcImage2 = gdal.Open(raster2)
# geoTrans2 = srcImage2.GetGeoTransform()
# print("geoTrans2: ", geoTrans2)

# raster3 = "final_merged_5_image_1012.tif"
# srcImage3 = gdal.Open(raster3)
# geoTrans3 = srcImage3.GetGeoTransform()
# print("geoTrans3: ", geoTrans3)

# path = "results/1/"
# imgnames = glob.glob(path + "*.tif")
# imgnames.sort()
# for name in imgnames:
#     srcImage = gdal.Open(name)
#     geoTrans = srcImage.GetGeoTransform()
#     print(geoTrans)

# print("done!")

# from osgeo import osr

# source = osr.SpatialReference()
# source.ImportFromEPSG(4326)  # WGS 84

# target = osr.SpatialReference()
# target.ImportFromEPSG(32649)  # UTM 49N

# transform = osr.CoordinateTransformation(source, target)

# center_longitude = 110.52827278999999
# center_latitude = 19.192521065

# ULx, ULy, _ = transform.TransformPoint(center_longitude, center_latitude)

# print(ULx, ULy)

# import math

# def get_utm_zone(longitude):
#     """
#     根据给定的经度计算UTM区带
#     """
#     return int((longitude + 180) / 6) + 1

# def get_utm_epsg(longitude, latitude):
#     """
#     根据给定的经纬度返回相应的UTM EPSG代码
#     """
#     zone = get_utm_zone(longitude)
#     if latitude >= 0:
#         return 32600 + zone
#     else:
#         return 32700 + zone


# def get_geotransform(center_longitude, center_latitude, img_width, img_height, altitude, hfov, vfov):

#     # 定义WGS84坐标系
#     srs = osr.SpatialReference()
#     srs.ImportFromEPSG(4326)
    
#     # 定义投影坐标系（如UTM）
#     utm = osr.SpatialReference()
#     utm_epsg = get_utm_epsg(center_longitude, center_latitude)
#     print("==========================")
#     print("utm_epsg: ", utm_epsg)
#     print("==========================")
#     utm.ImportFromEPSG(utm_epsg)  # 选择一个适合的UTM带，这里使用了33N
    
#     # 创建一个坐标转换对象
#     transform = osr.CoordinateTransformation(srs, utm)
    
#     # 将WGS84坐标转换为UTM坐标
#     utm_x, utm_y, _ = transform.TransformPoint(center_longitude, center_latitude)
    
#     # 根据相机的高度和视场角计算地面上的实际宽度和高度
#     ground_width = 2 * altitude * math.tan(math.radians(hfov) / 2)
#     ground_height = 2 * altitude * math.tan(math.radians(vfov) / 2)
    
#     # 计算每像素代表的实际尺寸
#     pixel_width = ground_width / img_width
#     pixel_height = ground_height / img_height
    
#     # 计算左上角的坐标
#     ULx = utm_x - (ground_width / 2)
#     ULy = utm_y + (ground_height / 2)
    
#     return (ULx, pixel_width, 0, ULy, 0, -pixel_height)




# # 示例
# center_longitude = 110.52827278999999
# center_latitude = 19.192521065
# img_width = 640
# img_height = 512
# altitude = 20000  # 20 km
# hfov = 21.8
# vfov = 17.44

# print(get_geotransform(center_longitude, center_latitude, img_width, img_height, altitude, hfov, vfov))

# import math

# def compute_image_bounds(center_latitude, center_longitude, altitude, vfov, hfov):
#     # 将视场角转换为弧度
#     vfov_rad = math.radians(vfov)
#     hfov_rad = math.radians(hfov)

#     # 使用简单的三角关系计算图像的半宽和半高（在地面上）
#     half_width = altitude * math.tan(hfov_rad / 2.0)
#     half_height = altitude * math.tan(vfov_rad / 2.0)

#     # 使用经纬度和半宽、半高来计算边界
#     # 这里我们使用一个简化的方法，假设经度和纬度都可以直接转换为距离
#     # 实际上，这种转换取决于纬度，但是为了简化我们就这样做了
#     dlon = half_width / (111320 * math.cos(math.radians(center_latitude)))  # 一个经度大约为111320米
#     dlat = half_height / 110540  # 一个纬度大约为110540米

#     # 返回边界
#     return {
#         'min_lon': center_longitude - dlon,
#         'max_lon': center_longitude + dlon,
#         'min_lat': center_latitude - dlat,
#         'max_lat': center_latitude + dlat
#     }

# # 示例使用
# altitude = 20000  # 假设20km
# hfov = 21.8
# vfov = 17.44
# center_latitude = 19.192521065
# center_longitude = 110.52827278999999
# bounds = compute_image_bounds(center_latitude, center_longitude, altitude, vfov, hfov)
# print(bounds)

# from osgeo import gdal, osr
# import math

# def calculate_coverage(distance, resolution, angle):
#     D = 2 * distance * math.tan(math.radians(angle / 2))
#     return D / resolution


# def georeference_image(input_image_path, output_image_path, top_left_coords, bottom_right_coords):
#     # 打开图像文件
    

#     # 设置图像的地理变换参数
#     geotransform = [
#         top_left_coords[0],  # 左上角经度
#         calculate_coverage(20000, 630, 21.8),  # x像元宽度 (经度)
#         0,  # 旋转, 通常为0
#         top_left_coords[1],  # 左上角纬度
#         0,  # 旋转, 通常为0
#         (-1)*calculate_coverage(20000, 502, 17.44)  # y像元高度 (纬度, 通常为负值)
#     ]
#     ds = gdal.Open(input_image_path, gdal.GA_ReadOnly)
#     driver = gdal.GetDriverByName("GTiff")
#     outdata = driver.CreateCopy(output_image_path, ds)

#     # 设置坐标系为WGS84
#     srs = osr.SpatialReference()
#     srs.ImportFromEPSG(4326)
#     outdata.SetProjection(srs.ExportToWkt())
#     outdata.SetGeoTransform(geotransform)
#     print(geotransform)

#     outdata = None
#     ds = None

# 示例用法
# input_image = "cut_datasets/1/short_date0405_07h24m09s.jpg"
# output_image = "07h24m09s.tif"
# top_left = [110.52130946,  19.17117511]
# bottom_right = [110.47435586,  19.24153]
# georeference_image(input_image, output_image, top_left, bottom_right)

# input_image = "cut_datasets/1/short_date0405_07h22m29s.jpg"
# output_image = "07h22m29s.tif"
# top_left = [110.52066658,  19.15820534]
# bottom_right = [110.47132004,  19.22683679]
# georeference_image(input_image, output_image, top_left, bottom_right)


import math
import numpy as np
from osgeo import gdal, osr
from scipy.interpolate import RectBivariateSpline

def deg2rad(deg):
    """Convert degrees to radians."""
    return deg * math.pi / 180.0

def correct_image(image_path, focal_length, sensor_size, optical_center, camera_position, camera_attitude):
    # 读取图像
    dataset = gdal.Open(image_path)
    image = dataset.ReadAsArray()

    # 获取图像大小
    rows, cols = image.shape[1], image.shape[2]

    # 创建空的输出图像
    output = np.zeros((rows, cols, image.shape[0]), dtype=image.dtype)

    # 确定地球半径（WGS84椭球体）
    R = 6378137.0  # 地球赤道半径, 米

    # 转换相机位置的经纬度高度到笛卡尔坐标系
    lat, lon, alt = camera_position
    lat_rad = deg2rad(lat)
    lon_rad = deg2rad(lon)

    # 将相机位置转换为ECEF坐标
    Xc = (R + alt) * math.cos(lat_rad) * math.cos(lon_rad)
    Yc = (R + alt) * math.cos(lat_rad) * math.sin(lon_rad)
    Zc = (R + alt) * math.sin(lat_rad)

    # 设置相机旋转矩阵
    roll, pitch, yaw = map(deg2rad, camera_attitude)

    # 使用欧拉角创建旋转矩阵
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(roll), -math.sin(roll)],
                   [0, math.sin(roll), math.cos(roll)]])
    
    Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                   [0, 1, 0],
                   [-math.sin(pitch), 0, math.cos(pitch)]])
    
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                   [math.sin(yaw), math.cos(yaw), 0],
                   [0, 0, 1]])

    # 组合成一个单一的旋转矩阵
    R = Rz.dot(Ry).dot(Rx)

    # 针孔相机模型的内部参数矩阵
    K = np.array([[focal_length, 0, optical_center[0]],
                  [0, focal_length, optical_center[1]],
                  [0, 0, 1]])

    # 遍历输出图像中的每个像素
    for i in range(rows):
        for j in range(cols):
            # 构建图像空间中的点
            p_img = np.array([j, i, 1])  # 注意这里是列号先，行号后，因为图像坐标系是x先列后行

            # 使用内参矩阵进行逆向投影到相机空间中的点
            p_cam = np.linalg.inv(K).dot(p_img)

            # 将相机空间中的点转换到世界空间
            p_world = R.dot(p_cam) + np.array([Xc, Yc, Zc])

            # 假设地面为水平面，Z=0，求解地面点
            # 因为这里没有DEM数据，我们假设相机高度远大于地物高度变化
            t = -Zc / p_world[2]
            Xg, Yg, Zg = p_world * t

            # 计算输出图像中的坐标
            # 注意：这里并没有考虑地面的实际坐标转换到图像坐标系中的逻辑
            # 这需要更复杂的GIS转换，这里只是一个简化的近似
            new_i, new_j = i, j  # 这需要根据地面点转换逻辑来计算

            # 确保计算出的新坐标在图像范围内
            if 0 <= new_i < rows and 0 <= new_j < cols:
                output[new_i, new_j, :] = image[:, i, j]
    
    # 重采样和内插
    # 使用双三次插值进行重采样
    for band in range(image.shape[0]):
        spline = RectBivariateSpline(np.arange(rows), np.arange(cols), image[band])
        for i in range(rows):
            for j in range(cols):
                output[i, j, band] = spline.ev(i, j)
    
    # 返回校正后的图像数组
    return output


def calculate_gsd(sensor_size, focal_length, altitude, image_width, image_height):
    # 计算水平和垂直的视场角（Field of View, FOV）
    fov_horizontal = 2 * math.atan((sensor_size[0] / 2) / focal_length)
    fov_vertical = 2 * math.atan((sensor_size[1] / 2) / focal_length)
    
    # 计算地面采样距离（Ground Sample Distance, GSD）
    gsd_horizontal = (2 * altitude * math.tan(fov_horizontal / 2)) / image_width
    gsd_vertical = (2 * altitude * math.tan(fov_vertical / 2)) / image_height
    
    return gsd_horizontal, gsd_vertical


