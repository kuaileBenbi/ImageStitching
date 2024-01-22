from osgeo import gdal, osr
import pyproj
import math
from typing import Tuple
import numpy as np

def calculate_horizontal_coverage(alt: float, horizontal_angle: float, img_width: float) -> float:
    """
    计算每个像素水平覆盖的范围 单位：米

    参数:
        alt: 相机高度
        horizontal_angle: 水平视场角 (单位: 度)
        img_width: 图像的宽度

    返回:
        每个像素水平覆盖的范围 单位：米
    """
    D = 2 * alt * math.tan(math.radians(horizontal_angle) / 2)

    return D / img_width


def calculate_vertical_coverage(alt: float, vertical_angle: float, img_height: float) -> float:
    """
    计算每个像素垂直覆盖的范围 单位：米
    参数:
        alt: 相机高度
        vertical_angle: 垂直视场角 (单位: 度)
        img_height: 图像的高度

    返回:
        每个像素垂直覆盖的范围 单位：米
    """
    D = 2 * alt * math.tan(math.radians(vertical_angle) / 2)

    return D / img_height


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


def compute_GeoTransform(img_center_lon, img_center_lat, alt, horizontal_angle, vertical_angle, img_height, img_width):
    """
    计算gdal的地理校正参数

    参数:
        img_center_lon: 图像中心的经度
        img_center_lat: 图像中心的纬度
        alt: 相机高度
        vertical_angle: 垂直视场角 (单位: 度)
        horizontal_angle: 水平视场角 (单位: 度)
        img_width: 图像的宽度
        img_height: 图像的高度
        
    返回:
        一个包含六个参数的列表:
        1.左上角 X 坐标: 栅格的左上角的 X 坐标。
        2.Pixel Width: 一个像素在 X 方向上的尺寸。
        3.Rotation (about Y-axis): 通常为 0，除非存在旋转。
        4.左上角 Y 坐标: 栅格的左上角的 Y 坐标。
        5.Rotation (about X-axis): 通常为 0，除非存在旋转。
        6.Pixel Height: 一个像素在 Y 方向上的尺寸。
        note: 为统一单位，函数中全部采用经纬度的度作为单位
    """

    pixel_width_deg, pixel_height_deg = compute_pixel_size_ellipsoid([calculate_horizontal_coverage(alt, horizontal_angle, img_width),
                                                                      calculate_vertical_coverage(alt, vertical_angle, img_height)])

    ULx = img_center_lon - (pixel_width_deg * img_width) / 2
    ULy = img_center_lat + (pixel_height_deg * img_height) / 2

    return [ULx, pixel_width_deg, 0, ULy, 0, -pixel_height_deg]


def georeference_image_in_local(corrected_image: np.ndarray,
                                img_center_lon,
                                img_center_lat,
                                camera_alt,
                                horizontal_angle,
                                vertical_angle,
                                output_image_path):
    """
    输入一副原始图像，根据图像中心经纬度进行地理校正，仅保存在本地里

    参数:
        camera_param:
            lng, lat, alt: 相机地理坐标 (经度、纬度、高度)
            yaw: 方位角 (单位: 度)
            pitch: 俯仰角 (单位: 度)
        image_param: 
            vertical_angle: 垂直视场角a (单位: 度)
            horizontal_angle: 水平视场角β (单位: 度)
            img_width: 图像的宽度
            img_height: 图像的高度
        img_center_lon: 图像中心的经度
        img_center_lat: 图像中心的纬度
        input_image_data: 待校正图像
        output_image_path: 校正后图像的保存路径+名字

    返回:
       校正后含有地理信息的gdal outdata保存到本地
    """
    
    img_height, img_width = corrected_image.shape[:2]
    # Open the image using GDAL
    # 创建一个新的GDAL内存数据集
    mem_driver = gdal.GetDriverByName('MEM')
    mem_ds = mem_driver.Create('', img_width, img_height, 1, gdal.GDT_Byte)

    # 将校正后的图像数据复制到内存数据集
    mem_ds.GetRasterBand(1).WriteArray(corrected_image[:,:,0])  # 单波段图像
    
    # Set geotransformation
    geotransform = compute_GeoTransform(img_center_lon,
                                        img_center_lat,
                                        camera_alt,
                                        horizontal_angle,
                                        vertical_angle,
                                        img_height,
                                        img_width)
    mem_ds.SetGeoTransform(geotransform)

    # Set spatial reference
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS('WGS84')
    # note！：进行地理校正用的WGS84（EPSG:4326）坐标系，贴图到地图上要转换投影到 Web Mercator（EPSG:3857）
    mem_ds.SetProjection(srs.ExportToWkt())

    # 创建GeoTIFF的副本
    driver = gdal.GetDriverByName('GTiff')
    driver.CreateCopy(output_image_path, mem_ds, 0)
    
    # 清理
    mem_ds = None

    print("Correct done for local!")


if __name__ == "__main__":
    input_image_path = 'cut_datasets/1/short_date0405_07h22m29s.jpg'
    img_center_lon, img_center_lat = 110.1, 19.3
