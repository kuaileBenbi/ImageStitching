from osgeo import gdal, osr
from typing import Tuple


def correct_skewed_image(input_file: str,
                         output_file: str,
                         top_left: Tuple[float, float],
                         top_right: Tuple[float, float],
                         bottom_right: Tuple[float, float],
                         bottom_left: Tuple[float, float]) -> None:
    """
    根据四角地理坐标校正斜视图像, 示例中视场四角坐标由EstimateFovLngLat.py计算得到

    参数:
    input_file (str): 原始图像的路径。
    output_file (str): 校正后的图像的输出路径。
    top_left, top_right, bottom_right, bottom_left (tuple): 四角的地理坐标（经度，纬度）。
    """

    # 读取原始图像
    dataset = gdal.Open(input_file, gdal.GA_ReadOnly)
    width = dataset.RasterXSize
    height = dataset.RasterYSize

    # 创建输出图像的地理参考
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(4326)

    # 使用 GCP（地面控制点）
    gcps = [
        gdal.GCP(top_left[0], top_left[1], 0, 0, 0),
        gdal.GCP(top_right[0], top_right[1], 0, width, 0),
        gdal.GCP(bottom_right[0], bottom_right[1], 0, width, height),
        gdal.GCP(bottom_left[0], bottom_left[1], 0, 0, height)
    ]

    # 设置地面控制点和投影
    dataset.SetGCPs(gcps, dst_srs.ExportToWkt())

    # 执行校正
    gdal.Warp(output_file, dataset, format='GTiff', dstSRS=dst_srs)


if __name__ == "__main__":
    correct_skewed_image(
        'datasets/1/short_date0405_07h22m29s.jpg',
        'corrected_image.tif',
        (110.49067675230712, 19.247178659498466),  # top_left
        (110.51525358758893, 19.240641030602593),  # top_right
        (110.48933892610636, 19.191363600353696),  # bottom_right
        (110.4858839411186, 19.192282635697328)   # bottom_left
    )
