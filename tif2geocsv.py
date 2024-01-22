"""
把Geo图片的包含的经纬度信息单独提取出来存到csv文件中
"""
import os
import csv
import datetime
from osgeo import gdal

# 文件夹路径，包含所有图像文件
image_folder = "corrected"

# 创建一个 CSV 文件用于存储坐标数据
csv_filename = "image_coordinates.csv"

# 获取文件夹中所有的图像文件名
image_files = [image_filename for image_filename in os.listdir(image_folder) if image_filename.endswith('.tif')]

# 定义一个排序函数，提取时间信息并排序
def sort_by_time(filename):
    # 从文件名中提取时间信息，这里假设文件名的格式是 "short_dateMMDD_hh'h'mm's'.tif"
    parts = filename.split('_')
    time_part = parts[-1].split('.')[0]
    time_format = "%Hh%Mm%Ss"
    try:
        time_obj = datetime.datetime.strptime(time_part, time_format)
    except ValueError:
        # 如果解析失败，返回一个默认时间
        time_obj = datetime.datetime(1900, 1, 1, 0, 0, 0)
    return time_obj

# 使用排序函数对文件名进行排序
sorted_image_files = sorted(image_files, key=sort_by_time)

# 打开 CSV 文件以写入数据
with open(csv_filename, mode='w', newline='') as csvfile:
    fieldnames = ['Image Name', 'Left Top Longitude', 'Left Top Latitude', 'Right Bottom Longitude', 'Right Bottom Latitude']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # 写入 CSV 文件的表头
    writer.writeheader()
    
    # 遍历文件夹中的图像文件
    for image_filename in sorted_image_files:
        if image_filename.endswith('.tif'):
            # 打开图像文件
            image_path = os.path.join(image_folder, image_filename)
            srcImage = gdal.Open(image_path)
            
            if srcImage is not None:
                # 获取地理变换信息
                geoTrans = srcImage.GetGeoTransform()
                
                if geoTrans is not None:
                    # 获取图像的宽度和高度
                    width = srcImage.RasterXSize
                    height = srcImage.RasterYSize

                    # 计算左上和右下的经纬度坐标
                    lon_left_top = geoTrans[0]
                    lat_left_top = geoTrans[3]
                    lon_right_bottom = geoTrans[0] + geoTrans[1] * width
                    lat_right_bottom = geoTrans[3] + geoTrans[5] * height
                    
                    # 写入 CSV 文件
                    writer.writerow({'Image Name': image_filename, 
                                     'Left Top Longitude': lon_left_top, 
                                     'Left Top Latitude': lat_left_top, 
                                     'Right Bottom Longitude': lon_right_bottom, 
                                     'Right Bottom Latitude': lat_right_bottom})
    
    print(f"Coordinates saved to {csv_filename}")
