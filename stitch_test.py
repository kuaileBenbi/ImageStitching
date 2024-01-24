# import cv2
# import os

# def stitch_images(images_folder, output_path):
#     # 获取文件夹中的所有图像文件
#     image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.JPG', '.jpeg'))]

#     # 读取图像
#     images = [cv2.imread(os.path.join(images_folder, f)) for f in image_files]

#     # 创建Stitcher对象
#     stitcher = cv2.Stitcher_create()

#     # 进行拼接操作
#     status, stitched = stitcher.stitch(images)

#     if status == cv2.Stitcher_OK:
#         # 拼接成功，保存或显示图像
#         cv2.imwrite(output_path, stitched)
#         cv2.imshow('Stitched Image', stitched)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     else:
#         print("Stitching failed: ", status)

# # 使用函数
# stitch_images('/Users/lixiwang/Downloads/dataset-mosaicking/yongzhou-small', 'stitch_image.jpg')
# from osgeo import gdal, gdalconst

# def RasterMosaic(inputfilePath, referencefilefilePath, outputfilePath):
#     print("图像拼接")
#     inputrasfile1 = gdal.Open(inputfilePath, gdal.GA_ReadOnly) # 第一幅影像
#     inputProj1 = inputrasfile1.GetProjection()
#     inputrasfile2 = gdal.Open(referencefilefilePath, gdal.GA_ReadOnly) # 第二幅影像
#     inputProj2 = inputrasfile2.GetProjection()
#     options=gdal.WarpOptions(srcSRS=inputProj1, dstSRS=inputProj1,format='GTiff',resampleAlg=gdalconst.GRA_Bilinear)
#     gdal.Warp(outputfilePath,[inputrasfile1,inputrasfile2],options=options)

# inputfilePath = "corrected"
# referencefilefilePath = "corrected/short_date0405_10h11m01s.tif"
# outputfilePath = "RasterMosaic2.tif"

# RasterMosaic(inputfilePath, referencefilefilePath, outputfilePath)


from osgeo import gdal, gdalconst
import os
import datetime

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



def mosaic_images(images_folder, output_path):
    # 获取文件夹中的所有图像文件
    image_files = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.endswith(('.tif', '.tiff'))]
    # 使用排序函数对文件名进行排序
    sorted_image_files = sorted(image_files, key=sort_by_time)
    print(sorted_image_files)
    inputrasfile1 = gdal.Open(sorted_image_files[0], gdal.GA_ReadOnly)
    inputProj1 = inputrasfile1.GetProjection()
    options=gdal.WarpOptions(srcSRS=inputProj1, dstSRS=inputProj1,format='GTiff',resampleAlg=gdalconst.GRA_Bilinear)

    # 使用gdal.Warp进行镶嵌
    gdal.Warp(output_path, sorted_image_files, options=options)

# 使用函数
# mosaic_images('corrected', 'RasterMosaicAll.tif')
# img = 
