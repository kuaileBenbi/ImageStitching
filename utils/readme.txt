README
======

This README provides an overview of the Python scripts contained in this folder and their respective functionalities.

SCRIPTS
-------

1. config.py
   - Description: This script is responsible for initializing all the parameters required during the calculation.

2. EstimateImgLngLat.py
   - Description: This script is used for estimating the longitude and the latitude of the center of the image and the upper and lower left corners.
   - Usage: from EstimateImgLngLat import calculate_coordinates
            camera_params = CameraParams((110.4960263, 19.1924765, 19038), 15.64, -2.69)
            image_params = ImageParams( 17.44, 21.8, (640, 512))
            coords_center, coords_top_left, coords_bottom_right = calculate_coordinates(camera_params, image_params)

   - Input: CameraParams = namedtuple('Camera', ['coords', 'yaw', 'pitch'])
            ImageParams = namedtuple('ImageParams', ['vertical_angle', 'horizontal_angle', 'size']).
   - Output: center_coords, top_left_coords, bottom_right_coords

3. GeometricCorrect_gdal.py
   - Description: This script is used for geometric correction and relies heavily on the GDAL library.
   - usage: from GeometricCorrect_gdal import reprojection
            correct_skewed_image(
                                 'datasets/1/short_date0405_07h22m29s.jpg',
                                 'corrected_image.tif',
                                 (110.49067675230712, 19.247178659498466),  # top_left
                                 (110.51525358758893, 19.240641030602593),  # top_right
                                 (110.48933892610636, 19.191363600353696),  # bottom_right
                                 (110.4858839411186, 19.192282635697328)   # bottom_left
                              )

   - Input: input_file (str): 原始图像的路径。
            output_file (str): 校正后的图像的输出路径。
            top_left, top_right, bottom_right, bottom_left (tuple): 四角的地理坐标（经度，纬度）
   - Output: Geocorrected image.

4. GeometricCorrect_cv.py
   - Description: This script is used for geometric correction and relies heavily on the OpenCV library.
   - Usage: from GeometricCorrect_cv import reprojection
            corrected_image = reprojection(image_path, current_camera_position,
                                   pod_angles, aircraft_angles, focal_length_mm, sensor_size_mm, save_path)
            
   - Input: image_path: 待处理畸变图像的路径
            save_path: 保存路径
            current_camera_position: 相机当前地理位置（经、纬、高，单位：度，米）
            pod_angles: 载机姿态（横滚roll, 俯仰pitch, 航向yaw）
            aircraft_angles: 吊舱姿态（方位azimuth, 高低elevation
            focal_length_mm: 焦距（单位：毫米）
            sensor_size_mm: 传感器尺寸（长、宽，单位：毫米）
   - Output: Geocorrected image.



REQUIREMENTS
------------

- Python version: 3.12.0 or above
- Required packages: Please see requirements.txt for a complete list

INSTALLATION
------------

To install the required Python packages, run the following command:
`pip install -r requirements.txt`

USAGE
-----

Each script can be run individually as described in their usage. Make sure to navigate to the folder containing the scripts before executing them.

CONTRIBUTIONS
-------------

For any bugs, feature requests, or contributions, please open an issue or a pull request on the repository.

CONTACT
-------

For any queries regarding the scripts, please contact leeyaahoo (zuihaobuyao xiexie! youcuowuqingjinliangzijijiejue).

END OF README
