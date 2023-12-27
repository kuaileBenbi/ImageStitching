# 畸变原因

由相机高低角（也称为仰俯角）造成的图像畸变主要是指在拍摄时相机与被摄物体平面不垂直导致的视角变化。当相机向上或向下倾斜时，图像中的物体会出现透视扭曲。例如，当相机向下倾斜拍摄地面时，近处的物体会显得比远处的物体大，物体的形状和比例也会发生改变。这种畸变在航拍、建筑摄影等领域中尤为常见，理解和矫正这种畸变对于获取准确的图像信息非常重要。

# 解决方法

- 数学模型和算法：基于相机的内外参数矩阵，采用数学模型对图像进行变换和校正
- 深度学习方法：利用深度学习，学习畸变特征并自动进行矫正

## 数学方法
相机内外参解释可参考 [视觉slam十四讲](https://github.com/qomo/LearnSensorFusion/tree/master/books)

坐标系转换关系如下。其中，世界坐标系为局部坐标系且Zw为0，未有统一定义，需按照具体语境指定。

![坐标系转换关系](/pddon/坐标转换.png)

外参矩阵：描述世界坐标中的相机位置及其指向方向。向量t描述了世界坐标系原点在相机坐标系的位置，R代表了相机坐标系中世界坐标系轴的方向

对放置在高空的相机考虑小孔成像，则地面物点P(lng,lat,altitude)可探测器上成像为P(u,v)，如图所示：

![成像过程示意图](/pddon/成像示意图.png)

由于载机、吊舱等姿态，会使成像产生畸变。左图为仿真正射投影，右图为仿真畸变图像。

![模拟畸变效果-正射投影](pddon/orthograhic.png)
![模拟畸变效果-畸变结果](pddon/distorted.png)

举个🌰：
![像素坐标系到相机坐标系到世界坐标系变换示意图](/pddon/三个坐标系.png)

对世界坐标进行处理，使其转换为在图像分辨率内、按比例缩放的规范化坐标，保持长宽比例。
![规范化世界坐标系](/pddon/规范化世界坐标.png)

接下来确定映射关系：使用图像的四角像素坐标和它们对应的世界坐标来确定一个映射关系。这个关系可以通过计算一个变换矩阵来建立，这个矩阵通常被称为单应性矩阵（Homography Matrix）或几何校正矩阵。例如使用opencv计算：
cv2.findHomography(image_points, normalized_coords)

最后应用变换：使用计算出的单应性矩阵对整个图像进行变换。例如：
corrected_image = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]))

变换结果如图所示：


![校正结果](/pddon/校正结果.png)

相机位置在空间的可视化？

## 深度学习方法

参考大神论文 [Blind Geometric Distortion Correction on Images Through Deep Learning](https://github.com/xiaoyu258/GeoProj)
