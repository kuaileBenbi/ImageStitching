import cv2 as cv
import numpy as np
from math import tan, cos, sin, radians, atan, degrees


def cal_Delta(alpha, beta, gamma, height):
    '''
    返回经过三维旋转之后，飞艇视点在海平面的偏移
    Parameters
    ----------
    alpha : 飞艇朝向角，正北为0,顺时针旋转为正
    beta : 飞艇俯仰角，水平为0，抬头为正
    gamma : 飞艇滚转角，水平为0，向右滚转为正
    height : 飞艇中心高度

    Returns
    -------
    返回在水平，竖直方向上的偏移量，单位：千米
    '''

    # 飞艇顶视图，计算飞艇在有朝向角alpha和俯仰角beta下的视点P
    visionPrv = np.array([abs(height) * tan(radians(beta)) * sin(radians(alpha)),
                          abs(height) * tan(radians(beta)) * cos(radians(alpha)),
                          0 - height])

    # 飞艇x轴方向向量
    directF = np.array([cos(radians(beta)) * sin(radians(alpha)),
                      cos(radians(beta)) * cos(radians(alpha)),
                      sin(radians(beta))])
    Dx = directF[0]
    Dy = directF[1]
    Dz = directF[2]

    # 飞艇顶视图，计算视点visionPre绕飞艇x轴旋转滚转角gamma后的视点
    Px = visionPrv[0]
    Py = visionPrv[1]
    Pz = visionPrv[2]

    Ax = Px * cos(radians(gamma)) + (Dy * Pz - Dz * Py) * sin(radians(gamma)) + \
        Dx * (Dx * Px + Dy * Py + Dz * Pz) * (1 - cos(radians(gamma)))
    Ay = Py * cos(radians(gamma)) + (Dz * Px - Dx * Pz) * sin(radians(gamma)) + \
        Dy * (Dx * Px + Dy * Py + Dz * Pz) * (1 - cos(radians(gamma)))
    Az = Pz * cos(radians(gamma)) + (Dx * Py - Dy * Px) * sin(radians(gamma)) + \
        Dy * (Dx * Px + Dy * Py + Dz * Pz) * (1 - cos(radians(gamma)))

    # 得到从飞艇(0,0,0)点到视点visionAft的向量
    visionAft = np.array([Ax, Ay, Az])

    # 得到视线与海平面的交点vsionCross
    # 空间直线(x-Ax)/Ax＝(y-Ay)/Ay＝(z-Az)/Az
    seaPx = (-height - Az) * Ax / Az + Ax
    seaPy = (-height - Az) * Ay / Az + Ay

    return seaPx, seaPy


def cal_JW(longi, lanti, alpha, beta, gamma, height):
    '''
    返回经过三维旋转之后，飞艇视点的经纬度值
    Parameters
    ----------
    longi : 飞艇经度
    lanti : 飞艇纬度
    alpha : 飞艇朝向角，正北为0,顺时针旋转为正
    beta : 飞艇俯仰角，水平为0，抬头为正
    gamma : 飞艇滚转角，水平为0，向右滚转为正
    height : 飞艇中心高度

    Returns
    -------
    目前飞艇视点的经纬度值
    '''
    Delta_x, Delta_y = cal_Delta(alpha, beta, gamma, height)
    Delta_longi = degrees(Delta_x / 6378.2)   # 赤道半径
    Delta_lanti = degrees(Delta_y / 6371)    # 地球半径
    # print("东西：%f  南北：%f" % (Delta_x, Delta_y))
    # print(longi + Delta_longi, lanti + Delta_lanti)
    return longi + Delta_longi, lanti + Delta_lanti


def cal_target_JW(resolution, imgshape, oinfo, apos, alpha):
    """
    :param oJ: 相机视场中心的经度
    :param oW: 相机视场中心的纬度
    :param ax: 目标中心在图像的行
    :param ay: 目标中心在图像的列
    :param alpha: 飞艇航向角
    :return: 目标中心的经纬度aJ,aW
    """
    oJ, oW = oinfo
    ax, ay = apos
    # return oJ, oW
    width = round(imgshape[1] / 2)
    height = round(imgshape[0] / 2)

    x0 = (ax - height) * resolution[1]
    y0 = (ay - width) * resolution[0]

    theta = 90 - alpha

    x1 = x0 * cos(theta) - y0 * sin(theta)    # dst_east(km)
    y1 = x0 * sin(theta) + y0 * cos(theta)    # dst_north(km)

    deltaJ = degrees(x1 / 6378.2)        #  * 360  # 赤道半径
    deltaW = degrees(y1 / 6371)         # * 360    # 地球半径

    aJ = oJ + deltaJ
    aW = oW + deltaW
    # print("东西：", deltaJ)
    # print("南北", deltaW)
    return aJ, aW


def thresholdSegmention(img, tmin, tmax):

    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    _, t1 = cv.threshold(img, tmin, 255, cv.THRESH_BINARY)
    _, t2 = cv.threshold(img, tmax, 255, cv.THRESH_BINARY_INV)
    t1 = 255 - t1
    t2 = 255 - t2
    r1 = t1 + t2
    r2 = 255 - r1

    # openImg = cv.morphologyEx(r2, cv.MORPH_OPEN, kernel)

    return r2


def adaptiveSegmenation(gray):
    res = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 15, 10)
    # cv.imshow("ada", res)
    # cv.waitKey(300)
    return res


def histSegmentation(gray):
    # blur = cv.GaussianBlur(gray, (5, 5), 0)
    _, res = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    # cv.imshow("ada", res)
    # cv.waitKey(300)
    return res


def SobelSegmentation(gray):
    # 高斯模糊处理:去噪(效果最好)
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    # Sobel计算XY方向梯度
    gradX = cv.Sobel(blur, ddepth=cv.CV_32F, dx=1, dy=0)
    gradY = cv.Sobel(blur, ddepth=cv.CV_32F, dx=0, dy=1)
    # 计算梯度差
    gradient = cv.subtract(gradX, gradY)
    # 绝对值
    gradient = cv.convertScaleAbs(gradient)
    # 高斯模糊处理:去噪(效果最好)
    blured = cv.GaussianBlur(gradient, (3, 3), 0)
    # 二值化
    _, dst = cv.threshold(blured, 90, 255, cv.THRESH_BINARY)
    # 滑动窗口
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (107, 76))
    # 形态学处理:形态闭处理(腐蚀)
    closed = cv.morphologyEx(dst, cv.MORPH_CLOSE, kernel)
    # 腐蚀与膨胀迭代
    closed = cv.erode(closed, None, iterations=4)
    closed = cv.dilate(closed, None, iterations=4)
    # cv.imshow("oper", closed)
    # cv.waitKey(300)
    return closed


def isFlag(lat, lng, FThigh, beta, gamma, alpha):
    """
    :param lat: 纬度
    :param lng: 经度
    :param FThigh: 高度
    :param beta: 俯仰角
    :param gamma: 滚转角
    :param alpha: 航向角
    :return: True or False
    """
    if abs(lat) > 90:
        return False
    elif abs(lng) > 180:
        return False
    elif FThigh < -200 or FThigh > 30000:
        return False
    elif abs(beta) > 90:
        return False
    elif abs(gamma) > 180:
        return False
    elif abs(alpha) > 360:
        return False
    else:
        return True


def drawRectangle(openImg, srcImg, resolution, shipLng, shipLat, alpha, IfLngLat, IfImg):
    aW = 0
    aJ = 0
    draw = srcImg.copy()
    # print("shape: ", srcImg.shape)
    m, n = srcImg.shape[:2]
    rect = [[round(n/2), round(m/2)], [n, m]]
    contoursShort, _ = cv.findContours(openImg.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contoursShort:
        # print("DrawRectangle1")
        c = sorted(contoursShort, key=cv.contourArea, reverse=True)[0]
        rect = cv.minAreaRect(c)
        # print(rect1[1][0], rect1[1][1])
        box = np.int0(cv.boxPoints(rect))
        # print(box1.shape)
        draw = cv.drawContours(srcImg.copy(), [box], -1, (255, 255, 255), 2)
        if IfLngLat:
            aJ, aW = cal_target_JW(resolution, srcImg, shipLng, shipLat, rect[0][1], rect[0][0], alpha)
            # print("tools: {}, {}".format(round(aW, 5), round(aJ, 5)))
        
        if IfLngLat and IfImg:
            text = "la:" + str(round(aW, 3)) + " " + "lg:" + str(round(aJ, 3)) + "\n" +\
                 "w:" + str(round(rect[1][0])) + " " + "h:" + str(round(rect[1][1]))
        elif IfImg:
            text = "w:" + str(round(rect[1][0])) + " " + "h:" + str(round(rect[1][1]))
        elif IfLngLat:
            text = "la:" + str(round(shipLat, 3)) + " " + "lg:" + str(round(shipLng, 3))
        else:
            text = "None"

        tx = round(rect[0][0] - 0.5 * rect[1][0]) - 30
        ty0 = round(rect[0][1] - 0.5 * rect[1][1]) - 30
        d0 = 30
        for i, txt in enumerate(text.split('\n')):
            ty = ty0 + i * d0
            cv.putText(draw, txt, (tx, ty), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    return draw, aW, aJ, rect[1][0], rect[1][1]


def drawRectangleMouse(openImg, srcImg, resolution, shipLng, shipLat, alpha, IfLngLat, IfImg):
    aW = 0
    aJ = 0
    draw = srcImg.copy()
    m, n = srcImg.shape
    rect = [[round(m / 2), round(n / 2)], [m, n]]
    contoursShort, _ = cv.findContours(openImg.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contoursShort:
        # print("DrawRectangle1")
        c = sorted(contoursShort, key=cv.contourArea, reverse=True)[0]
        rect = cv.minAreaRect(c)
        # print(rect1[1][0], rect1[1][1])
        box = np.int0(cv.boxPoints(rect))
        # print(box1.shape)
        draw = cv.drawContours(srcImg.copy(), [box], -1, (255, 255, 255), 3)
        if shipLat:
            aJ, aW = cal_target_JW(resolution, srcImg, shipLng, shipLat, rect[0][0], rect[0][1], alpha)
            # print("tools: {}, {}".format(round(aW, 5), round(aJ, 5)))
        if IfLngLat and IfImg:
            text = "la:" + str(round(aW, 3)) + " " + "lg:" + str(round(aJ, 3)) + "\n" + "w:" + \
                   str(round(rect[1][0])) + " " + "h:" + str(round(rect[1][1]))
        elif IfImg:
            text = "w:" + str(round(rect[1][0])) + " " + "h:" + str(round(rect[1][1]))
        elif IfLngLat:
            text = "la:" + str(round(shipLat, 3)) + " " + "lg:" + str(round(shipLng, 3))
        else:
            text = "None"
        tx = round(rect[0][0] - 0.5 * rect[1][0]) - 30
        ty0 = round(rect[0][1] - 0.5 * rect[1][1]) - 30
        d0 = 30
        for i, txt in enumerate(text.split('\n')):
            ty = ty0 + i * d0
            cv.putText(draw, txt, (tx, ty), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    return draw, aW, aJ, rect[1][0], rect[1][1]


def Tenen_grad(img):
    img_copy = img.copy()
    img_copy = cv.filter2D(img_copy, -1, np.ones([3, 3]) / 9.)
    sobel_x = cv.Sobel(img_copy, -1, 1, 0)
    sobel_y = cv.Sobel(img_copy, -1, 0, 1)
    sobel = sobel_x + sobel_y
    tenen_grad = cv.mean(sobel)[0]
    return tenen_grad


def calMouseInfo(Point0, Point1, resolution, img, labelW, labelH, shipInfo, ifLngLat):
    resJ = 0
    resW = 0
    imgDis = img.copy()
    imgCorner0 = (Point0[0] * imgDis.shape[1] // labelW,
                  Point0[1] * imgDis.shape[0] // labelH)
    imgCorner1 = (Point1[0] * imgDis.shape[1] // labelW,
                  Point1[1] * imgDis.shape[0] // labelH)
    cv.rectangle(img, imgCorner0, imgCorner1, (255, 255, 255), 2)
    targetY = imgCorner0[1] + (imgCorner1[1] - imgCorner0[1]) // 2
    targetX = imgCorner0[0] + (imgCorner1[0] - imgCorner0[0]) // 2

    if ifLngLat:
        resJ, resW = cal_target_JW(resolution, img.copy(), shipInfo[0], shipInfo[1], targetY, targetX, shipInfo[2])
    return imgCorner0, imgCorner1, resJ, resW

def transformer(data):
    du = int(data)
    fen = int((data - du) * 60)
    miao = int((((data - du)*60 - fen) * 60))
    res = str(du) + '°'+str(fen) + '′' + str(miao) + '″'
    return res




if __name__ == "__main__":
    img = cv.imread(r'D:\A_p\pycharm_pro\ship_info\img_test\qqq\100000243.bmp', cv.CV_8UC1)
    # print(img.max(), img.min())
    res = histSegmentation(img)
    cv.imshow("ada", res)
    cv.waitKey(0)