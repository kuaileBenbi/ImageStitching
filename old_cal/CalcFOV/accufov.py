# -*- coding: utf-8 -*-
"""
Created on Mar 14 2022
@author: Sunmer
"""

from turtle import position
import numpy as np
from math import radians, degrees, cos, sin, sqrt

# EARTHRADIUS = 6375.844

def rotate(point, axis, theta):    
    #according to https://www.cnblogs.com/graphics/archive/2012/08/08/2609005.html

    '''
    Rotate about arbitrary axis

    point' = (x',y',z') = (x,y,z)T, T = f(axis, theta)
    
    T = \n
        | a*a+(1-a*a)*cos(theta), a*b*(1-cos(theta))+c*sin(theta), a*c*(1-cos(theta))-b*sin(theta) |\n
        | a*b*(1-cos(theta))-c*sin(theta), b*b+(1-b*b)*cos(theta), b*c*(1-cos(theta))+a*sin(theta) |\n
        | a*c*(1-cos(theta))+b*sin(theta), b*c*(1-cos(theta))-a*sin(theta), c*c+(1-c*c)*cos(theta) |\n

    Parameters
    ----------
    point = (x,y,z)
    axis = (a,b,c)
    theta, unit:degrees

    Returns
    -------
    point' = (x',y',z')
    '''

    a,b,c = axis
    theta = radians(theta)
    tmat = np.array([
        [a*a+(1-a*a)*cos(theta),            a*b*(1-cos(theta))+c*sin(theta),    a*c*(1-cos(theta))-b*sin(theta)],
        [a*b*(1-cos(theta))-c*sin(theta),   b*b+(1-b*b)*cos(theta),             b*c*(1-cos(theta))+a*sin(theta)],
        [a*c*(1-cos(theta))+b*sin(theta),   b*c*(1-cos(theta))-a*sin(theta),    c*c+(1-c*c)*cos(theta)          ]])

    return tuple(np.dot(point, tmat))
    
    

def calcfov(lng, lat, height, alpha, beta, gamma, fovw, fovh, mode):
    '''
    根据飞艇的飞行姿态数据、相机的当前视场角与安装模式，计算相机当前对地面所成像的四个顶点对应的经纬度

    Parameters
    ----------
    lng : 经度
    lat : 纬度
    height : 高度（单位：千米）
    alpha : 飞艇朝向角，单位：角度，0~360，正北为0,正东为90
    beta : 飞艇俯仰角，单位：角度，水平为0，抬头为正
    gamma : 飞艇横滚角，单位：角度，水平为0，左翼抬升为正
    fovw : 相机视场宽（单位：角度）
    fovh : 相机视场高（单位：角度）
    mode : 相机安装方式, 0~3。在《相机视场的计算》所定义的飞艇坐标系中，若图像(0,0)∈(-y,+z)则mode=0,(+y,+z)->1, (+y,-z)->2, (-y,-z)->3

    Returns
    -------
    fov=list((lng_00,lat_00), (lng_w0,lat_w0), (lng_wh,lat_wh), (lng_0h,lat_0h))

    '''
    # 飞艇坐标系 ftx,fty,ftz见 《相机视场的计算》
    # 地球坐标系 ex,ey,ez 同 飞艇初始状态

    ftx = (1,0,0)
    fty = (0,1,0)
    ftz = (0,0,1)

    # 一、计算视线方向向量在飞艇坐标系下的表示
    # 计算vecfov
    # mode=0
    vecmid = (0,0,-1)    # 视场中心方向向量
    vecfovtopmid = rotate(vecmid, ftx, fovh/2)  #vecfovtopmid.x = 0, apparently
    vecfovleftmid = rotate(vecmid, fty, fovw/2) #vecfovleftmid.y = 0
    point_vecfovtopmid = (0, -vecfovtopmid[1]/vecfovtopmid[2], -1) #与平面 z = -1 交点
    point_vecfovleftmid = (-vecfovleftmid[0]/vecfovleftmid[2], 0, -1) #与平面 z = -1 交点
    vecfov00 = np.multiply((point_vecfovleftmid[0], point_vecfovtopmid[1], -1), (1,1,1))
    vecfovw0 = np.multiply((point_vecfovleftmid[0], point_vecfovtopmid[1], -1), (-1,1,1))
    vecfovwh = np.multiply((point_vecfovleftmid[0], point_vecfovtopmid[1], -1), (-1,-1,1))
    vecfov0h = np.multiply((point_vecfovleftmid[0], point_vecfovtopmid[1], -1), (1,-1,1))
    # mode=0,1,2,3
    vecfov00, vecfovw0, vecfovwh, vecfov0h = rotate((vecfov00, vecfovw0, vecfovwh, vecfov0h), ftz, -90*mode)
    
    # 二、计算飞艇坐标系的方向在地球坐标系下的表示
    # step1 绕ftz轴旋转 (-alpha)
    ftx, fty = rotate((ftx,fty), ftz, -alpha)
    # step2 绕ftx轴旋转 beta
    fty, ftz = rotate((fty,ftz), ftx, beta)
    # step3 绕fty轴旋转 gamma
    ftz, ftx = rotate((ftz,ftx), fty, gamma)
    
    # 三、计算视线方向向量在地球坐标系下的表示
    v00,vw0,vwh,v0h = np.dot((vecfov00, vecfovw0, vecfovwh, vecfov0h), (ftx, fty, ftz))
    
    # 四、计算视线与海平面交点
    p00, pw0, pwh, p0h = [(-height*vecfov[0]/vecfov[2], -height*vecfov[1]/vecfov[2]) for vecfov in (v00,vw0,vwh,v0h)] # 舍去了3th dim
    # print(p00, pw0, pwh, p0h, sep='\n')
    positions = [np.add((lng,lat), list(map(degrees, np.dot(p,1/EARTHRADIUS)))) for p in (p00, pw0, pwh, p0h)]
    return positions

def locatetarget(tl, br, imgshape, fov):
    '''
    当在图像中使用矩形框出目标后，计算目标的尺寸（矩形对角线长）和中心经纬度
    
    Parameters
    ----------
    tl: tuple(leftcol, toprow)
    br: tuple(rightcol, bottomrow)
    imgshape: tuple(width, height)
    fov: return of calcfov

    Returns
    -------
    tuple(targetsize, targetlng, targetlat)
    Units：meters，degrees，degrees
    '''

    left, top = tl
    right, bottom = br
    width, height = imgshape
    pos00, posw0, poswh, pos0h = fov

    calc = lambda i,col,row: pos00[i] + col/width*(posw0[i]-pos00[i])+row/height*(pos0h[i]-pos00[i])
    tl_lng, tl_lat = [calc(i, left, top) for i in range(2)]
    br_lng, br_lat = [calc(i, right, bottom) for i in range(2)]

    targetsize = radians(sqrt((tl_lng-br_lng)**2+(tl_lat-br_lat)**2))*EARTHRADIUS*1000
    targetlng = (tl_lng + br_lng)/2
    targetlat = (tl_lat + br_lat)/2
    return targetsize, targetlng, targetlat

if __name__ == '__main__':
    from random import random,randint

    # positions = calcfov(110, 19, 20, 0, 0, 0, 22.6, 18, 0) # 经纬高 航向 俯仰 滚转 fovw fovh mode
    EARTHRADIUS = 6375.844+0.2
    
    positions = calcfov(110, 19, 20, 0, 30, 0, 28, 17, 0)
    for p in positions:
        print(p)

    # positions = calcfov(110.12, 19.07, 20, 0, 0.41, 4.82+0.1, 0, 0, 0)
    # print(positions)

    # positions = calcfov(110.12, 19.07, 20, 0, 0.41, 4.82-0.1, 0, 0, 0)
    # print(positions)
    # lng_max = np.max(positions,axis=0)
    # lng_min = np.min(positions, axis=0)
    # lat_max = np.max(positions, axis=1)
    # lat_min = np.min(positions, axis=1)
    # print(lng_min,lng_max,lat_min,lat_max)
    # ret = locatetarget(tl=(0,0), br=(640, 512), imgshape=(640, 512), fov=positions)
    # print(ret)
    # for i in range(1):
    #     fovw = 0 #random()*radians(10)
    #     fovh = 0 #random()*radians(10)
    #     alpha = 0 #randint(0, 36000) / 100
    #     beta = 0 # randint(-85, 85) / 100
    #     gamma = 0 #randint(-85, 85) / 100
    #     height = 100000 #randint(200, 30000)

    #     positions = calcfov(110, 19, height, alpha, beta, gamma, fovw, fovh, 0)
        