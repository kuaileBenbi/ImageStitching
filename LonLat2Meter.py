import math 
from math import cos, radians
 
def position_turn(Latitude,Longitude):
    '''
    参考地址：https://blog.csdn.net/Dust_Evc/article/details/102847870?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Edefault-7.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Edefault-7.control
    度数与弧度转化公式：1°=π/180°，1rad=180°/π。
    地球半径：6371000M
    地球周长：2 * 6371000M  * π = 40030173
    纬度38°地球周长：40030173 * cos38 = 31544206M
    任意地球经度周长：40030173M
    经度（东西方向）1M实际度：360°/31544206M=1.141255544679108e-5=0.00001141
    纬度（南北方向）1M实际度：360°/40030173M=8.993216192195822e-6=0.00000899
    '''
    R=637100
    L=2*math.pi* R
    Lat_l=L*cos(math.radians(Latitude))#当前纬度地球周长，弧度转化为度数
    Lng_l=40008000#当前经度地球周长
    Lat_C=Lat_l/360
    Lng_C=Lng_l/360
    Latitude_m=Latitude*Lat_C
    Longitude_m=Longitude*Lng_C
    return Latitude_m,Longitude_m

if __name__ == "__main__":
    # print(position_turn(0.0001099707431880824, 0.00011649782690321656))
    print(position_turn(1.9045140239462289e-06, 2.0086895415738377e-06))
