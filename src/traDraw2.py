from email import header
from email.header import Header


'''/**
!/usr/bin/env tensorflow
# -*- coding: utf-8 -*-
 * Copyright © 2019 Jianfeng_Wu. All rights reserved.
 * 
 * @author: Jianfeng_Wu
 * @date: 2022-06-03 
 * @time: 21:46:09 
 * Version:1.0.0
 * description:将轨迹转换为RGB值写入txt文件
 */'''

import pandas as pd
import matplotlib.pyplot as plt
# import numpy as np
# import tensorflow as tf
import os

from RL2class import RL2class

'''
这里将轨迹转化成RGB的TXT文件
'''

WIDTH = 64
HEIGTH = 256
# file_path = r'E:\code\scenarioagentcnn\scnarioData\baseline\1.csv'
# df = pd.read_csv(file_path)

# total_row = len(df.iloc[:, 0])
# total_column = len(df.loc[1, :])
# print("总共",total_row,"帧")

def addTrajectory(minx:int,miny:int,df: pd.DataFrame(),xi:int,ego:bool):
    '''
    注意上面的minx和miny都是用于坐标转换的
    需要确定道路边界坐标之后，再重新确定
    '''
    total_row = len(df.iloc[:, 0])
    x_arry = []
    y_arry = []
    vx_arry = []
    vy_arry=[]
    if ego==True:
        # min_vx = min(df.iloc[:,xi+2])
        # min_vy = min(df.iloc[:,xi+3])

        for i in range(total_row):
            x = int((df.iloc[i, xi]-minx+2)*10)
            y = int((df.iloc[i,xi+1]-miny)+1)
            vx = int((df.iloc[i,xi+2])*8) # 10
            vy = int((df.iloc[i,xi+3])*8) # 10
            x_arry.append(x)
            y_arry.append(y)
            vx_arry.append(vx)
            vy_arry.append(vy)
    else:
        # min_vx = min(df.iloc[:,xi+4])
        # min_vy = min(df.iloc[:,xi+5])
        for i in range(total_row):
            x = int((df.iloc[i, xi]-minx+2)*10)
            y = int((df.iloc[i,xi+1]-miny)+1)
            vx = int((df.iloc[i,xi+4])*8) # 10
            vy = int((df.iloc[i,xi+5])*8) # 10
            x_arry.append(x)
            y_arry.append(y)
            vx_arry.append(vx)
            vy_arry.append(vy)

    return x_arry,y_arry,vx_arry,vy_arry


def addTrajectory2(minx:int,miny:int,df: pd.DataFrame(),xi:int,ego:bool):
    '''
    针对LK文件的
    注意上面的minx和miny都是用于坐标转换的
    需要确定道路边界坐标之后，再重新确定
    表格里面第二列是横轴，第一列是行驶方向（纵轴）    
    '''
    total_row = len(df.iloc[:, 0])
    x_arry = []
    y_arry = []
    vx_arry = []
    vy_arry=[]
    if ego==True:
        # min_vx = min(df.iloc[:,xi+2])
        # min_vy = min(df.iloc[:,xi+3])

        for i in range(total_row):
            x = int(((df.iloc[i, xi]-minx)*0.5+5)) # # 行驶方向 图片最多容纳 500m 根据h=256计算得来
            y = int(((df.iloc[i,xi+1]-miny)*5+5)) # # 宽度方向 图片最多容纳 64/5 - 1 = 11m宽的路端 
            vx = int((df.iloc[i,xi+2])*10) # 10
            vy = int((df.iloc[i,xi+3])*10) # 10
            x_arry.append(x)
            y_arry.append(y)
            vx_arry.append(vx)
            vy_arry.append(vy)
    else:

        for i in range(total_row):
            x = int(((df.iloc[i, xi]-minx)*0.5+5))
            y = int((df.iloc[i,xi+1]-miny)*5+5)
            vx = int((df.iloc[i,xi+4])*8) # 10
            vy = int((df.iloc[i,xi+5])*8) # 10
            x_arry.append(x)
            y_arry.append(y)
            vx_arry.append(vx)
            vy_arry.append(vy)

    return x_arry,y_arry,vx_arry,vy_arry


def trajectoryDraw(filepath:str,df: pd.DataFrame()):
    total_row = len(df.iloc[:, 0])
    egox,egoy,egovx,egovy = addTrajectory(6040,-2500,df,9,True)

    su1x,su1y,su1vx,su1vy = addTrajectory(6040,-2500,df,0,False)

    su2x,su2y,su2vx,su2vy = addTrajectory(6040,-2500,df,18,False)

    k1,k2,k3,douTra = 0,0,0,0

    for h in range(HEIGTH):
        for w in range(WIDTH):

            if (w == egox[k1] and h == egoy[k1] and k1<1): # 只画自车初始位置
                print(egovx[k1],",",egovy[k1],",",(k1+1),file=filepath)
                douTra =1
#                print(egovx[k1],",",egovy[k1],",",k1+1)
                if k1 <total_row-1:
                    k1+=1
                    while (egox[k1-1] == su1x[k2] and egoy[k1-1] == su1y[k2]): # 防止交点位置不更新
                        k2+=1
                    while  (egox[k1-1] == egox[k1] and egoy[k1-1] == egoy[k1]): # 跳过同一个位置点，需不需要更新速度？
                        k1+=1
            elif(w == su1x[k2] and h == su1y[k2]):
                print(su1vx[k2],",",su1vy[k2],",",(k2+1),file=filepath)
                douTra=2
 #               print('su1vx:',su1vx[k2],",",su1vy[k2],",",k2+1)
                if k2 <total_row-1:
                    k2+=1
                    while  (su2x[k2-1] == su2x[k2] and su2y[k2-1] == su2y[k2]): # 跳过同一个位置点，需不需要更新速度？
                        k2+=1
            elif(w == su2x[k3] and h == su2y[k3]):
                print(su2vx[k3],",",su2vy[k3],",",(k3+1),file=filepath)
                douTra=3
                print('su2vxpixel:',su2vx[k3],",",su2vy[k3],",",(k3+1))
                if k3 <total_row-1:
                    k3+=1
                    while  (su2x[k3-1] == su2x[k3] and su2y[k3-1] == su2y[k3]): # 跳过同一个位置点，需不需要更新速度？
                        k3+=1
            elif(douTra==1):
                print(egovx[k1],",",egovy[k1],",",k1+1,file=filepath)   
                douTra = 0
            elif(douTra==2):
                print(su1vx[k2],",",su1vy[k2],",",k2+1,file=filepath)
                douTra = 0
            elif(douTra==3):
                print(su2vx[k3],",",su2vy[k3],",",k3+1,file=filepath)
                douTra = 0
            else:
                # print(255,",",2,",",125,file=filepath) # 效果不错
                # print(255,",",2,",",h,file=filepath)
                # print(2,",",2,",",2,file=filepath) # test accuracy:[0.28125000]
                # print(0,",",0,",",0,file=filepath)  # test accuracy: [0.15625000] # 1 test accuracy: [0.18750000]
                print(255,",",255,",",255,file=filepath)



def trajectoryDrawBox(filepath:str,df: pd.DataFrame()):
    total_row = len(df.iloc[:, 0])
    egox,egoy,egovx,egovy = addTrajectory(6040,-2500,df,9,True)

    su1x,su1y,su1vx,su1vy = addTrajectory(6040,-2500,df,0,False)

    su2x,su2y,su2vx,su2vy = addTrajectory(6040,-2500,df,18,False)

    k1,k2,k3,douTra = 0,0,0,0

    for h in range(HEIGTH):
        for w in range(WIDTH):

            if (w == egox[k1] and h == egoy[k1] and k1<1): # 只画自车初始位置
                print(egovx[k1],",",egovy[k1],",",(k1+1),file=filepath)
                douTra =1
#                print(egovx[k1],",",egovy[k1],",",k1+1)
                if k1 <total_row-1:
                    k1+=1
                    while (egox[k1-1] == su1x[k2] and egoy[k1-1] == su1y[k2]): # 防止交点位置不更新
                        k2+=1
                    while  (egox[k1-1] == egox[k1] and egoy[k1-1] == egoy[k1]): # 跳过同一个位置点，需不需要更新速度？
                        k1+=1
            elif(w == su1x[k2] and h == su1y[k2]):
                print(su1vx[k2],",",su1vy[k2],",",(k2+1),file=filepath)
                douTra=2
 #               print('su1vx:',su1vx[k2],",",su1vy[k2],",",k2+1)
                if k2 <total_row-1:
                    k2+=1
                    while  (su2x[k2-1] == su2x[k2] and su2y[k2-1] == su2y[k2]): # 跳过同一个位置点，需不需要更新速度？
                        k2+=1
            elif(w == su2x[k3] and h == su2y[k3]):
                print(su2vx[k3],",",su2vy[k3],",",(k3+1),file=filepath)
                douTra=3
                print('su2vxpixel:',su2vx[k3],",",su2vy[k3],",",(k3+1))
                if k3 <total_row-1:
                    k3+=1
                    while  (su2x[k3-1] == su2x[k3] and su2y[k3-1] == su2y[k3]): # 跳过同一个位置点，需不需要更新速度？
                        k3+=1
            elif(douTra==1):
                print(egovx[k1],",",egovy[k1],",",k1+1,file=filepath)   
                douTra = 0
            elif(douTra==2):
                print(su1vx[k2],",",su1vy[k2],",",k2+1,file=filepath)
                douTra = 0
            elif(douTra==3):
                print(su2vx[k3],",",su2vy[k3],",",k3+1,file=filepath)
                douTra = 0
            else:
                # print(255,",",2,",",125,file=filepath) # 效果不错
                # print(255,",",2,",",h,file=filepath)
                # print(2,",",2,",",2,file=filepath) # test accuracy:[0.28125000]
                # print(0,",",0,",",0,file=filepath)  # test accuracy: [0.15625000] # 1 test accuracy: [0.18750000]
                print(255,",",255,",",255,file=filepath)




# f = open("pic\wo2.txt", 'w+')  
def minPosition(filedir:str,num_total:int):
    minx_array,miny_array=[],[]

    for i in range(0,num_total,1):
        # file_path = r'E:\code\scenarioagentcnn\scenarioData2\LK\%s' % (i+1) + '-LK.csv'
        file_path = filedir+'%s' % (i+1) + '.csv'
        df = pd.read_csv(file_path,header=None)
        miny1 = min(df.iloc[:, 1])
        miny_array.append(miny1)
        miny2 = min(df.iloc[:, 10])
        miny_array.append(miny2)
        miny3 = min(df.iloc[:, 19])
        miny_array.append(miny3)

        minx1 = min(df.iloc[:, 0])
        minx_array.append(minx1)
        minx2 = min(df.iloc[:, 9])
        minx_array.append(minx2)
        minx3 = min(df.iloc[:, 18])
        minx_array.append(minx3)

    minx_total = min(minx_array)
    miny_total = min(miny_array)
    return minx_total,miny_total


def trajectoryWithoutV(filepath,filedir:str,df: pd.DataFrame(),minx_total,miny_total):
    r_num = 125
    total_row = len(df.iloc[:, 0])   

    egoy,egox,egovx,egovy = addTrajectory2(minx_total,miny_total,df,0,True) # egoy是行驶方向

    su1y,su1x,su1vx,su1vy = addTrajectory2(minx_total,miny_total,df,9,False)

    su2y,su2x,su2vx,su2vy = addTrajectory2(minx_total,miny_total,df,18,False)

    k1,k2,k3,douTra = 0,0,0,0

    for h in range(HEIGTH):
        for w in range(WIDTH):
            if (w == egox[k1] and h == egoy[k1] and k1<1): # 只画自车初始位置
                print(r_num,",",r_num,",",(k1+1),file=filepath)
                douTra =1
#                print(egovx[k1],",",egovy[k1],",",k1+1)
                if k1 <total_row-1:
                    k1+=1
                    while (egox[k1-1] == su1x[k2] and egoy[k1-1] == su1y[k2]): # 防止交点位置不更新
                        k2+=1
                    while  (egox[k1-1] == egox[k1] and egoy[k1-1] == egoy[k1]): # 跳过同一个位置点，需不需要更新速度？
                        k1+=1
            elif(w == su1x[k2] and h == su1y[k2]):
                # print(r_num,",",r_num,",",(k2+1),file=filepath)
                print(su1vx[k2],",",su1vy[k2],",",(k2+1),file=filepath)
                # print('su1xpixel:',su1x[k2],",",su1y[k2],",",(k2+1))
                douTra=2
 #               print('su1vx:',su1vx[k2],",",su1vy[k2],",",k2+1)
                if k2 <total_row-1:
                    k2+=1
                    while  (su1x[k2-1] == su1x[k2] and su1y[k2-1] == su1y[k2] and k2 <total_row-1): # 跳过同一个位置点，需不需要更新速度？
                        k2+=1
            elif(w == su2x[k3] and h == su2y[k3]):
                # print(r_num,",",r_num,",",(k3+1),file=filepath)
                print(su2vx[k3],",",su2vy[k3],",",(k3+1),file=filepath)
                douTra=3
                # print('su2vxpixel:',su2vx[k3],",",su2vy[k3],",",(k3+1))
                if k3 <total_row-1:
                    k3+=1
                    while  (su2x[k3-1] == su2x[k3] and su2y[k3-1] == su2y[k3] and k3 <total_row-1): # 跳过同一个位置点，需不需要更新速度？
                        k3+=1
            elif(douTra==1):
                print(egovx[k1],",",egovy[k1],",",k1+1,file=filepath)   
                # print(r_num,",",r_num,",",k1+1,file=filepath)   
                douTra = 0
            elif(douTra==2):
                print(su1vx[k2],",",su1vy[k2],",",k2+1,file=filepath)
                # print(r_num,",",r_num,",",k2+1,file=filepath)
                douTra = 0
            elif(douTra==3):
                print(su2vx[k3],",",su2vy[k3],",",k3+1,file=filepath)
                # print(r_num,",",r_num,",",k3+1,file=filepath)
                douTra = 0
            else:
                # print(255,",",2,",",125,file=filepath) # 效果不错
                # print(255,",",2,",",h,file=filepath)
                # print(2,",",2,",",2,file=filepath) # test accuracy:[0.28125000]
                # print(0,",",0,",",0,file=filepath)  # test accuracy: [0.15625000] # 1 test accuracy: [0.18750000]
                print(255,",",255,",",255,file=filepath)


def DrawBox(w,h,su1x,su1y,k_stop1,r_num:int,k2,filepath):
    if (w == su1x[k2]-1 and h == su1y[k2]-1 and k_stop1 == 1):
        print(r_num,",",r_num,",",(k2+1),file=filepath)
        print(r_num,",",r_num,",",(k2+1))
    elif (w == su1x[k2]-1 and h == su1y[k2] and k_stop1 == 1):
        print(r_num,",",r_num,",",(k2+1),file=filepath)
    elif (w == su1x[k2]-1 and h == su1y[k2]+1 and k_stop1 == 1):
        print(r_num,",",r_num,",",(k2+1),file=filepath)
    elif (w == su1x[k2] and h == su1y[k2]-1 and k_stop1 == 1):
        print(r_num,",",r_num,",",(k2+1),file=filepath)
    elif (w == su1x[k2] and h == su1y[k2] and k_stop1 == 1):
        print(r_num,",",r_num,",",(k2+1),file=filepath)
    elif (w == su1x[k2] and h == su1y[k2]+1 and k_stop1 == 1):
        print(r_num,",",r_num,",",(k2+1),file=filepath)
    elif (w == su1x[k2]+1 and h == su1y[k2]-1 and k_stop1 == 1):
        print(r_num,",",r_num,",",(k2+1),file=filepath)
    elif (w == su1x[k2]+1 and h == su1y[k2] and k_stop1 == 1):
        print(r_num,",",r_num,",",(k2+1),file=filepath)
    elif (w == su1x[k2]+1 and h == su1y[k2]+1 and k_stop1 == 1):
        print(r_num,",",r_num,",",(k2+1),file=filepath)
        k_stop1 = 2
    return k_stop1



def trajectoryWithoutVBox(filepath,filedir:str,df: pd.DataFrame(),minx_total,miny_total):
    r_num = 125
    r_ego = 10
    total_row = len(df.iloc[:, 0])   

    egoy,egox,egovx,egovy = addTrajectory2(minx_total,miny_total,df,0,True) # egoy是行驶方向

    su1y,su1x,su1vx,su1vy = addTrajectory2(minx_total,miny_total,df,9,False)

    su2y,su2x,su2vx,su2vy = addTrajectory2(minx_total,miny_total,df,18,False)

    k1,k2,k3,douTra = 0,0,0,0
    k_stopego,k_stop1,k_stop2 = 0,0,0
    
    if max(su1x) == min(su1x) and max(su1y) == min(su1y):
        k_stop1 = 1
    elif max(su2x) == min(su2x) and max(su2y) == min(su2y):
        k_stop2 = 1
        print(k_stop2)

    for h in range(HEIGTH):
        for w in range(WIDTH):
            # if k_stop1 ==1 and w == su1x[k2] and h == su1y[k2] :
            #     k_stop1 = DrawBox(w,h,su1x,su1y,k_stop1,r_num,k2,filepath)
            # elif k_stop2==1 and w == su2x[k3] and h == su2y[k3]:
            #     k_stop2 = DrawBox(w,h,su2x,su2y,k_stop2,r_num,k3,filepath)
            if (w == egox[k1] and h == egoy[k1] and k1<1): # 只画自车初始位置
                print(r_num,",",r_num,",",(k1+1),file=filepath)
                douTra =1
#                print(egovx[k1],",",egovy[k1],",",k1+1)
                if k1 <total_row-1:
                    k1+=1
                    while (egox[k1-1] == su1x[k2] and egoy[k1-1] == su1y[k2]): # 防止交点位置不更新
                        k2+=1
                    while  (egox[k1-1] == egox[k1] and egoy[k1-1] == egoy[k1]): # 跳过同一个位置点，需不需要更新速度？
                        k1+=1

            elif (w == egox[k1]-1 and h == egoy[k1]-1 and k1<2):
                print(r_ego,",",r_ego,",",(k2+1),file=filepath)
                print(r_ego,",",r_ego,",",(k2+1))
            elif (w == egox[k1]-1 and h == egoy[k1] and k1<2):
                print(r_ego,",",r_ego,",",(k2+1),file=filepath)
            elif (w == egox[k1]-1 and h == egoy[k1]+1 and k1<2):
                print(r_ego,",",r_ego,",",(k2+1),file=filepath)
            elif (w == egox[k1] and h == egoy[k1]-1 and k1<2):
                print(r_ego,",",r_ego,",",(k2+1),file=filepath)
            elif (w == egox[k1] and h == egoy[k1] and k1<2):
                print(r_ego,",",r_ego,",",(k2+1),file=filepath)
            elif (w == egox[k1] and h == egoy[k1]+1 and k1<2):
                print(r_ego,",",r_ego,",",(k2+1),file=filepath)
            elif (w == egox[k1]+1 and h == egoy[k1]-1 and k1<2):
                print(r_ego,",",r_ego,",",(k2+1),file=filepath)
            elif (w == egox[k1]+1 and h == egoy[k1] and k1<2):
                print(r_ego,",",r_ego,",",(k2+1),file=filepath)
            elif (w == egox[k1]+1 and h == egoy[k1]+1 and k1<2):
                print(r_ego,",",r_ego,",",(k2+1),file=filepath)


            elif(w == su1x[k2] and h == su1y[k2]):
                # print(r_num,",",r_num,",",(k2+1),file=filepath)
                print(su1vx[k2],",",su1vy[k2],",",(k2+1),file=filepath)
                # print('su1xpixel:',su1x[k2],",",su1y[k2],",",(k2+1))
                douTra=2
 #               print('su1vx:',su1vx[k2],",",su1vy[k2],",",k2+1)
                if k2 < total_row-1:
                    k2+=1
                    while  (su1x[k2-1] == su1x[k2] and su1y[k2-1] == su1y[k2] and k2 <total_row-1): # 跳过同一个位置点，需不需要更新速度？
                        k2+=1

            elif (w == su1x[k2]-1 and h == su1y[k2]-1 and k2<2):
                print(r_num,",",r_num,",",(k2+1),file=filepath)
                # print(r_num,",",r_num,",",(k2+1))
            elif (w == su1x[k2]-1 and h == su1y[k2] and k2<2):
                print(r_num,",",r_num,",",(k2+1),file=filepath)
            elif (w == su1x[k2]-1 and h == su1y[k2]+1 and k2<2):
                print(r_num,",",r_num,",",(k2+1),file=filepath)
            elif (w == su1x[k2] and h == su1y[k2]-1 and k2<2):
                print(r_num,",",r_num,",",(k2+1),file=filepath)
            elif (w == su1x[k2] and h == su1y[k2] and k2<2):
                print(r_num,",",r_num,",",(k2+1),file=filepath)
            elif (w == su1x[k2] and h == su1y[k2]+1 and k2<3):
                print(r_num,",",r_num,",",(k2+1),file=filepath)
            elif (w == su1x[k2]+1 and h == su1y[k2]-1 and k2<3):
                print(r_num,",",r_num,",",(k2+1),file=filepath)
            elif (w == su1x[k2]+1 and h == su1y[k2] and k2<3):
                print(r_num,",",r_num,",",(k2+1),file=filepath)
            elif (w == su1x[k2]+1 and h == su1y[k2]+1 and k2<4):
                print(r_num,",",r_num,",",(k2+1),file=filepath)
                print(r_num,",",r_num,",",(k2+1))


            elif (w == su2x[k3]-1 and h == su2y[k3]-1 and k_stop2 == 1):
                print(r_num,",",r_num,",",(k3+1),file=filepath)
                print(r_num,",",r_num,",",(k3+1))
            elif (w == su2x[k3]-1 and h == su2y[k3] and k_stop2 == 1):
                print(r_num,",",r_num,",",(k3+1),file=filepath)
            elif (w == su2x[k3]-1 and h == su2y[k3]+1 and k_stop2 == 1):
                print(r_num,",",r_num,",",(k3+1),file=filepath)
            elif (w == su2x[k3] and h == su2y[k3]-1 and k_stop2 == 1):
                print(r_num,",",r_num,",",(k3+1),file=filepath)
            elif (w == su2x[k3] and h == su2y[k3] and k_stop2 == 1):
                print(r_num,",",r_num,",",(k3+1),file=filepath)
            elif (w == su2x[k3] and h == su2y[k3]+1 and k_stop2 == 1):
                print(r_num,",",r_num,",",(k3+1),file=filepath)
            elif (w == su2x[k3]+1 and h == su2y[k3]-1 and k_stop2 == 1):
                print(r_num,",",r_num,",",(k3+1),file=filepath)
            elif (w == su2x[k3]+1 and h == su2y[k3] and k_stop2 == 1):
                print(r_num,",",r_num,",",(k3+1),file=filepath)
            elif (w == su2x[k3]+1 and h == su2y[k3]+1 and k_stop2 == 1):
                print(r_num,",",r_num,",",(k3+1),file=filepath)
            elif (w == su2x[k3]-1 and h == su2y[k3]-1 and k_stop2 == 1):
                print(r_num,",",r_num,",",(k3+1),file=filepath)
                print(r_num,",",r_num,",",(k3+1))
            elif (w == su2x[k3]-1 and h == su2y[k3] and k_stop2 == 1):
                print(r_num,",",r_num,",",(k2+1),file=filepath)
            elif (w == su2x[k3]-1 and h == su2y[k2]+1 and k_stop2 == 1):
                print(r_num,",",r_num,",",(k2+1),file=filepath)
            elif (w == su2x[k2] and h == su2y[k2]-1 and k_stop1 == 1):
                print(r_num,",",r_num,",",(k2+1),file=filepath)
            elif (w == su1x[k2] and h == su1y[k2] and k_stop1 == 1):
                print(r_num,",",r_num,",",(k2+1),file=filepath)
            elif (w == su1x[k2] and h == su1y[k2]+1 and k_stop1 == 1):
                print(r_num,",",r_num,",",(k2+1),file=filepath)
            elif (w == su1x[k2]+1 and h == su1y[k2]-1 and k_stop1 == 1):
                print(r_num,",",r_num,",",(k2+1),file=filepath)
            elif (w == su1x[k2]+1 and h == su1y[k2] and k_stop1 == 1):
                print(r_num,",",r_num,",",(k2+1),file=filepath)
            elif (w == su1x[k2]+1 and h == su1y[k2]+1 and k_stop1 == 1):
                print(r_num,",",r_num,",",(k2+1),file=filepath)



            elif(w == su2x[k3] and h == su2y[k3]):
                # print(r_num,",",r_num,",",(k3+1),file=filepath)
                print(su2vx[k3],",",su2vy[k3],",",(k3+1),file=filepath)
                douTra=3
                # print('su2vxpixel:',su2vx[k3],",",su2vy[k3],",",(k3+1))
                if k3 <total_row-1:
                    k3+=1
                    while  (su2x[k3-1] == su2x[k3] and su2y[k3-1] == su2y[k3] and k3 <total_row-1): # 跳过同一个位置点，需不需要更新速度？
                        k3+=1
            # elif(douTra==1):
            #     print(egovx[k1],",",egovy[k1],",",k1+1,file=filepath)   
            #     # print(r_num,",",r_num,",",k1+1,file=filepath)   
            #     douTra = 0
            # elif(douTra==2):
            #     print(su1vx[k2],",",su1vy[k2],",",k2+1,file=filepath)
            #     # print(r_num,",",r_num,",",k2+1,file=filepath)
            #     douTra = 0
            # elif(douTra==3):
            #     print(su2vx[k3],",",su2vy[k3],",",k3+1,file=filepath)
            #     # print(r_num,",",r_num,",",k3+1,file=filepath)
            #     douTra = 0
            else:
                # print(255,",",2,",",125,file=filepath) # 效果不错
                # print(255,",",2,",",h,file=filepath)
                # print(2,",",2,",",2,file=filepath) # test accuracy:[0.28125000]
                # print(0,",",0,",",0,file=filepath)  # test accuracy: [0.15625000] # 1 test accuracy: [0.18750000]
                print(0,",",0,",",0,file=filepath)


def trajectoryWithVBoxLine(filepath,filedir:str,df: pd.DataFrame(),minx_total,miny_total):
    r_num = 125
    r_ego = 10
    total_row = len(df.iloc[:, 0])   

    egoy,egox,egovx,egovy = addTrajectory2(minx_total,miny_total,df,0,True) # egoy是行驶方向

    su1y,su1x,su1vx,su1vy = addTrajectory2(minx_total,miny_total,df,9,False)

    su2y,su2x,su2vx,su2vy = addTrajectory2(minx_total,miny_total,df,18,False)

    k1,k2,k3,douTra = 0,0,0,0
    k_stopego,k_stop1,k_stop2 = 0,0,0
    r_ego = round(egovy[1]/10)
    
    if max(su1x) == min(su1x) and max(su1y) == min(su1y):
        k_stop1 = 1
    elif max(su2x) == min(su2x) and max(su2y) == min(su2y):
        k_stop2 = 1
        print(k_stop2)

    for h in range(HEIGTH):
        for w in range(WIDTH):
            if (w == 2 ):
                print(r_num,",",r_num,",",r_num,file=filepath)
            elif (w == 61 ):
                print(r_num,",",r_num,",",r_num,file=filepath)
            elif (w == egox[k1] and h == egoy[k1] and k1==0): # 只画自车初始位置
                print(r_ego,",",r_ego,",",(k2+1),file=filepath)
                douTra =1
#                print(egovx[k1],",",egovy[k1],",",k1+1)
                if k1 <total_row-1:
                    k1+=1
                    while (egox[k1-1] == su1x[k2] and egoy[k1-1] == su1y[k2]): # 防止交点位置不更新
                        k2+=1
                    while  (egox[k1-1] == egox[k1] and egoy[k1-1] == egoy[k1]): # 跳过同一个位置点，需不需要更新速度？
                        k1+=1
            elif(w<=(egox[0]+1) and w >= (egox[0]-1) and h<=(egoy[0]+1) and h >= (egoy[0]-1)):
                # while(w<(egox[k1]+2) and w > (egox[k1]-2) and h<(egox[k1]+2) and h > (egox[k1]-2)):
                print(r_ego,",",r_ego,",",(r_ego),file=filepath)
                # print(r_ego)

            elif(w == su1x[k2] and h == su1y[k2]):
                # print(r_num,",",r_num,",",(k2+1),file=filepath)
                print(su1vx[k2],",",su1vy[k2],",",(k2+1),file=filepath)
                # print('su1xpixel:',su1x[k2],",",su1y[k2],",",(k2+1))
                douTra=2
 #               print('su1vx:',su1vx[k2],",",su1vy[k2],",",k2+1)
                if k2 < total_row-1:
                    k2+=1
                    while  (su1x[k2-1] == su1x[k2] and su1y[k2-1] == su1y[k2] and k2 <total_row-1): # 跳过同一个位置点，需不需要更新速度？
                        k2+=1
            elif(w<=(su1x[0]+1) and w >= (su1x[0]-1) and h<=(su1y[0]+1) and h >= (su1y[0]-1)):
                print(r_num,",",r_num,",",(k2+1),file=filepath)
                # print(r_num)

            elif(w == su2x[k3] and h == su2y[k3]):
                # print(r_num,",",r_num,",",(k3+1),file=filepath)
                print(su2vx[k3],",",su2vy[k3],",",(k3+1),file=filepath)
                douTra=3
                # print('su2vxpixel:',su2vx[k3],",",su2vy[k3],",",(k3+1))
                if k3 <total_row-1:
                    k3+=1
                    while  (su2x[k3-1] == su2x[k3] and su2y[k3-1] == su2y[k3] and k3 <total_row-1): # 跳过同一个位置点，需不需要更新速度？
                        k3+=1
            elif(w<=(su2x[0]+1) and w >= (su2x[0]-1) and h<=(su2y[0]+1) and h >= (su2y[0]-1)):
                print(r_num,",",r_num,",",(k3+1),file=filepath)
                # print(r_num)

            # elif(douTra==1):
            #     print(egovx[k1],",",egovy[k1],",",k1+1,file=filepath)   
            #     # print(r_num,",",r_num,",",k1+1,file=filepath)   
            #     douTra = 0
            # elif(douTra==2):
            #     print(su1vx[k2],",",su1vy[k2],",",k2+1,file=filepath)
            #     # print(r_num,",",r_num,",",k2+1,file=filepath)
            #     douTra = 0
            # elif(douTra==3):
            #     print(su2vx[k3],",",su2vy[k3],",",k3+1,file=filepath)
            #     # print(r_num,",",r_num,",",k3+1,file=filepath)
            #     douTra = 0
            else:
                # print(255,",",2,",",125,file=filepath) # 效果不错
                # print(255,",",2,",",h,file=filepath)
                # print(2,",",2,",",2,file=filepath) # test accuracy:[0.28125000]
                # print(0,",",0,",",0,file=filepath)  # test accuracy: [0.15625000] # 1 test accuracy: [0.18750000]
                print(255,",",255,",",255,file=filepath)


def trajectoryWithoutVBoxLine(filepath,filedir:str,df: pd.DataFrame(),minx_total,miny_total):
    r_num = 125
    r_vehicle = 50
    r_ego = 10
    total_row = len(df.iloc[:, 0])   

    egoy,egox,egovx,egovy = addTrajectory2(minx_total,miny_total,df,0,True) # egoy是行驶方向

    su1y,su1x,su1vx,su1vy = addTrajectory2(minx_total,miny_total,df,9,False)

    su2y,su2x,su2vx,su2vy = addTrajectory2(minx_total,miny_total,df,18,False)

    k1,k2,k3 = 0,0,0
    k_stopego,k_stop1,k_stop2 = 0,0,0
    r_ego = round(egovy[1]/10)
    
    if max(su1x) == min(su1x) and max(su1y) == min(su1y):
        k_stop1 = 1
    elif max(su2x) == min(su2x) and max(su2y) == min(su2y):
        k_stop2 = 1
        print(k_stop2)

    for h in range(HEIGTH):
        for w in range(WIDTH):
            if (w == 2 ):
                print(r_num,",",r_num,",",r_num,file=filepath)
            elif (w == 61 ):
                print(r_num,",",r_num,",",r_num,file=filepath)
            elif (w == egox[k1] and h == egoy[k1] and k1==0): # 只画自车初始位置
                print(r_ego,",",r_ego,",",(k2+1),file=filepath)
#                print(egovx[k1],",",egovy[k1],",",k1+1)
                if k1 <total_row-1:
                    k1+=1
                    # while (egox[k1-1] == su1x[k2] and egoy[k1-1] == su1y[k2]): # 防止交点位置不更新
                    #     k2+=1
                    while  (egox[k1-1] == egox[k1] and egoy[k1-1] == egoy[k1]): # 跳过同一个位置点，需不需要更新速度？
                        k1+=1
            elif(w<=(egox[0]+1) and w >= (egox[0]-1) and h<=(egoy[0]+1) and h >= (egoy[0]-1)):
                # while(w<(egox[k1]+2) and w > (egox[k1]-2) and h<(egox[k1]+2) and h > (egox[k1]-2)):
                print(r_ego,",",r_ego,",",(r_ego),file=filepath)
                # print(r_ego)

            elif(w == su1x[k2] and h == su1y[k2]):
                # print(r_num,",",r_num,",",(k2+1),file=filepath)
                print(r_vehicle,",",r_vehicle,",",(k2+1),file=filepath)
                # print('su1xpixel:',su1x[k2],",",su1y[k2],",",(k2+1))
 #               print('su1vx:',su1vx[k2],",",su1vy[k2],",",k2+1)
                if k2 < total_row-1:
                    k2+=1
                    while  (su1x[k2-1] == su1x[k2] and su1y[k2-1] == su1y[k2] and k2 <total_row-1): # 跳过同一个位置点，需不需要更新速度？
                        k2+=1
            elif(w<=(su1x[0]+1) and w >= (su1x[0]-1) and h<=(su1y[0]+1) and h >= (su1y[0]-1)):
                print(r_num,",",r_num,",",(k2+1),file=filepath) # 填入起始速度值？
                # print(r_num)

            elif(w == su2x[k3] and h == su2y[k3]):
                # print(r_num,",",r_num,",",(k3+1),file=filepath)
                print(r_vehicle,",",r_vehicle,",",(k3+1),file=filepath)
                # print('su2vxpixel:',su2vx[k3],",",su2vy[k3],",",(k3+1))
                if k3 <total_row-1:
                    k3+=1
                    while  (su2x[k3-1] == su2x[k3] and su2y[k3-1] == su2y[k3] and k3 <total_row-1): # 跳过同一个位置点，需不需要更新速度？
                        k3+=1
            elif(w<=(su2x[0]+1) and w >= (su2x[0]-1) and h<=(su2y[0]+1) and h >= (su2y[0]-1)):
                print(r_num,",",r_num,",",(k3+1),file=filepath)
                # print(r_num)

            # elif(douTra==1):
            #     print(egovx[k1],",",egovy[k1],",",k1+1,file=filepath)   
            #     # print(r_num,",",r_num,",",k1+1,file=filepath)   
            #     douTra = 0
            # elif(douTra==2):
            #     print(su1vx[k2],",",su1vy[k2],",",k2+1,file=filepath)
            #     # print(r_num,",",r_num,",",k2+1,file=filepath)
            #     douTra = 0
            # elif(douTra==3):
            #     print(su2vx[k3],",",su2vy[k3],",",k3+1,file=filepath)
            #     # print(r_num,",",r_num,",",k3+1,file=filepath)
            #     douTra = 0
            else:
                # print(255,",",2,",",125,file=filepath) # 效果不错
                # print(255,",",2,",",h,file=filepath)
                # print(2,",",2,",",2,file=filepath) # test accuracy:[0.28125000]
                # print(0,",",0,",",0,file=filepath)  # test accuracy: [0.15625000] # 1 test accuracy: [0.18750000]
                print(255,",",255,",",255,file=filepath)

def trajectoryWithoutVBoxLine2(filepath,filedir:str,df: pd.DataFrame(),minx_total,miny_total):
    r_num = 125
    r_vehicle = 50
    r_ego = 10
    total_row = len(df.iloc[:, 0])   

    egoy,egox,egovx,egovy = addTrajectory2(minx_total,miny_total,df,0,True) # egoy是行驶方向

    su1y,su1x,su1vx,su1vy = addTrajectory2(minx_total,miny_total,df,9,False)

    su2y,su2x,su2vx,su2vy = addTrajectory2(minx_total,miny_total,df,18,False)

    k1,k2,k3 = 0,0,0
    k_stopego,k_stop1,k_stop2 = 0,0,0
    r_ego = round(egovy[1]/10)
    
    if max(su1x) == min(su1x) and max(su1y) == min(su1y):
        k_stop1 = 1
    elif max(su2x) == min(su2x) and max(su2y) == min(su2y):
        k_stop2 = 1
        print(k_stop2)

    for h in range(HEIGTH):
        for w in range(WIDTH):
            if (w == 2 ):
                print(r_num,",",r_num,",",r_num,file=filepath)
            elif (w == 61 ):
                print(r_num,",",r_num,",",r_num,file=filepath)
            elif (w == egox[k1] and h == egoy[k1] and k1==0): # 只画自车初始位置
                print(r_ego,",",r_ego,",",(k2+1),file=filepath)
#                print(egovx[k1],",",egovy[k1],",",k1+1)
                if k1 <total_row-1:
                    k1+=2
                    # while (egox[k1-1] == su1x[k2] and egoy[k1-1] == su1y[k2]): # 防止交点位置不更新
                    #     k2+=1
                    while  (egox[k1-1] == egox[k1] and egoy[k1-1] == egoy[k1]): # 跳过同一个位置点，需不需要更新速度？
                        k1+=2
            elif(w<=(egox[0]+1) and w >= (egox[0]-1) and h<=(egoy[0]+1) and h >= (egoy[0]-1)):
                # while(w<(egox[k1]+2) and w > (egox[k1]-2) and h<(egox[k1]+2) and h > (egox[k1]-2)):
                print(r_ego,",",r_ego,",",(r_ego),file=filepath)
                # print(r_ego)

            elif(w == su1x[k2] and h == su1y[k2]):
                # print(r_num,",",r_num,",",(k2+1),file=filepath)
                print(r_vehicle,",",r_vehicle,",",(k2+1),file=filepath)
                # print('su1xpixel:',su1x[k2],",",su1y[k2],",",(k2+1))
 #               print('su1vx:',su1vx[k2],",",su1vy[k2],",",k2+1)
                if 2*k2 < total_row-1:
                    k2+=1
                    while  (su1x[2*(k2-1)] == su1x[2*k2] and su1y[2*(k2-1)] == su1y[2*k2] and 2*k2 <total_row-1): # 跳过同一个位置点，需不需要更新速度？
                        k2+=1
            elif(w<=(su1x[0]+1) and w >= (su1x[0]-1) and h<=(su1y[0]+1) and h >= (su1y[0]-1)):
                print(r_num,",",r_num,",",(k2+1),file=filepath) # 填入起始速度值？
                # print(r_num)

            elif(w == su2x[k3] and h == su2y[k3]):
                # print(r_num,",",r_num,",",(k3+1),file=filepath)
                print(r_vehicle,",",r_vehicle,",",(k3+1),file=filepath)
                # print('su2vxpixel:',su2vx[k3],",",su2vy[k3],",",(k3+1))
                if k3 <total_row-1:
                    k3+=2
                    while  (su2x[k3-1] == su2x[k3] and su2y[k3-1] == su2y[k3] and k3 <total_row-1): # 跳过同一个位置点，需不需要更新速度？
                        k3+=2
            elif(w<=(su2x[0]+1) and w >= (su2x[0]-1) and h<=(su2y[0]+1) and h >= (su2y[0]-1)):
                print(r_num,",",r_num,",",(k3+1),file=filepath)
                # print(r_num)

            else:
                # print(255,",",2,",",125,file=filepath) # 效果不错
                print(255,",",255,",",255,file=filepath)


if __name__ == "__main__":
    # minx_array,miny_array=[],[]
    filedir_sou = r'data\FV_dec_35_5\base2'+'\\' # 原始场景文件位置
    filedir_tar = r'data\FV_dec_35_5\RGB_wv_box1'

    files = os.listdir(filedir_sou)   # 读入文件夹
    num_csv = len(files)       # 统计文件夹中的文件个数

    minx_total,miny_total = minPosition(filedir_sou,num_csv)

    print(minx_total,miny_total)

    for i in range(0,num_csv,1):
        file_path = filedir_sou+'\%s' % (i+1) + '.csv'

        # E:\code\scenarioagentcnn\PicClass\5
        # file_path2= 'imagergb\%s' % (i+1)+'.txt'
        file_path2= filedir_tar+'\%s' % (i+1)+'.txt'
        fileWriter = open(file_path2, 'w+')
        # filepath = 'E:\code\scenarioagentcnn\scnarioData\baseline\',1,'.csv'
        # file_path = r'E:\code\scenarioagentcnn\scnarioData\baseline\1.csv'

        df = pd.read_csv(file_path,header=None)
        # trajectoryDraw(fileWriter,df)
        # trajectoryWithoutV(fileWriter,filedir_sou,df,minx_total,miny_total)
        # trajectoryWithoutVBox(fileWriter,filedir_sou,df,minx_total,miny_total)
        trajectoryWithVBoxLine(fileWriter,filedir_sou,df,minx_total,miny_total)
    print(num_csv,'个RGB文件构建完成!写在'+filedir_tar+'位置')


# print(minx_arry)