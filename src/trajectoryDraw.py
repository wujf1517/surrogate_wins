'''/**
!/usr/bin/env tensorflow
# -*- coding: utf-8 -*-
 * Copyright © 2019 Jianfeng_Wu. All rights reserved.
 * 
 * @author: Jianfeng_Wu
 * @date: 2022-05-05 
 * @time: 12:40:27 
 * Version:1.0.0
 * description:将轨迹转换为RGB值写入txt文件
 */'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import tensorflow as tf

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
            x = int(((df.iloc[i, xi]-minx)*1+5)) # # 行驶方向
            y = int(((df.iloc[i,xi+1]-miny)*5+5)) 
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
            x = int(((df.iloc[i, xi]-minx)*1+5))
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

# def trajectoryDraw(filepath:str,df: pd.DataFrame()):
#     total_row = len(df.iloc[:, 0])
#     egox,egoy,egovx,egovy = addTrajectory(6040,-2500,df,0,True)

#     su1x,su1y,su1vx,su1vy = addTrajectory(6040,-2500,df,9,False)

#     su2x,su2y,su2vx,su2vy = addTrajectory(6040,-2500,df,18,False)

#     k1,k2,k3,douTra = 0,0,0,0

#     for h in range(HEIGTH):
#         for w in range(WIDTH):
#             if (w == egox[k1] and h == egoy[k1]):
#                 print(egovx[k1],",",egovy[k1],",",(k1+1),file=filepath)
#                 douTra =1
# #                print(egovx[k1],",",egovy[k1],",",k1+1)
#                 if k1 <total_row-1:
#                     k1+=1
#                     while (egox[k1-1] == su1x[k2] and egoy[k1-1] == su1y[k2]): # 防止交点位置不更新
#                         k2+=1
#                     while  (egox[k1-1] == egox[k1] and egoy[k1-1] == egoy[k1]): # 跳过同一个位置点，需不需要更新速度？
#                         k1+=1
#             elif(w == su1x[k2] and h == su1y[k2]):
#                 print(su1vx[k2],",",su1vy[k2],",",(k2+1),file=filepath)
#                 douTra=2
#  #               print('su1vx:',su1vx[k2],",",su1vy[k2],",",k2+1)
#                 if k2 <total_row-1:
#                     k2+=1
#                     while  (su2x[k2-1] == su2x[k2] and su2y[k2-1] == su2y[k2]): # 跳过同一个位置点，需不需要更新速度？
#                         k2+=1
#             elif(w == su2x[k3] and h == su2y[k3]):
#                 print(su2vx[k3],",",su2vy[k3],",",(k3+1),file=filepath)
#                 douTra=3
#                 print('su2vxpixel:',su2vx[k3],",",su2vy[k3],",",(k3+1))
#                 if k3 <total_row-1:
#                     k3+=1
#                     while  (su2x[k3-1] == su2x[k3] and su2y[k3-1] == su2y[k3]): # 跳过同一个位置点，需不需要更新速度？
#                         k3+=1
#             elif(douTra==1):
#                 print(egovx[k1],",",k1+1,",",h,file=filepath)  #   
#                 douTra = 0
#             elif(douTra==2):
#                 print(su1vx[k2],",",k2+1,",",h,file=filepath)
#                 douTra = 0
#             elif(douTra==3):
#                 print(su2vx[k3],",",k3+1,",",h,file=filepath)
#                 douTra = 0
#             else:
#                 # print(255,",",2,",",125,file=filepath) # 效果不错
#                 print(255,",",2,",",h,file=filepath)
#                 # print(2,",",2,",",2,file=filepath) # test accuracy:[0.28125000]
#                 # print(0,",",0,",",0,file=filepath)  # test accuracy: [0.15625000] # 1 test accuracy: [0.18750000]
#                 # print(255,",",255,",",255,file=filepath)


# f = open("pic\wo2.txt", 'w+')  
def minPosition(filedir:str,num_total:int):
    minx_array,miny_array=[],[]

    for i in range(0,num_total,1):
        # file_path = r'E:\code\scenarioagentcnn\scenarioData2\LK\%s' % (i+1) + '-LK.csv'
        file_path = filedir+'%s' % (i+1) + '.csv'
        # file_path = filedir+'%s' % (i+1) + '.csv'
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

    # filedir = r'E:\code\scenarioagentcnn\scenarioData2\LK\'
    # filepath = filedir+'%s' % (i+1) + '-LK.csv'
    # egox,egoy,egovx,egovy = addTrajectory(6040,-2500,df,0,True)

    # su1x,su1y,su1vx,su1vy = addTrajectory(6040,-2500,df,9,False)

    # su2x,su2y,su2vx,su2vy = addTrajectory(6040,-2500,df,18,False)

    # minx_total,miny_total = minPosition(filedir,2883)    

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
                print(r_num,",",r_num,",",(k2+1),file=filepath)
                douTra=2
 #               print('su1vx:',su1vx[k2],",",su1vy[k2],",",k2+1)
                if k2 <total_row-1:
                    k2+=1
                    while  (su2x[k2-1] == su2x[k2] and su2y[k2-1] == su2y[k2]): # 跳过同一个位置点，需不需要更新速度？
                        k2+=1
            elif(w == su2x[k3] and h == su2y[k3]):
                print(r_num,",",r_num,",",(k3+1),file=filepath)
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


# import pandas as pd
def filePre(filedir:str,filedir2:str,num_total:int):
    '''数据格式转换,调换列的位置'''
    for i in range(0,num_total,1):
        # file_path = r'E:\code\scenarioagentcnn\scenarioData2\LK\%s' % (i+1) + '-LK.csv'
        file_path = filedir+'%s' % (i+1) + '.csv'
        file_path2 = filedir2+'%s' % (i+1) + '.csv'
        df = pd.read_csv(file_path,header=None)
        print(min(df.iloc[:,0]))
        df[[0, 1]] = df[[1, 0]]
        df[[9, 10]] = df[[10, 9]]
        df[[18, 19]] = df[[19, 18]]
        df.to_csv(file_path2,header=None,index=None)

def desample(filedir:str,filedir2:str,num_total:int):
    '''间隔取数'''
    for i in range(0,num_total,1):
        # file_path = r'E:\code\scenarioagentcnn\scenarioData2\LK\%s' % (i+1) + '-LK.csv'
        file_path = filedir+'%s' % (i+1) + '.csv'
        file_path2 = filedir2+'%s' % (i+1) + '.csv'
        df = pd.read_csv(file_path,header=None)

     
        # data = pd.read_csv(file_path,header=None)
        # df = pd.DataFrame(data)
        a=[]
        for i in range(0,len(df),2):##每隔20行取数据
            a.append(i)
        file = df.iloc[a]
        f = pd.DataFrame(file)
        f.to_csv(file_path2,header=None,index=False)
        print('finished')




def fileContact(filedir:str,filedir2:str,num_total_sou:int,num_total_tar:int):
    '''数据格式转换,调换列的位置'''
    for i in range(0,num_total_sou,1):
        # file_path = r'E:\code\scenarioagentcnn\scenarioData2\LK\%s' % (i+1) + '-LK.csv'
        file_path = filedir+'%s' % (i+1) + '.csv'
        file_path2 = filedir2+'%s' % (i+num_total_tar+1) + '.csv'
        df = pd.read_csv(file_path,header=None)
        df.to_csv(file_path2,header=None,index=None)

def fileContactFile(filedir:str,filedir2:str):
    '''数据格式转换,调换列的位置'''
    df1 = pd.read_csv(filedir,header=None)
    df2 = pd.read_csv(filedir2,header=None)
    df = pd.concat([df1,df2],axis=0)

    df.to_csv(filedir,header=None,index=None)


if __name__ == "__main__":
    # minx_array,miny_array=[],[]
    filedir = r'data\base'+'\\'
    # filedir = r'E:\code\scenarioagentcnn\scnarioData\base'+'\\'

    # minx_total,miny_total = minPosition(filedir,4000)

    # print(minx_total,miny_total)

    # for i in range(0,4000,1):
    #     file_path = r'data\base\%s' % (i+1) + '.csv'
    #     file_path2= 'data\RGB\%s' % (i+1)+'.txt'

    #     # file_path2= 'imageRGBwithouV\%s' % (i+1)+'.txt'
    #     fileWriter = open(file_path2, 'w+')

    #     df = pd.read_csv(file_path,header=None)
    #     # trajectoryDraw(fileWriter,df)
    #     trajectoryWithoutV(fileWriter,filedir,df,minx_total,miny_total)
    # print('个RGB文件构建完成!写在'+file_path2+'位置')

    '''文件格式转换'''
    # # filedir = r'E:\code\scenarioagentcnn\scenarioData7\baseline'+'\\'
    # # filedir2 = r'E:\code\scenarioagentcnn\scenarioData7\base'+'\\'
    # filedir2 = r'E:\code\surrogate\data\FV_dec_35_5\base'+'\\'
    # filedir = r'E:\code\surrogate\record'+'\\'
    # filePre(filedir,filedir2,2499)

    '''base合并'''
    # filedir = r'E:\code\scenarioagentcnn\scenarioData4\base'+'\\'
    # filedir2 = r'E:\code\scenarioagentcnn\scenarioData6\base'+'\\'
    # fileContact(filedir,filedir2,1792,3844)

    '''result合并'''
    # filedir = r'E:\code\scenarioagentcnn\scenarioData6\计算结果\minResult.csv'
    # filedir2 = r'E:\code\scenarioagentcnn\scenarioData4\计算结果\minResult.csv'
    # fileContactFile(filedir,filedir2)

    '''修改采样频率'''
    # filedir = r'E:\code\scenarioagentcnn\scenarioData7\baseline'+'\\'
    # filedir2 = r'E:\code\scenarioagentcnn\scenarioData7\base'+'\\'
    filedir2 = r'data\FV_dec_35_5\base2'+'\\'
    filedir = r'data\FV_dec_35_5\base'+'\\'
    desample(filedir,filedir2,2499)