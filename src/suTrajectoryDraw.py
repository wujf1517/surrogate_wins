'''
/**
!/usr/bin/env tensorflow
# -*- coding: utf-8 -*-
 * Copyright © 2019 Jianfeng_Wu. All rights reserved.
 * 
 * @author: Jianfeng_Wu
 * @date: 2022-04-26 
 * @time: 19:11:43 
 * Version:1.0.0
 */
 '''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from RL2class import RL2class

'''
这里将轨迹转化成RGB的TXT文件
'''

WIDTH = 32
HEIGTH = 64
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


def trajectoryDraw(filepath:str,df: pd.DataFrame()):
    total_row = len(df.iloc[:, 0])
    egox,egoy,egovx,egovy = addTrajectory(6040,-2500,df,0,True)

    su1x,su1y,su1vx,su1vy = addTrajectory(6040,-2500,df,9,False)

    su2x,su2y,su2vx,su2vy = addTrajectory(6040,-2500,df,18,False)

    k1,k2,k3,douTra = 0,0,0,0

    for w in range(HEIGTH):
        for h in range(WIDTH):
            if w >5 and h==5 and k2 <total_row-1:
                print(su1x[k2],",",su1y[k2],",",(k2+1),file=filepath)
                k2+=1
            elif w >5 and h==10 and k3 <total_row-1:
                print(su2x[k3],",",su2y[k3],",",(k3+1),file=filepath)
                k3+=1
            elif w>5 and h ==15 and k1<1:
                print(egox[k1],",",egoy[k1],",",(k1+1),file=filepath)
                if k1 <total_row-1:
                     k1+=1
            else:
                 print(255,",",255,",",255,file=filepath) # 效果不错
        

            

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

if __name__ == "__main__":
    for i in range(0,1,1):
        file_path = r'E:\code\scenarioagentcnn\scnarioData\baseline\%s' % (i+1) + '.csv'

        # E:\code\scenarioagentcnn\PicClass\5
        file_path2= 'picRgb\%s' % (i+1)+'.txt'
        fileWriter = open(file_path2, 'w+')
        # filepath = 'E:\code\scenarioagentcnn\scnarioData\baseline\',1,'.csv'
        # file_path = r'E:\code\scenarioagentcnn\scnarioData\baseline\1.csv'

        df = pd.read_csv(file_path,header=None)
        trajectoryDraw(fileWriter,df)