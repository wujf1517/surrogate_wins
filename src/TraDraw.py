import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import tensorflow as tf
from PIL import Image

'''
这是写RGB.txt的测试文件
画不出轨迹的两种可能性：
一种是存在两个位置相同，无法迭代更新位置：增加重复自迭代部分
初始位置为负数，无法进入迭代；解决办法是重新选取基准点位置
su2x: [51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 
51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51]
su2y: [-1, 0, 3, 5, 7, 10, 12, 15, 17, 19, 22, 24, 27, 29, 31, 34, 36, 39, 41, 43, 46, 48, 51, 53, 55, 58, 60, 63, 65, 67, 70, 72, 75, 77, 
79, 82, 84, 87, 89, 91, 94, 96, 99, 101, 103, 106, 108, 111, 113, 115, 118, 120, 123, 125, 127, 130, 132, 135, 137, 139, 142, 144, 147, 149, 151, 154, 156, 159, 161, 163, 166, 168, 171, 173, 175, 178, 180, 183, 185, 187, 190, 192, 195, 197, 199, 202, 204, 207, 209, 211]  
'''


file_path = r'data\FV_dec_35_5\base\\1019.csv'
df = pd.read_csv(file_path,header=None) # 加上header=None，否则默认第一行为标题
WIDTH = 64
HEIGTH = 128*2

# a = np.arange(0, 5)
# # 把numpy格式的转化为tensor的数据类型
# b = tf.convert_to_tensor(a, dtype=tf.int64)
# print("a:", a)
# print("b:", b)

# su2x: [39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39]
# su2y: [0, 0, 2, 4, 5, 7, 9, 10, 12, 14, 15, 17, 19, 20, 22, 24, 25, 27, 29, 30, 32, 34, 35, 37, 39]

# su2x: [39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39]
# su2y: [0, 1, 2, 4, 6, 8, 9, 11, 13, 14, 16, 18, 20, 21, 23, 25, 27, 28, 30, 32, 34, 35, 37, 39, 40]

total_row = len(df.iloc[:, 0])
total_column = len(df.loc[1, :])
print("总共",total_row,"帧")
print("自车横坐标如下：")
print(df.iloc[:,0])
print("自车纵坐标y如下：")
print(df.iloc[:,1])
print("参数数目：",total_column)
plt.plot(df.iloc[:,0],df.iloc[:,1],df.iloc[:,9],df.iloc[:,10],df.iloc[:,18],df.iloc[:,19])
plt.show()
print(df.iloc[0,18]) # 从0开始计数


f = open("two2.txt", 'w+')  

def addTrajectory(minx:int,miny:int,df: pd.DataFrame(),xi:int,ego:bool):
    # min_x = min(df.iloc[:, xi])
    # min_y = min(df.iloc[:,xi+1])
    x_arry = []
    y_arry = []
    vx_arry = []
    vy_arry=[]
    if ego==True:
        min_vx = min(df.iloc[:,xi+2])
        min_vy = min(df.iloc[:,xi+3])

        for i in range(total_row):
            x = int(((df.iloc[i, xi]-minx)*1+2))
            y = int(((df.iloc[i,xi+1]-miny)*5+1))
            vx = int((df.iloc[i,xi+2])*8) # 10
            vy = int((df.iloc[i,xi+3])*8) # 10
            x_arry.append(x)
            y_arry.append(y)
            vx_arry.append(vx)
            vy_arry.append(vy)
    else:
        min_vx = min(df.iloc[:,xi+4])
        min_vy = min(df.iloc[:,xi+5])
        if df.iloc[2, xi] !=None:
            for i in range(total_row):
                x = int(((df.iloc[i, xi]-minx)*0.5+5))
                y = int((df.iloc[i,xi+1]-miny)*5+5)
                # x = (df.iloc[i, xi]-minx)//2+5
                # y = (df.iloc[i,xi+1]-miny)//0.2+5
                vx = int((df.iloc[i,xi+4])*8) # 10
                vy = int((df.iloc[i,xi+5])*8) # 10
                x_arry.append(x)
                y_arry.append(y)
                vx_arry.append(vx)
                vy_arry.append(vy)
        else:
            for i in range(total_row):
                x = 0
                y = 0
                vx = 0
                vy = 0
                x_arry.append(x)
                y_arry.append(y)
                vx_arry.append(vx)
                vy_arry.append(vy)

    return x_arry,y_arry,vx_arry,vy_arry


'''
su1x: [27, 28, 30, 31, 33, 34, 36, 37, 39, 40, 41, 43, 44, 46, 47, 49, 50, 52, 53, 55, 56, 57, 59, 60, 62, 63, 65, 66, 68, 69, 71, 72, 74, 75, 76, 
78, 79, 81, 82, 84, 85, 87, 88, 90, 91, 92, 94, 95, 97, 98, 99, 100, 102, 103, 104, 105, 106, 106, 107, 108, 109, 109, 110, 110, 111, 111, 112, 112]
su1y: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
68

su1x: [27.0, 28.0, 30.0, 31.0, 33.0, 34.0, 36.0, 37.0, 39.0, 40.0, 41.0, 43.0, 44.0, 46.0, 47.0, 49.0, 50.0, 52.0, 53.0, 55.0, 56.0, 57.0, 59.0, 60.0, 62.0, 63.0, 65.0, 66.0, 68.0, 69.0, 71.0, 72.0, 74.0, 75.0, 76.0, 78.0, 79.0, 81.0, 82.0, 84.0, 85.0, 87.0, 88.0, 90.0, 91.0, 92.0, 94.0, 95.0, 97.0, 98.0, 99.0, 100.0, 102.0, 103.0, 104.0, 105.0, 106.0, 106.0, 107.0, 108.0, 109.0, 109.0, 110.0, 110.0, 111.0, 111.0, 112.0, 112.0]
su1y: [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 
5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
68

'''

# min_x = min(df.iloc[:,0])
# min_y = min(df.iloc[:,1])
# min_vx = min(df.iloc[:,2])
# min_vy = min(df.iloc[:,3])
# print('最小值：',min_x)


# for i in range(total_row):
#     x = int((max_x - df.iloc[i, 0])*10)
#     y = int((max_y -df.iloc[i,1]))
#     vx = int((max_vx-df.iloc[i,2])*10)
#     vy = int((max_vy-df.iloc[i,3]))
#     for w in range(WIDTH):
#         for h in range(HEIGTH):
#             if (w == x and y==h):
#                 print(vx+10,",",vy+10,",",i,file=f)
#             else:
#                 print(255,",",255,",",255,file=f)
#                # print(x,",",y,",", vx,",",vy,",",i,file=f)
#-----------

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

filedir = r'E:\code\scenarioagentcnn\scenarioData7\base'+'\\'
minx_total,miny_total = minPosition(filedir,1509)    

egox,egoy,egovx,egovy = addTrajectory(minx_total,miny_total,df,0,True)
print('egox:',egox)
print('egoy:',egoy)

su1x,su1y,su1vx,su1vy = addTrajectory(minx_total,miny_total,df,9,False)
print('su1x:',su1x)
print('su1y:',su1y)
print(len(su1x))

su2x,su2y,su2vx,su2vy = addTrajectory(minx_total,miny_total,df,18,False)
print('#----------------')
print('su2x:',su2x)
print('su2y:',su2y)
print('su2vy:',su2vy)
print('su2vx:',su2vx)


k1,k2,k3,douTra = 0,0,0,0
#k2=0
for h in range(HEIGTH):
    for w in range(WIDTH):
        if (w == egox[k1] and h == egoy[k1]):
            print(egovx[k1],",",egovy[k1],",",k1+1,file=f)
            # print(egovx[k1],",",egovy[k1],",",k1+1)
            # w+=1
            douTra = 1
            print(egovx[k1],",",egovy[k1],",",k1+1)
            # print('wego:',w,'hego:',h)
            if k1 <total_row-1:
                k1+=1
                while (egox[k1-1] == su1x[k2] and egoy[k1-1] == su1y[k2]):
                    k2+=1
                    print('k2:',k2)
                # while  (egovx[k1-1] == egovx[k1] and egovy[k1-1] == egovy[k1]): # 跳过同一个位置点，需不需要更新速度？需要对自车和车一也改？
                #     k1+=1
        elif(w == su1x[k2] and h == su1y[k2]):
            print(su1vx[k2],",",su1vy[k2],",",k2+1,file=f)
            douTra = 2
            print('su1vx:',su1vx[k2],",",su1vy[k2],",",k2+1)
            print('su1x:',su1x[k2],",",su1y[k2],",",k2+1)
            if k2 <total_row-1:
                k2+=1
                print('w1:',w,'h1:',h)
                # while  (su1x[k2-1] == su1x[k2] and su1y[k2-1] == su1y[k2]): # 跳过同一个位置点，需不需要更新速度？
                #     k2+=1
        elif(w == su2x[k3] and h == su2y[k3]):
            print(su2vx[k3],",",su2vy[k3],",",k3+1,file=f)
            douTra = 3
            print('su2vxpixel:',su2vx[k3],",",su2vy[k3],",",k3+1)
            print('w2:',w,'h2:',h)
            if k3 <total_row-1: 
                k3+=1
                while  (su2x[k3-1] == su2x[k3] and su2y[k3-1] == su2y[k3]): # 跳过同一个位置点，需不需要更新速度？
                    k3+=1
        elif(douTra==1):
            print(egovx[k1],",",egovy[k1],",",k1+1,file=f)   
            douTra = 0
        elif(douTra==2):
            print(su1vx[k2],",",su1vy[k2],",",k2+1,file=f)
            douTra = 0
        elif(douTra==3):
            print(su2vx[k3],",",su2vy[k3],",",k3+1,file=f)
            douTra = 0
        else:
            print(255,",",255,",",255,file=f)
            douTra = 0


'''
txt转至图片
'''

