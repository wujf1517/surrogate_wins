import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
'''
这个函数根据minRL值对图片进行分类
'''

# file_path = r'E:\code\scenarioagentcnn\scnarioData\计算结果\\minResult.csv'
# file_path = r'E:\code\scenarioagentcnn\scenarioData2\LK_RESULT\\minResult.csv'
# file_path = r'E:\code\scenarioagentcnn\scenarioData4\计算结果\\minResult.csv'
# file_path = r'E:\code\scenarioagentcnn\scenarioData4\SO1213_RESULT\\SO1213_minResult.csv'

# df = pd.read_csv(file_path,header=None) # 加上header=None，否则默认第一行为标题
# print(df.iloc[:,1])

# def RL2class(i):
#     if df.iloc[i,1]<0.2:
#         return 1
#     elif df.iloc[i,1]<0.4:
#         return 2
#     elif df.iloc[i,1]<0.6:
#         return 3
#     elif df.iloc[i,1]<0.8:
#         return 4
#     else:
#         return 5

'''
上面设置五类，找不到足够的文件，这里设置成三类
'''

# def RL2class(i):
#     if df.iloc[i,1]<0.3:
#         return 1
#     elif df.iloc[i,1]<0.7:
#         return 2
#     else:
#         return 3

def setDir(filepath):
    '''
    如果文件夹不存在就创建，如果文件存在就清空！
    :param filepath:需要创建的文件夹路径
    :return:
    '''
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)  # 递归删除filepath目录的内容
        os.mkdir(filepath)


def RL2class(i,df: pd.DataFrame()):
    col = df.shape[1]
    if df.iloc[i,col-1]<0.7: # 0.7
        return 1
    else:
        return 2

if __name__=='__main__':
    '''
    测试
    '''
    file_res_path = r'data\FV_dec_35_5\RL\res.csv'

    df1 = pd.read_csv(file_res_path) # 加上header=None，否则默认第一行为标题
    print(df1.shape[1],df1.iloc[2,df1.shape[1]-1])
    print(df1.shape[0])
    # print(RL2class(0))
    # print(RL2class(10))
    # print(RL2class(11))
    # print(RL2class(24))
 
