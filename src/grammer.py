# import tensorflow as tf
# mytuple = ("c", "python", "java")

# myit = iter(mytuple) # 用iter()方法创建了一个iterator对象

# print(next(myit))
# print(next(myit))
# print(next(myit))
# print(myit)

# for x in myit:
#     print(x)
# a = tf.constant([1,2,3])
# print(a)
# # c = tf.expand_dims(a,[2,3])
# # print(c)

'''/**
!/usr/bin/env tensorflow
# -*- coding: utf-8 -*-
 * Copyright © 2019 Jianfeng_Wu. All rights reserved.
 * 
 * @author: Jianfeng_Wu
 * @date: 2022-04-23 
 * @time: 21:53:50 
 * Version:1.0.0
 */'''
# from mimetypes import init
# from sklearn.decomposition import KernelPCA
# import tensorflow as tf

# class student(object):
#     def __init__(self,name =12 ,age=175,*args,**kwargs) -> None:
#         self.name = name
#         self.age = age
#         print(self.name,self.age)
#         print(args,kwargs)
#     def __call__(self,*args,**kwargs):
#         print('my friend is',args)
#         print('my friends age is',kwargs)

# stu1 = student('Good Student',13,174,143,ti = 1, ha =2) # 实例化为对象stu1
# stu1('tim','harry',tim = 1, harry =2)  # stu1这时候可以当作函数来用，输入参数之后，默认执行类中的_call_方法

# '''/**
# !/usr/bin/env tensorflow
# # -*- coding: utf-8 -*-
#  * Copyright © 2019 Jianfeng_Wu. All rights reserved.
#  * 
#  * @author: Jianfeng_Wu
#  * @date: 2022-04-29 
#  * @time: 21:36:31 
#  * Version:1.0.0
#  * description:打印输出tensor的值
#  */'''
# from operator import le
# import tensorflow as tf
# x = tf.constant([[1., 1.], [2., 2.]])
# a = tf.reduce_sum(x,reduction_indices=[0]) # reduction_indices计算tensor指定轴方向上的所有元素的累加和;

# c = [[1,0],[0,1],[1,0],[0,1]]

# print(len(c))

# # print只能打印输出shape的信息，而要打印输出tensor的值，需要借助class tf.Session, class tf.InteractiveSession。
# # 因为我们在建立graph的时候，只建立tensor的结构形状信息，并没有执行数据的操作。
# with tf.Session() as sess:
#     print(a)
#     a = sess.run(a)
#     print(a)
#     print(len(a))

# '''/**
# !/usr/bin/env tensorflow
# # -*- coding: utf-8 -*-
#  * Copyright © 2019 Jianfeng_Wu. All rights reserved.
#  * 
#  * @author: Jianfeng_Wu
#  * @date: 2022-04-30 
#  * @time: 12:18:45 
#  * Version:1.0.0
#  * description:生成随机数
#  */'''

# import random

# k1,k3 = 0,0
# for i in range(100):
#     Uniform = random.uniform(0,9)
#     # print(Uniform)    
#     if Uniform<1.4:
#         k1 += 1
#         print(Uniform)
#     elif Uniform<2:
#         k3 += 1
#         print("0.3")
# print("小于1.4:",k1,'小于2:',k3)
    
# '''/**
# !/usr/bin/env tensorflow
# # -*- coding: utf-8 -*-
#  * Copyright © 2019 Jianfeng_Wu. All rights reserved.
#  * 
#  * @author: Jianfeng_Wu
#  * @date: 2022-04-30 
#  * @time: 20:20:32 
#  * Version:1.0.0
#  * description:用tensorflow构建数据集
#  */'''

# import tensorflow as tf
# import tensorflow.contrib.eager as tfe
# import os
# # tf.enable_eager_execution()

# import numpy as np



# classes={'1','2'}  #人为设定2类

# cwd='./PicClassTest/'
# data = []
# label = []
# for index,name in enumerate(classes):
#     file_path = cwd+name+'/'
#     # file_path = r'./PicClassTest/'
#     # print(os.listdir(file_path)) #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。['1', '2']

#     data= [os.path.join(file_path,i) for i in os.listdir(file_path)] # os.path.join()函数用于路径拼接文件路径，可以传入多个路径
#     label = [int(name)]*np.ones(len(data))
#     # print(data)
#     # print(label)
#     dataset = tf.data.Dataset.from_tensor_slices((data,label))
#     print(dataset)

#     iterator = dataset.make_one_shot_iterator()
#     img_name, label = iterator.get_next()

#     with tf.Session() as sess:
#         while 1:
#             try:
#                 name, num = sess.run([img_name,label])
#                 print(name,num)
#                 assert num != 0, "fail to read label"
#             except tf.errors.OutOfRangeError:
#                 print("iterator done")
#                 break




# #  <DatasetV1Adapter shapes: ((), ()), types: (tf.string, tf.int32)>


#     #['E:\\dataset\\DAVIS\\JPEGImages\\480p\\bear\\00000.jpg', 'E:\\dataset\\DAVIS\\JPEGImages\\480p\\bear\\00001.jpg', ......]
#     #82


# # data = [1,2,3]
# # data.extend([1,3,54])
# # print(data)


# from requests import Session, session
# import tensorflow as tf

# filenames = tf.placeholder(tf.string, shape=[None])
# dataset = tf.data.TFRecordDataset(filenames)
# #如何将数据解析（parse）为Tensor见 3.1 节
# dataset = dataset.map(...)  # Parse the record into tensors.
# dataset = dataset.repeat()  # Repeat the input indefinitely.
# dataset = dataset.batch(32)
# iterator = dataset.make_initializable_iterator()

# # You can feed the initializer with the appropriate filenames for the current
# # phase of execution, e.g. training vs. validation.
# with tf.Session() as sess:
#     # Initialize `iterator` with training data.
#     training_filenames = "123train.tfrecords"
#     sess.run(iterator.initializer, feed_dict={filenames: training_filenames})

#     # Initialize `iterator` with validation data.
#     validation_filenames = "123test.tfrecords"
#     sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})


# a = 500
# c = 7*a/2600
# print(type(c))


# import numpy as np

# def one_hot(labels,Label_class):
#     one_hot_label = np.array([[int(i == int(labels[j])) for i in range(Label_class)] for j in range(len(labels))])   
#     return one_hot_label

# a = [1,2,1,1]
# print(one_hot(a,2))
# # print(one_hot(2,2))



# import os
# import shutil
# def setDir(filepath):
#     '''
#     如果文件夹不存在就创建，如果文件存在就清空！
#     :param filepath:需要创建的文件夹路径
#     :return:
#     '''
#     if not os.path.exists(filepath):
#         os.mkdir(filepath)
#     else:
#         shutil.rmtree(filepath)  # 递归删除filepath目录的内容
#         os.mkdir(filepath)

# setDir('PicClassTrain\\2')



# import tensorflow as tf 
# tf.data.Dataset.folder.

# import pandas as pd
# def minPosition(filedir:str,num_total:int):
#     minx_array,miny_array=[],[]

#     for i in range(0,num_total,1):
#         # file_path = r'E:\code\scenarioagentcnn\scenarioData2\LK\%s' % (i+1) + '-LK.csv'
#         file_path = filedir+'%s' % (i+1) + '-LK.csv'
#         df = pd.read_csv(file_path,header=None)
#         miny1 = min(df.iloc[:, 1])
#         miny_array.append(miny1)
#         miny2 = min(df.iloc[:, 10])
#         miny_array.append(miny2)
#         miny3 = min(df.iloc[:, 19])
#         miny_array.append(miny3)

#         minx1 = min(df.iloc[:, 0])
#         minx_array.append(minx1)
#         minx2 = min(df.iloc[:, 9])
#         minx_array.append(minx2)
#         minx3 = min(df.iloc[:, 18])
#         minx_array.append(minx3)

#     minx_total = min(minx_array)
#     miny_total = min(miny_array)
#     return minx_total,miny_total

# filedir = r'E:\code\scenarioagentcnn\scenarioData2\LK'+'\\'
# minx_total,miny_total = minPosition(filedir,2883)   
# print(minx_total,miny_total)

# import imp
# import pandas as pd
# # file_path = r'E:\code\scenarioagentcnn\scenarioData2\LK\%s' % (i+1) + '-LK.csv'
# file_path = r'E:\code\scenarioagentcnn\scenarioData2\LK\1-LK.csv'
# file_path2 = r'E:\code\scenarioagentcnn\scenarioData2\base\1.csv'

# df = pd.read_csv(file_path,header=None)
# # data.iloc[0,:] = data.iloc[1,:]
# print(min(df.iloc[:,0]))
# df[[0, 1]] = df[[1, 0]]
# df[[9, 10]] = df[[10, 9]]
# df[[18, 19]] = df[[19, 18]]
# df.to_csv(file_path2,header=None,index=None)
# df.loc[:, [0, 1]] = df.loc[:, [1,0]]

# print(min(df.iloc[:,0]))
# data.to_csv()
# print(egovx[k1],",",egovy[k1],",",k1+1,file=filepath)


# import pandas as pd
# def filePre(filedir:str,filedir2:str,num_total:int):
#     for i in range(0,num_total,1):
#         # file_path = r'E:\code\scenarioagentcnn\scenarioData2\LK\%s' % (i+1) + '-LK.csv'
#         file_path = filedir+'%s' % (i+1) + '-LK.csv'
#         file_path2 = filedir2+'%s' % (i+1) + '.csv'
#         df = pd.read_csv(file_path,header=None)
#         print(min(df.iloc[:,0]))
#         df[[0, 1]] = df[[1, 0]]
#         df[[9, 10]] = df[[10, 9]]
#         df[[18, 19]] = df[[19, 18]]
#         df.to_csv(file_path2,header=None,index=None)

#     # minx_total = min(minx_array)
#     # miny_total = min(miny_array)
#     # return minx_total,miny_total

# filedir = r'E:\code\scenarioagentcnn\scenarioData2\LK'+'\\'
# filedir2 = r'E:\code\scenarioagentcnn\scenarioData2\base'+'\\'
# filePre(filedir,filedir2,10)

# import os
# path = r'E:\code\scenarioagentcnn\scenarioData5\base'+'\\'     # 输入文件夹地址
# files = os.listdir(path)   # 读入文件夹
# num_png = len(files)       # 统计文件夹中的文件个数
# print(num_png)             # 打印文件个数
# print(num_png,'个RGB文件构建完成！写在'+path+'位置')
# # 输出所有文件名
# # print("所有文件名:")
# # for file in files:
# #     print(file)

# import sklearn
# y_true = [1,1,1,1]
# y_pred = [0,1,1,0]
# sklearn.metrics.confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)

# import numpy as np
# import tensorflow as tf
# t = []

# from sklearn.metrics import confusion_matrix
# y_true = [[1, 0], [0, 1],[1, 0],[0, 1]]
# y_pred = [[1, 0], [1, 0],[1, 0],[0, 1]]
# y_true = [1,0,1,0]
# y_pred = [0,1,1,1]
# # y_pred.extend(y_true)
# # print(y_pred)
# y1=[[1,0],[0,1]]
# t = tf.confusion_matrix(y_true, y_pred)
# with tf.Session() as sess:
#     t1 = sess.run(t)
#     t2 = sess.run(t)
#     t = t1 +t2
#     print(t,t[1,1])

# print(np.argmax(y1,1))

# epoch = 12
# for i in range(epoch):
#     epoch +=1

# print (epoch)

# test = [1,2,3,3]
# print(max(test))
# print(min(test))


            # elif (w == su2x[k3]-1 and h == su2y[k3]-1 and k_stop2 == 1):
            #     print(r_num_ego,",",r_num_ego,",",(k3+1),file=filepath)
            #     print(r_num_ego,",",r_num_ego,",",(k3+1))
            # elif (w == su2x[k3]-1 and h == su2y[k3] and k_stop2 == 1):
            #     print(r_num_ego,",",r_num_ego,",",(k3+1),file=filepath)
            # elif (w == su2x[k3]-1 and h == su2y[k3]+1 and k_stop2 == 1):
            #     print(r_num_ego,",",r_num_ego,",",(k3+1),file=filepath)
            # elif (w == su2x[k3] and h == su2y[k3]-1 and k_stop2 == 1):
            #     print(r_num_ego,",",r_num_ego,",",(k3+1),file=filepath)
            # elif (w == su2x[k3] and h == su2y[k3] and k_stop2 == 1):
            #     print(r_num_ego,",",r_num_ego,",",(k3+1),file=filepath)
            # elif (w == su2x[k3] and h == su2y[k3]+1 and k_stop2 == 1):
            #     print(r_num_ego,",",r_num_ego,",",(k3+1),file=filepath)
            # elif (w == su2x[k3]+1 and h == su2y[k3]-1 and k_stop2 == 1):
            #     print(r_num_ego,",",r_num_ego,",",(k3+1),file=filepath)
            # elif (w == su2x[k3]+1 and h == su2y[k3] and k_stop2 == 1):
            #     print(r_num_ego,",",r_num_ego,",",(k3+1),file=filepath)
            # elif (w == su2x[k3]+1 and h == su2y[k3]+1 and k_stop2 == 1):
            #     print(r_num_ego,",",r_num_ego,",",(k3+1),file=filepath)

            # elif (w == egox[k1]-1 and h == egoy[k1]-1 and k1<2):
            #     print(r_num_ego,",",r_num_ego,",",(k2+1),file=filepath)
            #     print(r_num_ego,",",r_num_ego,",",(k2+1))
            # elif (w == egox[k1]-1 and h == egoy[k1] and k1<2):
            #     print(r_num_ego,",",r_num_ego,",",(k2+1),file=filepath)
            # elif (w == egox[k1]-1 and h == egoy[k1]+1 and k1<2):
            #     print(r_num_ego,",",r_num_ego,",",(k2+1),file=filepath)
            # elif (w == egox[k1] and h == egoy[k1]-1 and k1<2):
            #     print(r_num_ego,",",r_num_ego,",",(k2+1),file=filepath)
            # elif (w == egox[k1] and h == egoy[k1] and k1<2):
            #     print(r_num_ego,",",r_num_ego,",",(k2+1),file=filepath)
            # elif (w == egox[k1] and h == egoy[k1]+1 and k1<2):
            #     print(r_num_ego,",",r_num_ego,",",(k2+1),file=filepath)
            # elif (w == egox[k1]+1 and h == egoy[k1]-1 and k1<2):
            #     print(r_num_ego,",",r_num_ego,",",(k2+1),file=filepath)
            # elif (w == egox[k1]+1 and h == egoy[k1] and k1<2):
            #     print(r_num_ego,",",r_num_ego,",",(k2+1),file=filepath)
            # elif (w == egox[k1]+1 and h == egoy[k1]+1 and k1<2):
            #     print(r_num_ego,",",r_num_ego,",",(k2+1),file=filepath)


            # elif (w == su1x[k2]-1 and h == su1y[k2]-1 and k2<2):
            #     print(r_num_ego,",",r_num_ego,",",(k2+1),file=filepath)
            #     print(r_num_ego,",",r_num_ego,",",(k2+1))
            # elif (w == su1x[k2]-1 and h == su1y[k2] and k2<2):
            #     print(r_num_ego,",",r_num_ego,",",(k2+1),file=filepath)
            # elif (w == su1x[k2]-1 and h == su1y[k2]+1 and k2<2):
            #     print(r_num_ego,",",r_num_ego,",",(k2+1),file=filepath)
            # elif (w == su1x[k2] and h == su1y[k2]-1 and k2<2):
            #     print(r_num_ego,",",r_num_ego,",",(k2+1),file=filepath)
            # elif (w == su1x[k2] and h == su1y[k2] and k2<2):
            #     print(r_num_ego,",",r_num_ego,",",(k2+1),file=filepath)
            # elif (w == su1x[k2] and h == su1y[k2]+1 and k2<2):
            #     print(r_num_ego,",",r_num_ego,",",(k2+1),file=filepath)
            # elif (w == su1x[k2]+1 and h == su1y[k2]-1 and k2<2):
            #     print(r_num_ego,",",r_num_ego,",",(k2+1),file=filepath)
            # elif (w == su1x[k2]+1 and h == su1y[k2] and k2<2):
            #     print(r_num_ego,",",r_num_ego,",",(k2+1),file=filepath)
            # elif (w == su1x[k2]+1 and h == su1y[k2]+1 and k2<2):
            #     print(r_num_ego,",",r_num_ego,",",(k2+1),file=filepath)


# outputs, end_points = vgg.all_cnn(Xinputs,
#                                           num_classes=num_classes,
#                                           is_training=True,
#                                           dropout_keep_prob=0.5,
#                                           spatial_squeeze=True,
#                                           scope='all_cnn'
 
#     cross_entrys=tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=Yinputs)
#     # w_temp = tf.matmul(Yinputs, w_ls) #代价敏感因子w_ls=tf.Variable(np.array(w,dtype='float32')，name="w_ls",trainable=False)，w是权重项链表
#     # loss=tf.reduce_mean(tf.multiply(cross_entrys,w_temp))  #代价敏感下的交叉熵损失

# import tensorflow as tf
# import numpy as np

# def test_cost():
#     # tf.nn.sigmoid_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, name=None)
#     labels = np.array(np.random.randint(0, 10, (10, 2)), dtype=np.float32)
#     logits = np.array(np.random.normal(1, 20, (10, 2)), dtype=np.float32)
#     # pos = np.array(np.random.normal(1, 20, (2,)), dtype=np.float32)
#     res = tf.nn.weighted_cross_entropy_with_logits(targets=labels, logits=logits, pos_weight=2.0)
#     return res


# with tf.Session()as sess:
#     c = test_cost()
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(c))


# import tensorflow as tf
# from tensorflow.python.ops import array_ops

# def binary_focal_loss(target_tensor,prediction_tensor, alpha=0.25, gamma=2):
#     zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)
#     target_tensor = tf.cast(target_tensor,prediction_tensor.dtype)
#     pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - prediction_tensor, zeros)
#     neg_p_sub = array_ops.where(target_tensor > zeros, zeros, prediction_tensor)
#     per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.math.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0)) \
#                           - (1 - alpha) * (neg_p_sub ** gamma) * tf.math.log(tf.clip_by_value(1.0 - prediction_tensor, 1e-8, 1.0))
#     return tf.math.reduce_sum(per_entry_cross_ent)




# def binary_focal_loss_fixed(n_classes, logits, y_true,gamma=2, alpha=0.25):
#     alpha = tf.constant(alpha, dtype=tf.float32)
#     gamma = tf.constant(gamma, dtype=tf.float32)
#     epsilon = 1.e-8
#     # 得到y_true和y_pred
#     # y_true = tf.one_hot(true_label, n_classes)
#     probs = tf.nn.sigmoid(logits)
#     y_pred = tf.clip_by_value(probs, epsilon, 1. - epsilon)
#     # 得到调节因子weight和alpha
#     ## 先得到y_true和1-y_true的概率【这里是正负样本的概率都要计算哦！】
#     p_t = y_true * y_pred \
#             + (tf.ones_like(y_true) - y_true) * (tf.ones_like(y_true) - y_pred)
#     ## 然后通过p_t和gamma得到weight
#     weight = tf.pow((tf.ones_like(y_true) - p_t), gamma)
#     ## 再得到alpha，y_true的是alpha，那么1-y_true的是1-alpha
#     alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
#     # 最后就是论文中的公式，相当于：- alpha * (1-p_t)^gamma * log(p_t)
#     focal_loss = - alpha_t * weight * tf.log(p_t)
#     return tf.reduce_mean(focal_loss)

# def binary_focal_loss(gamma=2, alpha=0.25):
#     alpha = tf.constant(alpha, dtype=tf.float32)
#     gamma = tf.constant(gamma, dtype=tf.float32)
#     def binary_focal_loss_fixed(n_classes, logits, true_label):
#         epsilon = 1.e-8
#         # 得到y_true和y_pred
#         y_true = tf.one_hot(true_label, n_classes)
#         probs = tf.nn.sigmoid(logits)
#         y_pred = tf.clip_by_value(probs, epsilon, 1. - epsilon)
#         # 得到调节因子weight和alpha
#         ## 先得到y_true和1-y_true的概率【这里是正负样本的概率都要计算哦！】
#         p_t = y_true * y_pred \
#               + (tf.ones_like(y_true) - y_true) * (tf.ones_like(y_true) - y_pred)
#         ## 然后通过p_t和gamma得到weight
#         weight = tf.pow((tf.ones_like(y_true) - p_t), gamma)
#         ## 再得到alpha，y_true的是alpha，那么1-y_true的是1-alpha
#         alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
#         # 最后就是论文中的公式，相当于：- alpha * (1-p_t)^gamma * log(p_t)
#         focal_loss = - alpha_t * weight * tf.log(p_t)
#         return tf.reduce_mean(focal_loss)

# t = 1.5
# print(int(t))

#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
#
# 
# @param nums int整型一维数组 
# @param target int整型 
# @return int整型
#



# class Solution:
#     def search(self , nums , target ):
    
#         left = 0
#         right = len(nums)-1
#         while left<=right:
#             middle = int((left+right)/2)
#             print(middle)
#             if  nums[middle] == target:
#                     return middle
#             elif nums[middle] < target:
#                     left = middle+1 # nums[middle]!=target, 直接跳过nums[middle]
#             elif nums[middle] > target:
#                     right = middle-1  
#         return -1
        
#         # write code here
# s = Solution()
# t = s.search([],13)
# print(t)


from lib2to3.pytree import Node


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        

class Solution:
    l = []
    def preorderTraversal(self , root: TreeNode):
        # write code here
        if root == None:
            return 
        self.l.append(root.val)
        self.preorderTraversal(root.left)
        self.preorderTraversal(root.right)
        return self.l


def  test():
        treenode1 = TreeNode(1)
        treenode2 = TreeNode(3)
        treenode3 = TreeNode(11)
        treenode4 = TreeNode(21)
        
        treenode1.left = treenode2
        treenode1.right = treenode3
        treenode3.left = treenode4
        
        return treenode1



treenode = test()



s = Solution()
l = s.preorderTraversal(treenode)
print(l)
