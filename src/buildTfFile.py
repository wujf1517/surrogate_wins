# -*- coding: utf-8 -*-
"""
@author: caokai
"""
import os 
import tensorflow as tf 
from PIL import Image  
# import matplotlib.pyplot as plt 
import numpy as np

'''
修改cwd 和 writer= tf.io.TFRecordWriter("123test.tfrecords") #要生成的文件
''' 

# cwd='./PicClassTrain/'
# writer= tf.io.TFRecordWriter("123train.tfrecords") #要生成的文件

# cwd = './picclass/'
# cwd='./PicClassTest/'
# writer= tf.io.TFRecordWriter("123test.tfrecords") #要生成的文件

def buildSingleTfrecord():
        cwd='./PicClassTrain/'
        writer= tf.io.TFRecordWriter("123train.tfrecords") #要生成的文件
        #cwd='./data/test/'
        # classes={'dog','cat'}  #人为设定2类
        # classes={'1','2','3'}  #人为设定3类
        classes={'1','2'}  #人为设定2类
        for index,name in enumerate(classes):
            class_path=cwd+name+'/'
            for img_name in os.listdir(class_path): 
                img_path=class_path+img_name #每一个图片的地址
        
                img=Image.open(img_path)
                img= img.resize((128,128)) # 先不看shape??要不要调整
                print(np.shape(img))
                img_raw=img.tobytes()#将图片转化为二进制格式
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                })) #example对象对label和image数据进行封装
                writer.write(example.SerializeToString())  #序列化为字符串        
        writer.close()


def buildTfrecord():
    for i in range(2):
        if i == 0:
            cwd='./PicClassTrain/'
            writer= tf.io.TFRecordWriter("123train.tfrecords") #要生成的文件
        else:
            cwd='./PicClassTest/'
            writer= tf.io.TFRecordWriter("123test.tfrecords") #要生成的文件

        classes={'1','2'}  #人为设定2类

        for index,name in enumerate(classes):
            class_path=cwd+name+'/'
            for img_name in os.listdir(class_path): 
                img_path=class_path+img_name #每一个图片的地址
       
                img=Image.open(img_path)
                img= img.resize((128,128)) # 先不看shape??要不要调整
                print(np.shape(img))
                img_raw=img.tobytes()#将图片转化为二进制格式
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                })) #example对象对label和image数据进行封装
                writer.write(example.SerializeToString())  #序列化为字符串
        
        writer.close()

if __name__=='__main__':
    buildTfrecord()
    # buildSingleTfrecord()
    print('全部转换为tfrecord格式。')