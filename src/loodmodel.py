# #****************** test with network.py*********************
'''/**
!/usr/bin/env tensorflow
# -*- coding: utf-8 -*-
 * Copyright © 2019 Jianfeng_Wu. All rights reserved.
 * 
 * @author: Jianfeng_Wu
 * @date: 2022-05-02 
 * @time: 11:14:17 
 * Version:1.0.0
 * description:不用重新构造一遍模型，直接恢复,原因如下：
    # 通过meta文件，加载模型结构，返回的是一个saver对象
    saver = tf.train.import_meta_graph('saver/save_net.meta')
 */'''

import tensorflow as tf
import ReadMyownData
import numpy as np
img_test, label_test = ReadMyownData.read_and_decode("123test.tfrecords")
batch_size = 32
num_classes= 2

def one_hot(labels,Label_class):
    one_hot_label = np.array([[int(i == int(labels[j])) for i in range(Label_class)] for j in range(len(labels))])   
    return one_hot_label

# 这里采取顺寻读取的结构，与训练时不一样
# tf.train.batch() 按顺序读取队列中的数据
# 队列中的数据始终是一个有序的队列．队头一直按顺序补充，队尾一直按顺序出队
img_test, label_test = tf.train.batch([img_test, label_test],
                                                batch_size=batch_size, capacity=2000)

with tf.Session() as sess: #with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    
    #在Session当中,没有启动数据读入线程。所以,sess.run(train_input.input_data)就是无数据可取,程序就处于一种挂起的状态。
    coord = tf.train.Coordinator() # 多线程 ###不加多线程会挂起：
    threads=tf.train.start_queue_runners(sess=sess,coord=coord) 

    # 通过meta文件，加载模型结构，返回的是一个saver对象
    saver = tf.train.import_meta_graph('saver/save_net.meta')
    # 载入模型参数
    saver.restore(sess,'saver/save_net')
        
    graph = tf.get_default_graph()   # 获取当前图，为了后续训练时恢复变量
       
    # 获取模型的输入名称
    X = graph.get_tensor_by_name('inputs/x:0')   #从模型中获取输入的那个节点,二级名称
    keep_prob = graph.get_tensor_by_name('inputs/kkep_prob:0')   #从模型中获取输入的那个节点,二级名称
    y_ = graph.get_tensor_by_name('inputs/y_:0')   #从模型中获取输入的那个节点,二级名称
    # 获取模型的输出名称
    model_y = graph.get_tensor_by_name('fc2_layer/y_conv:0')
    c_p = graph.get_tensor_by_name('cross_entropy/c_p:0')
    acc = graph.get_tensor_by_name('cross_entropy/accuracy:0')
    print(X,model_y,c_p,acc)
    # 测试模型
    # img_test, label_test = graph.get_tensor_by_name('image/read_data:0')
    for i in range(10):
        test_x, l_test = sess.run([img_test, label_test])
        l_test = one_hot(l_test,num_classes)
        print('样本标签：',l_test)
        result=sess.run(model_y,feed_dict={X:test_x,keep_prob:1})  # 需要的就是模型预测值model_Y，这里存为result
        # acc= sess.run(acc,feed_dict = {X:test_x,y_:l_test,keep_prob:1})
        # l = one_hot(l,num_classes) # 原来为l = one_hot(l,2)

        print('预测结果：',result)
        print('accurcy:',acc)
    coord.request_stop()
    coord.join(threads)


# 检查操作名称，可以看出有的名称是二级关系的name: "input_producer/Greater/y"
    # graph_op = graph.get_operations()
    # for i in graph_op[:10]:
    #      print(i)

