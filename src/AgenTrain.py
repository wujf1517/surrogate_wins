'''
/**
!/usr/bin/env tensorflow
# -*- coding: utf-8 -*-
 * Copyright © 2019 Jianfeng_Wu. All rights reserved.
 * 
 * @author: Jianfeng_Wu
 * @date: 2022-04-22 
 * @time: 12:35:13 
 * Version:1.0.0
 */
'''
"""
训练代理模型
用tensorflow环境来跑
可视化----：tensorboard --logdir=E:\code\scenarioagentcnn\TrainLog
"""
import tensorflow as tf 
import numpy as np
import ReadMyownData
from unicodedata import name

# epoch = 12
# batch_size = 32

epoch = 1
batch_size =32

'''标签三个，原模型标签两个，需要修改吗？'''
def one_hot(labels,Label_class):
    one_hot_label = np.array([[int(i == int(labels[j])) for i in range(Label_class)] for j in range(len(labels))])   
    return one_hot_label

'''
[[int(i == int(labels[j])) 
for i in range(Label_class)] 
for j in range(len(labels))]
'''

#initial weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1) # 正态分布抽样初始化(shape, stddev = 0.02)
    return tf.Variable(initial,name='weight')
#initial bias
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)  # 原来为tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

#convolution layer
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

#max_pool layer
def max_pool_4x4(x):
    return tf.nn.max_pool(x, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')

with tf.name_scope("inputs"):
    # x = tf.placeholder(tf.float32, [batch_size,128,128,3])
    # y_ = tf.placeholder(tf.float32, [batch_size,2])
    keep_prob = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.float32, [batch_size,128,128,3])/255.
    y_ = tf.placeholder(tf.float32, [batch_size,3]) # 原来为2 y_ = tf.placeholder(tf.float32, [batch_size,2])

with tf.name_scope("conv1_layer"):
    #first convolution and max_pool layer
    W_conv1 = weight_variable([5,5,3,32])
    # variable_summaries('w1',W_conv1)
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_4x4(h_conv1)

with tf.name_scope('conv2_layer'):
    #second convolution and max_pool layer
    W_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_4x4(h_conv2)

with tf.name_scope('fc1_layer'):
    #变成全连接层，用一个MLP处理
    reshape = tf.reshape(h_pool2,[batch_size, -1])
    dim = reshape.get_shape()[1].value
    W_fc1 = weight_variable([dim, 1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(reshape, W_fc1) + b_fc1)

    #dropout
    # keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('fc2_layer'):
    W_fc2 = weight_variable([1024,3]) # 分类数目改了，这里权重也得改由2改为3
    b_fc2 = bias_variable([3])  # 分类数目改了，这里偏置量也得改由2改为3
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

with tf.name_scope('cross_entropy'):
    #损失函数及优化算法
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]),name='cross_entropy')
    tf.summary.scalar('cross_entropy',cross_entropy) # 只能紧跟其后cross_entropy后面，不可以跨行？
    train_step = tf.train.AdamOptimizer(0.00001).minimize(cross_entropy) # 修改学习率，原来是0.001
    #tf.summary.histogram('cross_entropy:', cross_entropy)#在tensorboard中的 
    correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    tf.summary.scalar('accuracy:', accuracy)#在tensorboard中的 

with tf.name_scope('image_input'):
    img, label = ReadMyownData.read_and_decode("123train.tfrecords")
    img_test, label_test = ReadMyownData.read_and_decode("123test.tfrecords")
    # img, label = ReadMyownData.read_and_decode(".\dataRecord\successful\\123train.tfrecords")
    # img_test, label_test = ReadMyownData.read_and_decode(".\dataRecord\successful\\123test.tfrecords")
    # img, label = ReadMyownData.read_and_decode(".\dataRecord\\123train.tfrecords")E:\code\scenarioagentcnn\dataRecord\successful
    # img_test, label_test = ReadMyownData.read_and_decode(".\dataRecord\\123test.tfrecords") # 0.90625000

# sess = tf.Session()
# merged = tf.summary.merge_all()                                 # 合并所有,所有summary都保存在日志中，以便tensorboard进行显示。Merges all summaries collected in the default graph.
# writer = tf.summary.FileWriter("CNN_TEST",sess.graph)           # 把前面的全部信息收集起来，放入文件，最终生成可视化，参数为路径，graph是全部的框架


with tf.name_scope('in'):
    #使用shuffle_batch可以随机打乱输入
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=batch_size, capacity=2000,
                                                    min_after_dequeue=1000)
    img_test, label_test = tf.train.shuffle_batch([img_test, label_test],
                                                    batch_size=batch_size, capacity=2000,
                                                    min_after_dequeue=1000)

init = tf.initialize_all_variables()
t_vars = tf.trainable_variables()
print(t_vars)

with tf.Session() as sess:
    merged = tf.summary.merge_all()                         # 合并所有,所有summary都保存在日志中，以便tensorboard进行显示。Merges all summaries collected in the default graph.
            
    writer = tf.summary.FileWriter("TrainLog",sess.graph)   # 把前面的全部信息收集起来，放入文件，最终生成可视化，参数为路径，graph是全部的框架
    # merged = tf.summary.merge_all()                       # 合并所有,所有summary都保存在日志中，以便tensorboard进行显示。Merges all summaries collected in the default graph.
    # summary = sess.run(merged)
    sess.run(init) # 变量初始化
    coord = tf.train.Coordinator() # 多线程
    threads=tf.train.start_queue_runners(sess=sess,coord=coord) 
    # batch_idxs = int(2314/batch_size)
    batch_idxs = int(800/batch_size) # 设置迭代次数

    for i in range(epoch):
        for j in range(batch_idxs):
            val, l = sess.run([img_batch, label_batch])
            l = one_hot(l,3) # 原来为2分类 l = one_hot(l,2)
            # result = sess.run(merged, feed_dict={x: val, y_: l}) ###########
            # writer.add_summary(result,j)
            _, acc,cro = sess.run([train_step, accuracy,cross_entropy], feed_dict={x: val, y_: l, keep_prob: 0.6})
            print("Epoch:[%4d] [%4d/%4d], accuracy:[%.8f],cross_entropy:[%.8f]" % (i, j, batch_idxs, acc, cro) )
#--------------------------------------------------
            merged_result = sess.run(merged,
                                feed_dict={x: val, y_: l, keep_prob: 1}) # merged只有run了才有用""很重要""",不知道为什么非要有keep_pro
            writer.add_summary(merged_result,i*batch_idxs+j)
#--------------------------------------------------
            #   accuracy_result_sum = tf.Summary(value=[tf.Summary.Value(tag="accuracy", simple_value=acc)]) # 转换为tf.summary对象
            #   writer.add_summary(accuracy_result_sum,i*batch_size+j)         # 将上面的accuracy_result_sum加入summary可视化中，每五十步合并的结果都加入
            # 保存和提取参数
            '''记录训练时的准确性'''
            val_test, l_test = sess.run([img_test, label_test])
            l_test = one_hot(l_test,3) # 原来为l = one_hot(l,2) # one_hot的输入是label
            y_test, acc_test,cross_entropy_test = sess.run([y_conv,accuracy,cross_entropy], feed_dict={x: val_test, y_: l_test, keep_prob: 1})
            accuracy_result_sum_acc = tf.Summary(value=[tf.Summary.Value(tag="accuracy_test", simple_value=acc_test)]) # 转换为tf.summary对象
            writer.add_summary(accuracy_result_sum_acc,i*batch_idxs+j)
            accuracy_result_sum = tf.Summary(value=[tf.Summary.Value(tag="cross_entropy_test", simple_value=cross_entropy_test)]) # 转换为tf.summary对象
            writer.add_summary(accuracy_result_sum,i*batch_idxs+j)

    val, l = sess.run([img_test, label_test])
    # val, l = sess.run([img_batch, label_batch])

    l = one_hot(l,3) # 原来为l = one_hot(l,2)
    print(l)
    y, acc = sess.run([y_conv,accuracy], feed_dict={x: val, y_: l, keep_prob: 1})
    print(y)
    print("test accuracy: [%.8f]" % (acc))
    coord.request_stop()
    coord.join(threads)
