'''/**
!/usr/bin/env tensorflow
# -*- coding: utf-8 -*-
 * Copyright © 2019 Jianfeng_Wu. All rights reserved.
 * 
 * @author: Jianfeng_Wu
 * @date: 2022-05-20 
 * @time: 22:32:59 
 * Version:1.0.0
 * description: when i train this model by tfrecord file,some problem had appeared:establish a tfrecord file by same image
 lead to different result; Then i decide to train by image directly instead of by tfrecord file; 
 */'''

# from warnings import filters
import tensorflow as tf 
import numpy as np
from CoordConv import AddCoords, CoordConv
import ReadMyownData
from unicodedata import name
from RL2class import setDir
import os

# define args
batch_size =32
num_classes = 2             # 分类数目

img_resize = [128,128]
epoch =20                  # dataset.repeat() 的参数，设置为None，可以不断取数
num_examples = 820 
# path = r'picclass'
# path = r'dataRecord\2601\2601img'

path = r'PicClassTrain'
# path = r'dataRecord/2601withoutV'
# path2 = r'picclass'
path2 = r'PicClassTest'

# 清空tensorboard日志保存文件夹
setDir('TrainLog')

# 传入图片名，返回正则化后的图片的像素值
def read_img(img_name, lable):
    image = tf.read_file(img_name)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_images(image, img_resize)
    # image = tf.image.per_image_standardization(image)  #有这个必要吗?这里不需要因为模型训练用的数据没有这一步操作
    return image,lable

# # 传入图片所在的文件夹，图片名含有图片的lable，返回利用文件夹中图片创建的dataset
'''
输入路径下面有两个子文件夹--用分类命名的子文件夹,子文件夹下面是该分类所有图片
'''
def create_dataset(path,batch_size,epoch_num):
    img_names = []
    lables = []
    files = os.listdir(path)  # 返回path下面的子文件夹(分类文件夹)
    for i in range(len(files)):
        files_next = os.listdir(path+'/'+files[i])
        for j in range(len(files_next)):
            img_names.append(os.path.join(path,files[i],files_next[j]))     # 图片的完整路径append到文件名list中
            lables.append(int(i))                                           # 根据规则得到图片的lable
        # print(files_next)
    # print('img_names:',img_names)
    # print('labels:',lables)

    img_names = tf.convert_to_tensor(img_names, dtype=tf.string)
    lables = tf.convert_to_tensor(lables, dtype=tf.float32)  # 将图片名list和lable的list转换成Tensor类型
    dataset = tf.data.Dataset.from_tensor_slices((img_names,lables))  # 创建dataset，传入的需要是tensor类型
    dataset = dataset.map(read_img)   # 传入read_img函数，将图片名转为像素　　 　　# 将dataset打乱，设置一次获取batch_size条数据 
    dataset = dataset.shuffle(buffer_size=1600).batch(batch_size).repeat(epoch_num)  # shuffle(buffer_size=800) 取800个图片并打乱
    return dataset

dataset = create_dataset(path,batch_size,epoch+15)   # 图片所在的路径为path

iterator = dataset.make_initializable_iterator()
# print(iterator.get_next())
one_element = iterator.get_next()  # 创建dataset是batch_size 为多少这里一次就能获取多少个数据

dataset2 = create_dataset(path2,batch_size,epoch+1)   # 图片所在的路径为path
iterator2 = dataset2.make_initializable_iterator()
# print(iterator.get_next())
one_element2 = iterator2.get_next()  # 创建dataset是batch_size 为多少这里一次就能获取多少个数据

# with tf.Session() as sess:
#     images,l = sess.run(one_element) 
#     print(images,l)

'''/**
!/usr/bin/env tensorflow
# -*- coding: utf-8 -*-
 * Copyright © 2019 Jianfeng_Wu. All rights reserved.
 * 
 * @author: Jianfeng_Wu
 * @date: 2022-05-02 
 * @time: 10:37:54 
 * Version:1.0.0
 * description:加载恢复模型（加载参数就好），但本文的这种方式需要将模型重新构造一遍（把原来模型复制过来就可），
 *             这样避免了找入口（名称不好找）的尴尬问题；
 */'''

'''标签三个，原模型标签两个，需要修改吗？'''
def one_hot(labels,Label_class):
    one_hot_label = np.array([[int(i == int(labels[j])) for i in range(Label_class)] for j in range(len(labels))])   
    return one_hot_label

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
   
    x = tf.placeholder(tf.float32, [batch_size,128,128,3],name="x")#/255.
    y_ = tf.placeholder(tf.float32, [batch_size,num_classes],name="y_") # 原来为2 y_ = tf.placeholder(tf.float32, [batch_size,2])
    keep_prob = tf.placeholder(tf.float32,name='kkep_prob') # 放前面出错了
with tf.name_scope("coordConv_layer"):
     h_conv = CoordConv(128,128,False,32,(5,5)) # 通道数呢？卷积是在所有通道上进行运算的，只需指定两个维度如3*3或5*5
     h_conv1 = h_conv(x)
    #  h_conv1 = CoordC1(x,32,(5,5,5))
     h_pool1 = max_pool_4x4(h_conv1)
    # AddCRD = AddCoords(x_dim=128,y_dim=128,with_r=False) # 16384,16384
    # coordtensor = AddCRD(x)


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

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('fc2_layer'):
    W_fc2 = weight_variable([1024,num_classes]) # 分类数目改了，这里权重也得改由2改为3
    b_fc2 = bias_variable([num_classes])  # 分类数目改了，这里偏置量也得改由2改为3
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2,name='y_conv') # 输出概率

with tf.name_scope('cross_entropy'):
    #损失函数及优化算法
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]),name='cross_entropy') 
    
    #tf.reduce_mean计算张量的各个维度上的元素的平均值.reduction_indices计算tensor指定轴方向上的所有元素的累加和;
    
    tf.summary.scalar('cross_entropy',cross_entropy) # 只能紧跟其后cross_entropy后面，不可以跨行？
    train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy) # 修改学习率，原来是0.001
    #tf.summary.histogram('cross_entropy:', cross_entropy)#在tensorboard中的 
    correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1),name='c_p') # tf.argmax(input,axis)根据axis取值的不同返回每行(1)或者每列(0)最大值的索引。
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name='accuracy')
    tf.summary.scalar('accuracy:', accuracy)#在tensorboard中的 


with tf.name_scope('image_input'):
    img, label = ReadMyownData.read_and_decode("123train.tfrecords")
    img_test, label_test = ReadMyownData.read_and_decode("dataRecord\\Random_1600_7_3\\123train.tfrecords")  # 123test.tfrecords

# sess = tf.Session()
# merged = tf.summary.merge_all()                                 # 合并所有,所有summary都保存在日志中，以便tensorboard进行显示。Merges all summaries collected in the default graph.
# writer = tf.summary.FileWriter("CNN_TEST",sess.graph)           # 把前面的全部信息收集起来，放入文件，最终生成可视化，参数为路径，graph是全部的框架

init = tf.initialize_all_variables()
# init = tf.global_variables_initializer() # 用tf.global_variables_initializer()初始化will show 'batch' in tensorboard
t_vars = tf.trainable_variables()
print(t_vars)

saver = tf.train.Saver()

#%% 重载参数的模型
# with tf.Session() as sess:
#     sess.run(init)

#     #在Session当中,没有启动数据读入线程。所以,sess.run(train_input.input_data)就是无数据可取,程序就处于一种挂起的状态。
#     coord = tf.train.Coordinator() # 多线程 ###不加多线程会挂起：
#     threads=tf.train.start_queue_runners(sess=sess,coord=coord) 
#     num_true = 0
#     for i in range(num_loop):
#         val_test, l_test = sess.run(one_element)
#         l_test = one_hot(l_test,num_classes) # 原来为2分类 l = one_hot(l,2) one_hot的输入要从0开始
#         saver.restore(sess, 'saver\save_net')
#         acc, loss = sess.run([accuracy, cross_entropy], feed_dict={x:(val_test*(1. / 255)-0.5), y_:l_test,keep_prob:1})
#         # print(l_test)
#         num_true += acc
#         print(" Loss: " + str(loss) + ", Testing Acc: " + str(acc))
        
#     print("预测正确的:",num_true*batch_size)
#     coord.request_stop()
#     coord.join(threads)

with tf.Session() as sess:
    merged = tf.summary.merge_all()                         # 合并所有,所有summary都保存在日志中，以便tensorboard进行显示。Merges all summaries collected in the default graph.
            
    writer = tf.summary.FileWriter("TrainLog",sess.graph)   # 把前面的全部信息收集起来，放入文件，最终生成可视化，参数为路径，graph是全部的框架
    # merged = tf.summary.merge_all()                       # 合并所有,所有summary都保存在日志中，以便tensorboard进行显示。Merges all summaries collected in the default graph.
    # summary = sess.run(merged)
    sess.run(init) # 变量初始化
    coord = tf.train.Coordinator() # 多线程
    threads=tf.train.start_queue_runners(sess=sess,coord=coord) 
    # batch_idxs = int(2314/batch_size)
    batch_idxs = int(0.8*num_examples/batch_size) # 设置迭代次数 int(0.3*num_examples/batch_size)

    for i in range(epoch):
        sess.run(iterator.make_initializer(dataset)) # We need to intilal iterator before each loop start,which is different from  ..dataset.make_one_shot_iterator()..
        sess.run(iterator2.make_initializer(dataset2))
        for j in range(batch_idxs):
            # val, l = sess.run([img_batch, label_batch])
            # l = one_hot(l,num_classes) # 原来为2分类 l = one_hot(l,2)

            val, l = sess.run(one_element)
            l = one_hot(l,num_classes) # 原来为2分类 l = one_hot(l,2) one_hot的输入要从0开始

            # result = sess.run(merged, feed_dict={x: val, y_: l}) ###########
            # writer.add_summary(result,j)
            _, acc,cro = sess.run([train_step, accuracy, cross_entropy], feed_dict={x: val*(1. / 255)-0.5, y_: l, keep_prob: 0.5})
            print("Epoch:[%4d] [%4d/%4d], accuracy:[%.8f],cross_entropy:[%.8f]" % (i, j, batch_idxs, acc, cro) )
            
            # if acc < 0.9 and epoch >24:
            #     epoch +=1
#--------------------------------------------------
            merged_result = sess.run(merged,
                                feed_dict={x: val*(1. / 255)-0.5, y_: l, keep_prob: 1}) # merged只有run了才有用""很重要""",不知道为什么非要有keep_pro
            writer.add_summary(merged_result,i*batch_idxs+j)
#--------------------------------------------------
            #   accuracy_result_sum = tf.Summary(value=[tf.Summary.Value(tag="accuracy", simple_value=acc)]) # 转换为tf.summary对象
            #   writer.add_summary(accuracy_result_sum,i*batch_size+j)         # 将上面的accuracy_result_sum加入summary可视化中，每五十步合并的结果都加入
            # 保存和提取参数
            '''记录训练时的准确性'''
            # sess.run(iterator2.initializer)
            # val_test, l_test = sess.run(one_element2)
            # l_test = one_hot(l_test,num_classes) # 原来为l = one_hot(l,2) # one_hot的输入是label
            # y_test, acc_test,cross_entropy_test = sess.run([y_conv,accuracy,cross_entropy], feed_dict={x: val_test*(1. / 255)-0.5, y_: l_test, keep_prob: 1})
            # accuracy_result_sum_acc = tf.Summary(value=[tf.Summary.Value(tag="accuracy_test", simple_value=acc_test)]) # 转换为tf.summary对象
            # writer.add_summary(accuracy_result_sum_acc,i*batch_idxs+j)
            # accuracy_result_sum = tf.Summary(value=[tf.Summary.Value(tag="cross_entropy_test", simple_value=cross_entropy_test)]) # 转换为tf.summary对象
            # writer.add_summary(accuracy_result_sum,i*batch_idxs+j)

    save_path = saver.save(sess,"saver\save_net")
    print("参数保存文件：",save_path)

    sess.run(iterator.make_initializer(dataset2))
    val, l = sess.run(one_element2)
    # val, l = sess.run([img_batch, label_batch])

    l = one_hot(l,num_classes) # 原来为l = one_hot(l,2)
    print(len(l),l)
    y, acc = sess.run([y_conv,accuracy], feed_dict={x: val*(1. / 255)-0.5, y_: l, keep_prob: 1})
    print(np.argmax(y,1))
    print("test accuracy: [%.8f]" % (acc))
    coord.request_stop()
    coord.join(threads)