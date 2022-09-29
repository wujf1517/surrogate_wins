from __future__ import print_function
from unicodedata import name
import tensorflow as tf
import ReadMyownData
# from tensorflow.examples.tutorials.mnist import input_data # 调用数据集

# 先编辑功能
# 构建层结构

# View more python tutorial on my Youtube and Youku channel!!!

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""

# number 1 to 10 data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

batch_size = 20

def compute_accuracy(v_xs, v_ys):
    with tf.name_scope('accuracy'):
        global prediction  # 全局变量
        y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
        correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy',accuracy)
        result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
        return result

def weight_variable(shape): # 权重（卷积核）初始化
    initial = tf.truncated_normal(shape, stddev=0.1) # 正态分布抽样初始化
    return tf.Variable(initial) 

def bias_variable(shape): # 偏置初始化
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W): # x是输入，W是卷积核
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') # strides卷积核移动的速度， 这里采用same填充方式,池化后输入和输出宽度一样

def max_pool_2x2(x): # 构建一个池化层，采用最大池化方法，2x2即步长为2，大小缩小一倍，高度不变
    # stride [1, x_movement, y_movement, 1]，
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # padding填充的意思：这里采用same填充方式


### new add ###???what is the function显示参数的初始化过程？
def variable_summaries(name,var):
    with tf.name_scope(name+'_summaries'):
        mean = tf.reduce_mean(var)                                          # 求均值？
    tf.summary.scalar(name+'_mean', mean)
    with tf.name_scope(name+'_stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
    tf.summary.scalar(name+'_stddev', stddev)
    tf.summary.scalar(name+'_max',tf.reduce_max(var))
    tf.summary.scalar(name+'_min',tf.reduce_min(var))
    tf.summary.histogram(name+'_histogram', var)


with tf.name_scope('inputs'):                                               # 一个命名空间，里面可以包含命名空间和operation
    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, 784],name='xs')/255.             # 28x28
    ys = tf.placeholder(tf.float32, [None, 10],name='ys')                   # name: A name for the operation (optional).
    keep_prob = tf.placeholder(tf.float32,name='kp')

with tf.name_scope('image_reshape'):
    x_image = tf.reshape(xs, [-1, 28, 28, 1])                               # -1就是先不管xs的维度,1是指channel为1,自己有名字Reshape
    # print(x_image.shape)  # [n_samples, 28,28,1]                          # 可以输出x_image的维度
    tf.summary.image('input',x_image)                                       # 图像输入？？?之前没用过

with tf.name_scope('conv1_layer'):
    ## conv1 layer ##
    '''
    卷积之后高度变为32(输出通道数)
    池化之后，宽度减小一倍:7x7
    '''
    W_conv1 = weight_variable([5,5,1,32])                                   # patch 5x5, in size 1, out size 32
    variable_summaries('w1',W_conv1) ## 还有没有别的方法
    b_conv1 = bias_variable([32])                                           # bias的个数由上面的输出通道数决定
    variable_summaries('b1',b_conv1) ## 还有没有别的方法
    with tf.name_scope('Wx_puls_b'):
        Wx_puls_b= conv2d(x_image, W_conv1) + b_conv1
        tf.summary.histogram('pre_act',Wx_puls_b)
    h_conv1 = tf.nn.relu(Wx_puls_b,name='activiation')                      # output size 28x28x32          # tf.nn.relu()激励，使运算结果非线性化
    h_pool1 = max_pool_2x2(h_conv1)                                         # output size 14x14x32          # 池化之后，这一层最后的输出值

with tf.name_scope('conv2_layer'):
    ## conv2 layer ##
    W_conv2 = weight_variable([5,5, 32, 64])                                # patch 5x5, in size 32, out size 64
    variable_summaries('W2',W_conv2)
    b_conv2 = bias_variable([64])                                           # bias的个数由上面的输出通道数决定
    variable_summaries('b2',b_conv2)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)                # output size 14x14x64
    h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64

with tf.name_scope('fc1_layer'):
    ## fc1 layer ##
    '''
    神经网络的结构
    '''
    W_fc1 = weight_variable([7*7*64, 1024])                                 #输入的数据7x7x64,输出为1024(更宽？更高？)
    # variable_summaries('w_fc1',W_fc1)
    b_fc1 = bias_variable([1024])
    variable_summaries('b_fc1',b_fc1)
    #把h_pool2的形状[n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]            # -1表示不知道会是多少，但至少列数是7*7*64
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])                        # h_pool2 是一个二维数组[n_samples, 7*7*64]
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1,name='act')              # [n_samples, 7*7*64]x[7*7*64, 1024]+[1,1024]得出的结果的维度[n_samples, 1024]
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)                            # dropout会不会降低维度？

with tf.name_scope('fc2_layer'):
    ## fc2 layer ##
    W_fc2 = weight_variable([1024, 10])                                     # 10是我们想判断的分类数目（数字0~9）
    variable_summaries('W_fc2',W_fc2)
    b_fc2 = bias_variable([10])
    variable_summaries('b_fc2',b_fc2)
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2,name='softmax')        # [n_samples, 1024]x[1024, 10]+[1,10]=[n_samples, 10]?

with tf.name_scope('cross_entropy1'):
    # the error between prediction and real data
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                reduction_indices=[1]),name='cross_entropy')   # loss
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
'''
结构构建完成
'''

sess = tf.Session()
merged = tf.summary.merge_all()                                 # 合并所有,所有summary都保存在日志中，以便tensorboard进行显示。Merges all summaries collected in the default graph.
writer = tf.summary.FileWriter("CNN_TEST",sess.graph)           # 把前面的全部信息收集起来，放入文件，最终生成可视化，参数为路径，graph是全部的框架

# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

img, label = ReadMyownData.read_and_decode("123train.tfrecords")
img_test, label_test = ReadMyownData.read_and_decode("123test.tfrecords")
for i in range(1000):
    # batch_xs, batch_ys = mnist.train.next_batch(100)
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=batch_size, capacity=2000,
                                                    min_after_dequeue=1000)
    img_test, label_test = tf.train.shuffle_batch([img_test, label_test],
                                                    batch_size=batch_size, capacity=2000,
                                                    min_after_dequeue=1000)


    sess.run(train_step, feed_dict={xs: img_batch, ys: label_batch, keep_prob: 0.5}) # keep_prob:dropout之后保持的比例，例如想要drop0.4,则keep_prob=0.6
    if i % 50 == 0:
        accuracy_result = compute_accuracy(
            img_test, label_test)
        print(accuracy_result)
#       writer.add_summary(accuracy_result,i)                            # 每五十步合并的结果都加入

# 保存和提取参数