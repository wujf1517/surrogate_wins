#常用的直接读取方法实例：
#加载包
import tensorflow as tf
import os

#设置工作目录
os.chdir("E:\code\scenarioagentcnn\\two.txt")
#查看目录
print(os.getcwd())

#读取函数定义
def read_data(file_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)
    #定义列
    defaults = [[0], [0.], [0.], [0.], [0.], ['']]
 #编码   Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species = tf.decode_csv(value, defaults)

    #处理
    preprocess_op = tf.case({
        tf.equal(Species, tf.constant('Iris-setosa')): lambda: tf.constant(0),
        tf.equal(Species, tf.constant('Iris-versicolor')): lambda: tf.constant(1),
        tf.equal(Species, tf.constant('Iris-virginica')): lambda: tf.constant(2),
    }, lambda: tf.constant(-1), exclusive=True)

    #栈
    return tf.stack([SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]), preprocess_op


def create_pipeline(filename, batch_size, num_epochs=None):
    file_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
    example, label = read_data(file_queue)

    min_after_dequeue = 1000
    capacity = min_after_dequeue + batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )

    return example_batch, label_batch

x_train_batch, y_train_batch = create_pipeline('Iris-train.csv', 50, num_epochs=1000)
x_test, y_test = create_pipeline('Iris-test.csv', 60)
print(x_train_batch,y_train_batch)


# import timeit   #查看运行开始到结束所用的时间
# import tensorflow as tf
# import os
 
# def generate_tfrecords(input_filename, output_filename):
#     print("\nStart to convert {} to {}\n".format(input_filename, output_filename))
#     start_time = timeit.default_timer()
#     writer = tf.python_io.TFRecordWriter(output_filename)
 
#     for line in open(input_filename, "r"):
#         data = line.split(",")
#         label = float(data[9])
        
       
#         features = [float(i) for i in data[:9]]   #特征不要最后一列数据
#         #将数据转化为原生 bytes
#         example = tf.train.Example(features=tf.train.Features(feature={
#             "label":
#                 tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
#             "features":
#                 tf.train.Feature(float_list=tf.train.FloatList(value=features)),
#         }))
#         writer.write(example.SerializeToString()) #序列化为字符串
 
#     writer.close()
#     print("Successfully convert {} to {}".format(input_filename,
#                                                  output_filename))
#     end_time = timeit.default_timer()
#     print("\nThe pretraining process ran for {0} minutes\n".format((end_time - start_time) / 60))
 
# def main():
#     # current_path = "G:/Spyder/csv_TFrecords/"#E:\code\scenario_CNN
#     current_path = "E:/code/scenario_CNN/"
#     for filename in os.listdir(current_path):
#         if filename == "two.csv":    #当前路径下，需要转换的CSV文件
#             generate_tfrecords(current_path+filename, current_path+filename + ".tfrecords")        
 
#     return current_path+filename + ".tfrecords"
 
# if __name__ == "__main__":
#     main()