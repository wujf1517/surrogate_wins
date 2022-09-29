# from __future__ import division
# from __future__ import print_function
# from __future__ import absolute_import

# import os
# import io
# import pandas as pd
# import tensorflow as tf

# from PIL import Image
# from object_detection.utils import dataset_util
# from collections import namedtuple, OrderedDict

# flags = tf.app.flags
# # tf.app.flags.DEFINE_string("param_name", "default_val", "description")
# flags.DEFINE_string('csv_input', 'train.csv', 'Path to the CSV input')
# flags.DEFINE_string('image_dir', './data/train/', 'Path to the image directory')
# flags.DEFINE_string('output_path', 'train.record', 'Path to output TFRecord')
# FLAGS = flags.FLAGS


# # TO-DO replace this with label map
# # 为什么从1开始，而不是从0开始?????
# def class_text_to_int(row_label):
#     if row_label == 'dog':
#         return 1
#     if row_label == 'pig':
#         return 2
#     if row_label=='cat':
#         return 3
#     else:
#         return None


# def split(df, group):
# 	"""
#     对csv数据进行处理
#     :param df: 
#     :param group: 聚合关键字
#     :return: [('image_filename_1',DataFrame_1),('image_filename_2',DataFrame_2),...] 
#     (图片名，该图片的所有boxes信息)
#     """
#     data = namedtuple('data', ['filename', 'object']) #创建一个namedtuple的数据类型，有两个属性filename,object
#     gb = df.groupby(group) #对关键列group进行聚合，有同一张图片多个标记框的聚合在一起
#     return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


# def create_tf_example(group, path):
# 	 """
#     创建tf.Example消息
#     :param group: tuple,每一张图片的信息(filename,DataFrame)
#     :param path: 数据集的路径
#     :return:
#     """

#     #tf.gfile.GFile(filename, mode)
#     #获取文本操作句柄，类似于python提供的文本操作open()函数，filename是要打开的文件名，mode是以何种方式去读写，将会返回一个文本操作句柄。

#     with  tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
#         encoded_jpg = fid.read()
#     encoded_jpg_io = io.BytesIO(encoded_jpg)
#     image = Image.open(encoded_jpg_io)
#     width, height = image.size

#     filename = group.filename.encode('utf8')
#     image_format = b'jpg'
#     xmins = []
#     xmaxs = []
#     ymins = []
#     ymaxs = []
#     classes_text = []
#     classes = []

#     for index, row in group.object.iterrows():
#         xmins.append(row['xmin'] / width) #相对值
#         xmaxs.append(row['xmax'] / width)
#         ymins.append(row['ymin'] / height)
#         ymaxs.append(row['ymax'] / height)
#         classes_text.append(row['class'].encode('utf8'))
#         classes.append(class_text_to_int(row['class']))

#     tf_example = tf.train.Example(features=tf.train.Features(feature={
#         'image/height': dataset_util.int64_feature(height),
#         'image/width': dataset_util.int64_feature(width),
#         'image/filename': dataset_util.bytes_feature(filename),
#         'image/source_id': dataset_util.bytes_feature(filename),
#         'image/encoded': dataset_util.bytes_feature(encoded_jpg),
#         'image/format': dataset_util.bytes_feature(image_format),
#         'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
#         'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
#         'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
#         'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
#         'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
#         'image/object/class/label': dataset_util.int64_list_feature(classes),
#     }))
#     return tf_example


# def main(_):
#     writer = tf.io.TFRecordWriter(FLAGS.output_path)
#     path = FLAGS.image_dir
#     examples = pd.read_csv(FLAGS.csv_input)
#     grouped = split(examples, 'filename')
#     for group in grouped:
#         tf_example = create_tf_example(group, path)
#         writer.write(tf_example.SerializeToString())

#     writer.close()
#     output_path =FLAGS.output_path
#     print('Successfully created the TFRecords: {}'.format(output_path))


# if __name__ == '__main__':
#      tf.compat.v1.app.run()

