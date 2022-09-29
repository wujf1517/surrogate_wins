from cgi import test
import tensorflow as tf 
import numpy as np
from CoordConv import AddCoords, CoordConv
import ReadMyownData
from unicodedata import name
import cv2
import preprocessor

num_channels = 3
img_size = 128
img_size_flat = img_size*img_size*num_channels

img_shape = (img_size,img_size)
classes = ['1','2']
num_classes =  len(classes)

batch_size = 14
validation_size = 0.16
early_stopping = None
train_path = 'PicClassTrain'
test_path = 'PicClassTest'
# data = preprocessor.read_

