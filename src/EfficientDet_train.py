import numpy as np
from datetime import date
import os

from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

# data
label_map = { 1 : 'hand'}
today = date.today()
dir_name = str(today)
batch_num = 32
epoch_num = 10
model_name = "efficientdet_lite0"
aug_version = "v1"

label_map ={ 1 : 'hand'}
train_data_size = 5000
training_data = object_detector.DataLoader("../dataset/EfficientDet/train/hand.tfrecord",train_data_size,label_map)

valid_data_size = 1000
val_data = object_detector.DataLoader("../dataset/EfficientDet/valid/hand.tfrecord",valid_data_size,label_map)

test_data_size = 500
test_data = object_detector.DataLoader("../dataset/EfficientDet/test/hand.tfrecord",test_data_size,label_map)

# model
spec = model_spec.get(model_name)
spec.config.autoaugment_policy = aug_version
model = object_detector.create(training_data,
                                    model_spec=spec,
                                    batch_size=batch_num,
                                    epochs=epoch_num,
                                    train_whole_model=True)

print(model.evaluate(test_data))
model.export(export_dir='.', tflite_filename='../model/' + dir_name + '_' + str(epoch_num) + '_' + str(model_name.split('_')[1]) + '_' +aug_version + '.tflite')

config = QuantizationConfig.for_float16()
model.export(export_dir='.', tflite_filename='../model/' + dir_name + '_' + str(epoch_num)  +'_' + str(model_name.split('_')[1]) + '_' +aug_version +'_fp16.tflite', quantization_config=config)


