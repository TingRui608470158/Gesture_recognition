import numpy as np
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
label_map ={ 1 : 'hand'}
train_data_size = 5000
training_data = object_detector.DataLoader("data/only_my_hand/1114myhand/train/hand.tfrecord",train_data_size,label_map)

valid_data_size = 1000
val_data = object_detector.DataLoader("data/only_my_hand/1114myhand/valid/hand.tfrecord",valid_data_size,label_map)

test_data_size = 150    
test_data = object_detector.DataLoader("data/only_my_hand/1114myhand/test/hand.tfrecord",test_data_size,label_map)

# print(training_data)
# print(val_data)

# model
spec = model_spec.get('efficientdet_lite0')
model = object_detector.create(training_data, model_spec=spec, batch_size=32,epochs=50, train_whole_model=True, validation_data=val_data)


#test
model.evaluate(test_data)
config = QuantizationConfig.for_float16()
model.export(export_dir='.', tflite_filename='output/1114/1114myhand_50epoch_fp16.tflite', quantization_config=config)
model.evaluate_tflite('output/1114myhand_50epoch_fp16.tflite', test_data)


model.export(export_dir='.', tflite_filename='output/1114/1114myhand_50epoch.tflite')
model.evaluate_tflite('output/1114myhand_50epoch_int8.tflite', test_data)

config = QuantizationConfig.for_dynamic()
model.export(export_dir='.', tflite_filename='output/1114/1114myhand_50epoch_dynamic.tflite', quantization_config=config)
model.evaluate_tflite('output/1114myhand_50epoch_dynamic.tflite', test_data)


