# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

Gesture_class = ['stop', 'clockwise', 'anticlockwise', 'down', 'left', 'right', 'up']

class PointHistoryClassifier(object):
    def __init__(
        self,
        model_path='../model/lstm_model.tflite',
        score_th=0.5,
        invalid_value=0,
        num_threads=1,
    ): 
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.score_th = score_th
        self.invalid_value = invalid_value

    def __call__(
        self,
        point_history,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([point_history], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        if np.squeeze(result)[result_index] < self.score_th:
            result_index = self.invalid_value

        return result_index


point_history_classifier = PointHistoryClassifier()


#clockwise
curve_rate = [1, 2, 3, 4]
x_curve_mid = [0.2, 0.3, 0.4, 0.5]
# x_part = [0.75, 0.8, 0.85, 0.9, 0.95, 1]
x_part = 0.7
plt.ion()

# clockwise
for i in curve_rate:    #曲率
    for j in x_curve_mid:   #x中點
        X_range_list_all = []
        x = np.linspace(0, 2*j*x_part, 16)
        y = ((x-j)*(x-j) - j*j)*i

        X_range_list_single = []
        for k in range(16):
            X_range_list_single.append(x[k])
            X_range_list_single.append(y[k])
        print(X_range_list_single)
        True_id = 1
        gesture_id = point_history_classifier(X_range_list_single)
        if True_id == gesture_id:
            class_color = 'b'
        else:
            class_color = 'r'
        print(Gesture_class[gesture_id])


        plt.xlim(-1,1)
        plt.ylim(1,-1)
        plt.xlabel('x label') 
        plt.ylabel('y label') 

        plt.plot(x,y, c=class_color)
        plt.pause(1)

#anticlockwise
x_curve_mid = [-0.2, -0.3, -0.4, -0.5]

for i in curve_rate:    #曲率
    for j in x_curve_mid:   #x中點
        X_range_list_all = []
        x = np.linspace(0, 2*j*x_part, 16)
        y = ((x-j)*(x-j) - j*j)*i

        X_range_list_single = []
        for k in range(16):
            X_range_list_single.append(x[k])
            X_range_list_single.append(y[k])
        print(X_range_list_single)
        True_id = 2
        gesture_id = point_history_classifier(X_range_list_single)
        if True_id == gesture_id:
            class_color = 'b'
        else:
            class_color = 'r'
        print(Gesture_class[gesture_id])


        plt.xlim(-1,1)
        plt.ylim(1,-1)
        plt.xlabel('x label') 
        plt.ylabel('y label') 

        plt.plot(x,y, c=class_color)
        plt.pause(1)

plt.pause(1000)
plt.ioff()
