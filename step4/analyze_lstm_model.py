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

X_range_list_all = []

all_range = 1
#up
X_range_list_single = []
x_range = np.linspace(all_range, -all_range, 20)
y_range = -all_range

X_range_list_single.append("up")        #class  0
X_range_list_single.append(x_range)     #x      1
X_range_list_single.append(y_range)     #y      2   
X_range_list_single.append("b")         #color  3
X_range_list_all.append(X_range_list_single)

#down
X_range_list_single = []
x_range = np.linspace(all_range, -all_range, 20)
y_range = all_range
X_range_list_single.append("down")      #class  0
X_range_list_single.append(x_range)     #x      1
X_range_list_single.append(y_range)     #y      2
X_range_list_single.append("g")         #color  3
X_range_list_all.append(X_range_list_single)

#left
X_range_list_single = []
x_range = -all_range
y_range = np.linspace(all_range, -all_range, 20)

X_range_list_single.append("left")      #class  0
X_range_list_single.append(x_range)     #x      1
X_range_list_single.append(y_range)     #y      2
X_range_list_single.append("c")         #color  3
X_range_list_all.append(X_range_list_single)


#right
X_range_list_single = []
x_range = all_range
y_range = np.linspace(all_range, -all_range, 20)

X_range_list_single.append("right")      #class  0
X_range_list_single.append(x_range)     #x      1
X_range_list_single.append(y_range)     #y      2
X_range_list_single.append("y")         #color  3
X_range_list_all.append(X_range_list_single)

print(X_range_list_all)

plt.ion()
for single_gesture in X_range_list_all:
    print("Gesture = ",single_gesture[0])
    if single_gesture[0]=="up" or single_gesture[0]=="down":
        for x_range in single_gesture[1]:
            plt.xlim(-1,1)
            plt.ylim(1,-1)
            plt.xlabel('x label') # 設定 x 軸標題
            plt.ylabel('y label') # 設定 y 軸標題

            # slope = -y_range/x_range
            # print("slope = ",slope)

            x_list= np.linspace(0, x_range, 16)
            y_list= np.linspace(0, single_gesture[2], 16)
            point_list = []
            for i in range(len(x_list)):
                # single_point = [x_list[i],y_list[i]]
                # point_list.append(single_point)
                point_list.append(x_list[i])
                point_list.append(y_list[i])
                plt.scatter(x_list[i],y_list[i], c=single_gesture[3])
            plt.pause(0.01)

            print("point_list = ",point_list)
            gesture_id = point_history_classifier(point_list)
            print("True_id = ",single_gesture[0])
            print("gesture_id = ",Gesture_class[gesture_id])

            if single_gesture[0] == Gesture_class[gesture_id]:
                class_color = single_gesture[3]
            else:
                class_color = 'r'

            for i in range(len(x_list)):
                plt.scatter(x_list[i],y_list[i], c=class_color)

            

    if single_gesture[0]=="left" or single_gesture[0]=="right":
        for y_range in single_gesture[2]:
            plt.xlim(-1,1)
            plt.ylim(1,-1)
            plt.xlabel('x label') # 設定 x 軸標題
            plt.ylabel('y label') # 設定 y 軸標題

            # slope = -y_range/x_range
            # print("slope = ",slope)

            x_list= np.linspace(0, single_gesture[1], 16)
            y_list= np.linspace(0, y_range, 16)
            point_list = []
            for i in range(len(x_list)):
                # single_point = [x_list[i],y_list[i]]
                # point_list.append(single_point)
                point_list.append(x_list[i])
                point_list.append(y_list[i])
                # plt.scatter(x_list[i],y_list[i], c=single_gesture[3])
                
            plt.pause(0.01)
            
            print("point_list = ",point_list)
            gesture_id = point_history_classifier(point_list)
            print("True_id = ",single_gesture[0])
            print("gesture_id = ",Gesture_class[gesture_id])

            if single_gesture[0] == Gesture_class[gesture_id]:
                class_color = single_gesture[3]
            else:
                class_color = 'r'

            for i in range(len(x_list)):
                plt.scatter(x_list[i],y_list[i], c=class_color)

        print("Next Gesture")
plt.pause(1000)
plt.ioff()
