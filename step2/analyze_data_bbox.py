import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches


import cv2
import keyboard
import csv

gesture_class = 7

gesture_id = ["stop", "clockwise", "anticlock_wise", "down", "left", "right", "up","front"]
input_path = '../annotation/step1_output.csv'
output_path = '../annotation/step2_output.csv'
Last_row_index = 0
Last_row_flag = False
def write_row_data(single_row):
    with open(output_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)# 建立 CSV 檔寫入器
        writer.writerow(single_row)# 寫入一列資料

def delete_row_data():
    with open(output_path, 'r') as infile: 
        reader = csv.reader(infile)
        output_csv = []
        for row in reader:
            output_csv.append(row)

    with open(output_path, 'w',newline='') as infile: 
        writer = csv.writer(infile)
        print("output_csv = ",len(output_csv))
        print("output_csv[0:-1] = ",len(output_csv[:-1]))
        output_csv = output_csv[:-1]
        writer.writerows(output_csv)

# 資料讀取與前處理
hand_gesture_list = pd.read_csv(input_path,header=None)   #讀取dataset
# print(hand_gesture_list.head(3))                #頭數據
class_value = hand_gesture_list.iloc[:,0:1]     #預測值
input_value = hand_gesture_list.iloc[:,1:]      #座標值
print("input_value.shape, class_value.shape = ", input_value.shape, class_value.shape)

# 輸入資料前處理
## 按照class分類
class_value = class_value.values.reshape((1,-1)).tolist() #將class value轉成list

plt.ion()

i= 0
while i < len(class_value[0]):

    list_row = []
    if keyboard.read_key() == 'down':
        i = i+1
        print('next data')  
    elif keyboard.read_key() == 'enter':
        Last_row_flag = True
        Last_row_index = i
        list_row.append(class_value[0][i])
        for k in single_row_data.tolist()[0]:
            list_row.append(k)
        print(list_row)
        write_row_data(list_row)
        print('save data') 
        i = i+3
    elif keyboard.read_key() == 'up':
        print('privous data') 
        i = i-1
    elif keyboard.read_key() == 'delete':
        delete_row_data()
        print('delete') 

    if Last_row_flag:
        plt_text2 = "Last row index :" + str(Last_row_index)
        plt.text(-1.0,-1.22,plt_text2,fontsize=10)

    print("row(",i,")")
    #資料取值
    single_row_data = input_value.values[i:i+1] #取單一列的資料
    x_value = [float(single_row_data[:,j]) for j in range(0,single_row_data.shape[1],3) ] #將單一列的x值取出並轉呈list
    y_value = [float(single_row_data[:,j]) for j in range(1,single_row_data.shape[1],3) ] #將單一列的y值取出並轉呈list
    bbox_value = [float(single_row_data[:,j]) for j in range(2,single_row_data.shape[1],3) ]

    # print(single_row_data[0])
    # bbox_weight_list = []
    x_size = 0
    for j,bbox_size in enumerate(bbox_value):
        print(j,bbox_size) 
        # bbox_size = bbox_size*640*640/1920/1080
        bbox_weight = round((bbox_size)** 0.5, 3)
        plt_text = str(bbox_weight)
        plt.text(-1+x_size,-0.9,plt_text,fontsize=12*bbox_size*8)
        x_size = x_size +0.2
    plt.pause(0.01)
        # bbox_weight_list.append(bbox_weight)



    plt.xlim(-1,1)
    plt.ylim(1,-1)
    plt.xlabel('x label') # 設定 x 軸標題
    plt.ylabel('y label') # 設定 y 軸標題

    plt.scatter(x_value, y_value)

    plt_text = "Row:"+str(i)+"/"+str(len(class_value[0])-1) + ", label:" + str(class_value[0][i])+","+ gesture_id[int(class_value[0][i])]
    plt.text(-1,-1.1,plt_text,fontsize=18)
    # 繪製散點圖    
    plt.pause(0.1)
    plt.cla()
    print("next loop")
plt.ioff()
print("End!!!!!!!")