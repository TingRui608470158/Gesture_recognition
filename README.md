# 手勢辨識流程
1. 收集lstm的訓練集
* 先訓練物件辨識模型(yolov5)，用來追蹤手的軌跡
這裡用yolov5訓練,準確率會較高,但此github目標是要在手機端做,需要夠小的模型及能夠在app上使用的模型,我採用EfficientDet作為手機端上做偵測的模型,原因是此模型tensorflow官方有範例支援。而yolov5訓練完後,要用yolov5權重檔來做lstm的訓練集。
* 先拍手部動作影片,每一個class都會拍一個影片,之後每一個class的驗證就是對應相對應的影片的手部軌跡做驗證。
* 再將yolov5辨識影片的結果輸出成csv檔,初步的將所有時間序列的手軌跡都輸出成csv檔。

2. 對csv檔的資料做軌跡的驗證
這裡需要用人工驗證csv檔的所有軌跡,並移除步不需要的軌跡。

3. 訓練lstm
根據前面所建好的資料集進行訓練。


## 專案下載
```
$ git clone https://github.com/TingRui608470158/Gesture_recognition.git
$ cd Gesture_recognition
```
## Step1 
### yolov5

#### anaconda
```
$ conda create -n ray-pt2 python=3.7
$ conda activate ray-pt2
```
#### pytorch安裝
cuda需要提前安裝好
```
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```
#### yolov5安裝
```
$ cd step1
$ git clone https://github.com/ultralytics/yolov5  
$ mv create_csv_file.py yolov5
$ cd yolov5
$ pip install -r requirements.txt  
```

#### 準備資料集
手部資料集下載([連結](https://drive.google.com/file/d/1N59Gne5AfxXC6mqmHFVToakzUHqRj8nz/view?usp=share_link))
將資料集的train test valid資料集移至step1路徑下
data.taml 移至step1/yolov5路徑下


#### 訓練yolov5
```
$ python train.py --weights yolov5s --data data.yaml --cfg yolov5s.yaml
```
訓練完後的模型權重會在 runs/train/"exp"/weights 下,這裡的"exp"實際名稱會根據執行train.py的次數而不同,簡單來說就是最新的exp資料夾,後面的數字會越大,如下圖所示。
![](https://i.imgur.com/5Yilcj8.png)

### 用Windows內建的錄影軟體錄影
https://user-images.githubusercontent.com/58456895/211729580-d0d89a23-4349-4f30-81da-58718ed35ef6.mp4



