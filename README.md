# Gesture_recognition
using EfficientDet and lstm to recognition gesture

```
$ git clone https://github.com/TingRui608470158/Gesture_recognition.git
$ cd Gesture_recognition
```
## Step1 
### 用yolov5做object detection

Windows環境(Anaconda)
```
$ conda create -n ray-pt2 python=3.7
$ conda activate ray-pt2
```
pytorch安裝
```
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

```
yolov5安裝
```
$ cd step1
$ git clone https://github.com/ultralytics/yolov5  
$ mv create_csv_file.py yolov5
$ cd yolov5
$ pip install -r requirements.txt  
```

手部資料集下載([連結](https://drive.google.com/file/d/1N59Gne5AfxXC6mqmHFVToakzUHqRj8nz/view?usp=share_link))
將資料集的train test valid資料集移至step1路徑下
data.taml 移至step1/yolov5路徑下

訓練yolov5
```
$ python train.py --weights yolov5s --data data.yaml --cfg yolov5s.yaml
```

