# Gesture recognition 手勢辨識(Ubuntu20.04)
## 手勢辨識流程
1. 收集lstm的訓練集
* 先訓練物件辨識模型(yolov5)，用來追蹤手的軌跡
這裡用yolov5訓練,準確率會較高,但此github目標是要在手機端做,需要夠小的模型及能夠在app上使用的模型,我採用EfficientDet作為手機端上做偵測的模型,原因是此模型tensorflow官方有範例支援。而yolov5訓練完後,要用yolov5權重檔來做lstm的訓練集。
* 先拍手部動作影片,每一個class都會拍一個影片,之後每一個class的驗證就是對應相對應的影片的手部軌跡做驗證。
* 再將yolov5辨識影片的結果輸出成csv檔,初步的將所有時間序列的手軌跡都輸出成csv檔。
2. 對csv檔的資料做軌跡的驗證
這裡需要用人工驗證csv檔的所有軌跡,並移除步不需要的軌跡。
3. 訓練lstm
根據前面所建好的資料集進行訓練。

###  專案下載
```
git clone https://github.com/TingRui608470158/Gesture_recognition.git
cd Gesture_recognition
```
<font size=5> 1. 收集lstm的訓練集</font><br>

* ### yolov5

#### anaconda
```
conda create -n env1 python=3.8
conda activate env1
```
#### pytorch安裝
cuda需要提前安裝好
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
#or
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```
#### yolov5安裝
```
cd src
git clone https://github.com/ultralytics/yolov5  
mv create_csv_file.py yolov5
cd yolov5

pip install -r requirements.txt 
```

#### 準備資料集
手部資料集範例下載
```
pip install roboflow
python
from roboflow import Roboflow
rf = Roboflow(api_key="hyh1caIftsOB4z9RuLIs")
project = rf.workspace("leapsy").project("hand_dataset")
dataset = project.version(19).download("yolov5")
exit()
```
這裡僅是我訓練用的一小部分,僅供訓練範例使用。
將資料集的train,test,valid資料集移至src路徑下,data.yaml移至src/yolov5路徑下。
如果想自己收集資料集,可以看下一小節。
* 收集資料集
我使用Roboflow標記及輸出資料[(參考這裡)](https://ithelp.ithome.com.tw/articles/10305264)
輸出格式選擇YOLO v5 PyTorch
![](https://i.imgur.com/Hzjna7i.png)



#### 訓練yolov5
```
python train.py --weights yolov5s --data hand_dataset-19/data.yaml --cfg yolov5s.yaml

# 如果有遇到opencv問題可以試這個
# pip install opencv-python==4.5.1.48
```
訓練完後的模型權重會在 runs/train/"exp"/weights 下,這裡的"exp"實際名稱會根據執行train.py的次數而不同,簡單來說就是最新的exp資料夾,後面的數字會越大,如下圖所示。
![](https://i.imgur.com/5Yilcj8.png)

* ### 錄影動作
這裡分為5個動作: up, down, left, right, stop。如果想要增加其他動作也是沒有問題的,只是要注意動作的相似度,原本我還有clockwise,anticlockwsie動作,但手的揮動速度要抓的很準,所以後來刪除了。
下面影片為範例往上揮的動作。

https://user-images.githubusercontent.com/58456895/211729738-bc4b5742-485a-4359-94dc-7ae1416388d9.mp4

錄完後將影片放入dataet資料夾如下圖所示。
```
## 如果不想錄影，可以做以下步驟來下載影片。
pip install gdown
gdown --no-check-certificate --folder https://drive.google.com/drive/folders/1Nz1rAHkG7t-e51Vya1gdeoBhi_Wi4B5I?usp=share_link
mv ./lstm ../../dataset
```
![](https://i.imgur.com/pPWbCwu.png)

* ### 生成初步csv檔
這裡的weights路徑為yolov5訓練的權重檔路徑。
```
python create_csv_file.py --weights runs/train/exp1/weights/best.pt --source ../../dataset/lstm
```
會生成一個annotation/step1_output.csv檔。
* step1_output.csv
![](https://i.imgur.com/VtHgTJb.png)
    * 第1欄: class值。
    * 第2,3欄: 軌跡的起始點(x,y),所以永遠都為(0,0)。
    * 第4,5,6,7,8,9欄: 軌跡的剩下三個點相對於起始點的位置,並將值標準化至-1~1。
<br><br><br>

<font size=5> 2.對csv檔的資料做軌跡的驗證</font><br>
這裡會用到一個叫keyboard的模組,但在linux中必須要在root下才能使用,因此會跟anaconda環境衝突。所以要離開anaconda在root下重架一下環境。
*  環境架設
```
sudo su
pip install keyboard
pip install pandas
pip install matplotlib
pip install opencv-python
```

* 執行程式
```
cd ..
python3 analyze_data.py
```
* 介面說明
    * Row:7/5878,label:1,down
Row代表的是總共有5879筆資料(包含0),現在是第7筆。
label代表的是每一筆資料的第一欄的數值,這裡1對應到的是down。
所有label為["stop", "down", "left", "right", "up"]。

    * 軌跡
x軸與y軸標的數值準化到-1~1。
軌跡為圖中的4個藍色點,因為我採用4個frame的軌跡來做為訓練資料集。

* 操作說明 
先"down"一次後，就會顯示出介面。
    * down: 下一筆資料。
    * up:  上一筆資料。
    * enter: 確認這筆軌跡資料為正確的軌跡，將其存至新文件的資料內。
    * delete: 當不小心"enter"錯資料後，可以用來刪除上一筆存入新文件的資料。

![](https://i.imgur.com/BOpGaLm.png)
<br>
完成後，會生成一個annotation/step2_output.csv檔，訓練集就完成了。<br><br><br>

<font size=5> 3.訓練lstm </font><br>

*  環境架設
```
exit
pip install scikit-learn
pip install seaborn
pip install tensorflow
pip install jupyter
```
* train

打開jupyter notebook
```
cd ..
jupyter notebook
```
打開lstm_training
然後按照順序run就可以了
權重檔會存放在Gesture_recognition/model/lstm_model.tflite

* 驗證lstm
用人工生成的資料來驗證上下左右的軌跡。
```
python analyze_lstm_model.py
```
不同動作的軌跡範圍分別對應不同的顏色。

| 動作 |up | down | right | left |
| -------- | -------- | -------- |-------- |-------- |
| 顏色     | 藍     |  綠  |青     |黃     |

紅色區域則是會出現誤判的情況,可以藉此查看資料集是否有缺失。

![](https://i.imgur.com/1T5ACst.png)<br>
另外可以調整 all_range 參數,範圍為0~1,可以看不同長度的軌跡的辨識範圍。
ex: all_range = 0.5
![](https://i.imgur.com/pJNL8oQ.png)<br>

<br><br>
<font size=5> 4.訓練EfficientDet </font><br>
這裡是使用tensorflow 所提供的tflite-model-maker套件來訓練,模型使用最小的EfficientDet-lite0。
* 環境架設
```
conda deactivate
conda create -n env2 python=3.6
conda activate env2

pip install tflite-model-maker
pip install pycocotools
```
* 資料集
使用roboflow生成所需的資料集,格式使用Tensorflow TFRecord。
![](https://i.imgur.com/uTlPoyZ.png)

如果想要直接取得dataset
```
##安裝roboflow
pip install roboflow

python
##執行以下程式,就會下載
from roboflow import Roboflow
rf = Roboflow(api_key="hyh1caIftsOB4z9RuLIs")
project = rf.workspace("leapsy").project("hand_dataset")
dataset = project.version(19).download("tfrecord")
exit()
mv hand_dataset-19 ../dataset/EfficientDet

```

* 訓練
```
python EfficientDet_train.py
```
訓練完後會於model資料夾下輸出兩種模型,fp16是用於pc端的(PC對浮點數運算很快,但不支援量化操作),另一種是量化後的模型,用於mobile端的使用。


<br><br>
<font size=5> 5. Demo </font><br>
* Object Detection
"your_odt_model"為自己EfficientDet模型的名稱。
```
python EfficientDet_inference.py --model ../model/"your_odt_model.tflite"
```
* Object Detection-LSTM
```
python EfficientDet_LSTM_inference.py --odt_model ../model/"your_odt_model.tflite" --lstm_model ../model/lstm_model.tflite 
```




