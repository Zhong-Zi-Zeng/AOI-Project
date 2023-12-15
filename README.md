# AOI-Project

# 數據集處理
## Step1: 數據集自動拆分
1. 先準備好原始的dataset，裡面有包含image和json檔
2. 建立一個txt檔，裡面包含這個dataset所有的class (可以先使用classes_w.txt)
3. 在tools資料夾執行以下指令
```bash
python slice_data.py \
--source_dir {原始dataset的路徑} \
--classes_txt {這個dataset所包含的class，為一個txt檔} \
--assign_number {在testing dataset中每個class要包含多少張照片}
```
4. 執行完後會看到剛剛的source資料夾有5個檔案分別如下

* delete: 存放有問題的json檔和照片
* test: 存放testing的照片
* train: 存放training的照片
* all_detail.json: 整個資料集的描述，包含每個class的數量和在哪張照片裡面有這個class
* test_detail.json: 測試集的描述，包含每個class的數量和在哪張照片裡面有這個class

## Step2: 數據集格式轉換
目前支援轉換到以下格式 **coco**、**yoloSeg**、**yoloBbox**、**sa**

```bash
python create_data.py \
--source_dir {剛剛拆分好的數據集中，test或是train資料夾路徑} \
--output_dir {輸出路徑} \
--classes_txt {這個dataset所包含的class，為一個txt檔} \
--dataset_type {要做為traing用的dataset還是testing用的dataset，選項有'train' or 'test'}
--format {要轉換的格式，選項有'coco'、'yoloSeg', 'yoloBbox', 'sa'}
```

❗️❗️這邊要注意，test和train是分別產生的，所以要產生完整的資料集需要分別執行，以coco為例
先執行
```bash
python create_data.py \
--source_dir ./source_dir/train
--output_dir ./coco_dataset
--classes_txt ./classes_w.txt
--dataset_type train
--format coco
```
再執行
```bash
python create_data.py \
--source_dir ./source_dir/test
--output_dir ./coco_dataset
--classes_txt ./classes_w.txt
--dataset_type test
--format coco
```
# 評估工具使用
## Step1: 
安裝numpy(一定要這個版本)
```
pip install update numpy==1.20.3
```

安裝Cython
```
pip install Cython
```

安裝pycocotools
```
pip install git+https://github.com/Zhong-Zi-Zeng/cocoapi.git#subdirectory=PythonAPI
```

## Step2: 
這邊要進到AOI-Project這個資料下

## Step3: 
利用[數據集處理](https://github.com/Zhong-Zi-Zeng/AOI-Project/edit/main/README.md#%E6%95%B8%E6%93%9A%E9%9B%86%E8%99%95%E7%90%86)去生成coco資料格式的dataset
## Step4: 
打開configs/yolov7_custom.yaml檔案，這邊需要去更改對應檔案的位置(只列出需要更改的地方)
```yaml
coco_root: "../coco" # 更改成由create_data.py生成出來的coco dataset路徑

# ===========Hyperparameter===========
optimizer: "AdamW" # Adam、AdamW、SGD 這個model是用哪個optimizer train出來的
weight: "./best.pt" # yolov7的weight路徑
imgsz: [1024, 1024] # 這個model是用哪種image_size train出來的
```

## Step5:
開始評估，最後會顯示評估數據, 預設會自動在works_dir下產生excel和json檔去紀錄評估數據
```
python ./tools/evaluation.py -c {yolov7_seg.yaml這個檔案的路徑(建議用絕對路徑)}
```

如果想要添加已有的excel檔，則可使用下面的方式，會直接將評估數據附加在給定的excel檔下

```
python ./tools/evaluation.py -c {yolov7_seg.yaml這個檔案的路徑(建議用絕對路徑)} -e {excel檔案路徑}
```

Baseline:
![image](https://github.com/Zhong-Zi-Zeng/AOI-Project/assets/102845636/b4178e48-fe9c-4f60-a670-5630024a1434)





