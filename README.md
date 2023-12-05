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

