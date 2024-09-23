# AOI-Project

## 目錄

- [AOI-Project](#aoi-project)
    - [目錄](#目錄)
    - [環境安裝](#環境安裝)
        - [Anaconda安裝](#anaconda安裝)
        - [Docker安裝](#docker安裝)
            - [Windows-WSL](#windows-wsl)
            - [Windows-Docker-Desktop](#windows-docker-desktop)
            - [Ubuntu](#ubuntu)
    - [快速開始Windows-WSL](#快速開始windows-wsl)
    - [參數檔細節](#參數檔細節)
    - [常見問題解決](#常見問題解決)

## 環境安裝

### Anaconda安裝

<details>
<summary>使用Anaconda建立單機版環境</summary>  

> - **Python 版本:** 3.8.0
> - **PyTorch 版本:** 2.1.0
> - **Torchvision 版本:** 0.16.0
> - **CUDA 版本:** 11.8
> - **NumPy 版本:** 1.23.0
> - **PIP 版本:** 23.3.1

1. **下載Visual studio 2019 並安裝c++相關套件**:
   [Visual studio 2019](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=community&rel=16&utm_medium=microsoft&utm_campaign=download+from+relnotes&utm_content=vs2019ga+button)

2. **創建虛擬環境:**
   ```bash
    conda create --name "AOI" python==3.8
    conda activate AOI
    ```
3. **Clone專案:**
    ```bash
    git clone http://ntustcgalgit.synology.me:8000/foxlink_aoi/model-zoo.git
    cd model-zoo
    ```
4. **cuda安裝:**
    ```bash
    conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
    ```

   確認有沒有安裝成功，如果成功會看到cuda11.8
    ```bash
    nvcc -V 
    ```
6. **pytorch安裝:**
    ```bash
    conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
7. **安裝相依套件:**
    ```bash
    pip install -r requirements.txt
    ```
8. **mmdetection安裝:**
    ```bash
    pip install -U openmim
    mim install "mmengine==0.10.3"
    mim install "mmcv==2.1.0"
    ```
9. **mmpretrain安裝:**
    ```bash
    pip install -U openmim
    ```
    ```bash
    mim install "mmpretrain>=1.0.0rc8"
    ```
10. **pycocotools安裝**
    ```bash
    pip install git+https://github.com/Zhong-Zi-Zeng/cocoapi.git#subdirectory=PythonAPI
    ```
11. **numpy安裝:**
    ```bash
    pip install update numpy==1.23.0
    ```
12. **Pillow安裝:**
    ```bash
    pip install update Pillow==9.5
    ```

</details>

### Docker安裝

#### Windows-WSL

<details>
<summary>使用Windows WSL建立單機版環境</summary>

1. 確認wsl版本是否為2.0以上，打開終端機後輸入以下指令

    ```bash
    wsl --version
    ```
   成功的話會出現以下畫面

   ![/assets/img_3.png](/assets/img_3.png)

3. 輸入以下指令安裝Ubuntu-20.04版本
    ```bash
    wsl --install -d Ubuntu-20.04
    ```

   成功的話會需要輸入使用者名稱與密碼
   ![/assets/img_4.png](/assets/img_4.png)

   輸入完成後會自動進入到wsl裡
   ![/assets/img_6.png](/assets/img_6.png)

   後續想要進到該環境的話，則可以輸入以下指令
    ```bash
    wsl -d Ubuntu-20.04
    ```
4. Clone專案:
   ```bash
   git clone http://ntustcgalgit.synology.me:8000/foxlink_aoi/model-zoo.git
   cd model-zoo 
   ```
5. 安裝docker
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   ```

   ```bash
   sudo sh get-docker.sh
   ```
6. 安裝docker-compose:
    ```bash
    sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    ```
7. 賦予權限:
    ```bash
    sudo chmod +x /usr/local/bin/docker-compose
    ```
8. 啟動docker:
   ```bash
   sudo service docker start
   ```
9. 安裝NVIDIA Container Toolkit:
    ```bash
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
    && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    ```
    ```bash
    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
    ```
    ```bash
    sudo systemctl restart docker
    ```
10. 確認docker-compose是否安裝成功:

    ```bash
    docker-compose --version
    ```

    如果安裝成功會顯示以下訊息:
    ```bash
    docker-compose version 1.29.2, build 5becea4c
    ```
11. 執行docker環境:
    ``` bash
    sudo python3 ./main.py
    ```
12. 新開一個Command視窗，並執行以下命令:
    ```bash
    wsl -d Ubuntu-20.04
    ```

    ```bash
    sudo docker attach AOI
    ```
13. 賦予檔案執行權限:
    ```bash
    chmod 777 -R .
    ```

</details>

#### Windows-Docker-Desktop

<details>
<summary>使用Windows Docker Desktop建立單機版環境</summary>

1. 安裝docker，請參考 https://ithelp.ithome.com.tw/articles/10275627
2. 安裝完畢後，請記得開啟docker desktop，並在終端機執行以下命令

    ``` bash
    docker -h 
    ```
   這邊會出現以下畫面，說明安裝成功
   ![/assets/img_7.png](/assets/img_7.png)

3. Clone專案:
    ```bash
    git clone http://ntustcgalgit.synology.me:8000/foxlink_aoi/model-zoo.git
    cd model-zoo  
    ```
4. 執行docker環境:
    ``` bash
    python ./main.py
    ```
5. 新開一個Command視窗，並執行以下命令:
    ```bash
    docker attach AOI
    ```

</details>

#### Ubuntu

<details>
<summary>使用Ubuntu建立單機版環境</summary>

1. 安裝docker:
    ```bash
    curl -fsSL https://get.docker.com -o get-docker.sh
    ```

    ```bash
    sudo sh get-docker.sh
    ```
2. 安裝docker-compose:
    ```bash
    sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    ```
3. 賦予權限:
    ```bash
    sudo chmod +x /usr/local/bin/docker-compose
    ```
4. 啟動docker:
   ```bash
   sudo service docker start
   ```
5. 安裝NVIDIA Container Toolkit:
    ```bash
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
    && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    ```

    ```bash
    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
    ```

    ```bash
    sudo systemctl restart docker
    ```
6. 確認docker-compose是否安裝成功:
    ```bash
    docker-compose --version
    ```

   如果安裝成功會顯示以下訊息:
    ```bash
    docker-compose version 1.29.2, build 5becea4c
    ```
7. Clone專案:
    ```bash
    git clone http://ntustcgalgit.synology.me:8000/foxlink_aoi/model-zoo.git
    cd modle-zoo
    ```
8. 執行docker環境:
    ``` bash
    sudo python3 ./main.py
    ```
9. 新開一個Command視窗，並執行以下命令:
    ```bash
    sudo docker attach AOI
    ```

</details>

## 快速開始Windows-WSL

請先確認環境已安裝完成，再開始後續步驟

1. 開啟Command視窗，並執行以下命令:
   ```bash
   wsl -d Ubuntu-20.04
   ```
2. 執行docker環境:
   ``` bash
   sudo python3 ./main.py
   ```
3. 新開另一個Command，並執行以下命令:
   ```bash
   wsl -d Ubuntu-20.04
   ```
4. 進入到容器內:
   ```bash
   sudo docker attach AOI
   ```

## 參數檔細節

所有自定義的參數檔會放在 `configs` 資料夾下。**每個 config 檔都可以利用繼承的方式減少程式碼的重複率，如果有想要更改的地方，只需要繼承預設的
config 檔，然後針對需要更改的部分重新設定即可。**

以下是 `CO-DETR-ResNet50` 的範例：

```yaml
_base_: [ "./base/model/CO-DETR/r50.yaml",
          "./base/evaluation/object_detection.yaml",
          "./base/dataset.yaml",
          "./base/hyp.yaml"
]

# ===========Dataset===========
coco_root: null # Path to the coco dataset

# ===========Augmentation===========
hsv_h: 0.15  # image HSV-Hue augmentation (fraction), range 0-1
hsv_s: 0.7  # image HSV-Saturation augmentation (fraction), range 0-1
hsv_v: 0.4  # image HSV-Value augmentation (fraction), range 0-1
degrees: 10  # image rotation (+/- deg), range 0-360
translate: 0.1  # image translation (+/- fraction), range 0-1
scale: 0.9  # image scale (+/- gain), range 0-1, 1 means no scaling
shear: 0.  # image shear (+/- deg), range 0-180
perspective: 0.5  # image perspective (+/- fraction), range 0-1
flipud: 0.  # image flip up-down (probability), range 0-1
fliplr: 0.5  # image flip left-right (probability), range 0-1

# ===========Hyperparameter===========
optimizer: "AdamW" # 可選擇Adam、AdamW、SGD
weight: null # 權重檔路徑，訓練時可以繼續訓練，評估時必須指定
end_epoch: 50  # 訓練次數
warmup_epoch: 3 # 前 3 個 epoch 為暖身階段
initial_lr: 0.00005 # warmup開始的學習率
lr: 0.0002 # warmup結束後的學習率
minimum_lr: 0.000005 # 最小學習率
batch_size: 1
imgsz: [ 1600, 1600 ] # 訓練大小，在CO-DETR中代表長、短邊都不超1600
save_period: 5 # 儲存weight的週期
eval_period: 5 # 評估週期
device: "0" # 指定使用的GPU設備，例如 0 或 0,1,2,3 或 cpu
```

## 常見問題解決

1. 在使用Anaconda安裝方式，執行`pip install -r requirements.txt`，出現以下錯誤:

   ![/assets/messageImage_1727075300205.jpg](/assets/messageImage_1727075300205.jpg)
   
    解決辦法: 確認有無安裝Visual studio且安裝c++相關套件。如果還是無法成功執行，請安裝Visual studio Build Tools 


1. 運行Docker環境時，出現以下錯誤:
    ```bash
    Error response from daemon: failed to create shim task: OCI runtime create failed: runc create failed: unable to start container process: error during container init: error running hook #0: error running hook: exit status 1, stdout: , stderr: Auto-detected mode as 'legacy' nvidia-container-cli: initialization error: load library failed: libnvidia-ml.so.1: cannot open shared object file: no such file or directory: unknown
    ```

   解決辦法:
    ```bash
    sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    ```  
2. 在運行docker環境後，發現redis的端口被占用:

   ![/assets/img_8.png](/assets/img_8.png)

   解決辦法: 清除佔用端口的進程
    1. 查詢占用端口的進程
    ```bash
    sudo lsof -i :6379
    ``` 
    2. 停止該進程
    ```bash
    sudo kill -9 <PID>
    ```
3. 在anaconda中運行server後出現以下錯誤:
   ```bash
   OSError: [WinError 10013] 嘗試存取通訊端被拒絕，因為存取權限不足 
   ```
   ```bash
    An error occurred: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))
    ```
   解決辦法:
    1. 確保port有沒有被占用:
    ```bash
    netstat -ano | findstr :5000
    ```
   ![/assets/img_9.png](/assets/img_9.png)

    2. 查看是哪個程式正在佔用:
    ```bash
    tasklist /FI "PID eq 31972"
    ```
   ![/assets/img_10.png](/assets/img_10.png)

    3. 停止該進程:
    ```bash
    taskkill /PID 31972 /F
    ```

