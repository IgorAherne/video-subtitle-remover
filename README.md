Translation of the original repo's README:

```markdown
Simplified Chinese | [English](README_en.md)

## Project Introduction

![License](https://img.shields.io/badge/License-Apache%202-red.svg)
![python version](https://img.shields.io/badge/Python-3.8+-blue.svg)
![support os](https://img.shields.io/badge/OS-Windows/macOS/Linux-green.svg)  

Video-subtitle-remover (VSR) is software based on AI technology that removes hardcoded subtitles from videos.
It mainly implements the following functions:
- Removes hardcoded subtitles from videos at **lossless resolution**, generating a file with subtitles removed.
- Fills the area where subtitle text was removed using powerful AI algorithm models (not non-adjacent pixel filling or mosaic removal).
- Supports custom subtitle positions, removing only subtitles within the defined position (by passing the position).
- Supports automatic removal of all text throughout the entire video (without passing a position).
- Supports batch removal of watermark text from multiple selected images.

<p style="text-align:center;"><img src="https://github.com/YaoFANGUK/video-subtitle-remover/raw/main/design/demo.png" alt="demo.png"/></p>

**Usage Instructions:**

 - For usage issues, please join the QQ group for discussion: 806152575 (Full), 816881808
 - Download the compressed package directly, extract, and run. If it doesn't run, then follow the tutorial below to try installing the conda environment from source.

**Download Links:**

Windows GPU version v1.1.0 (GPU):

- Baidu Netdisk: <a href="https://pan.baidu.com/s/1zR6CjRztmOGBbOkqK8R1Ng?pwd=vsr1">vsr_windows_gpu_v1.1.0.zip</a> Extraction code: **vsr1**

- Google Drive: <a href="https://drive.google.com/drive/folders/1NRgLNoHHOmdO4GxLhkPbHsYfMOB_3Elr?usp=sharing">vsr_windows_gpu_v1.1.0.zip</a> 

> For users with Nvidia graphics cards only (AMD graphics cards won't work)

## Demo

- GUI Version:

<p style="text-align:center;"><img src="https://github.com/YaoFANGUK/video-subtitle-remover/raw/main/design/demo2.gif" alt="demo2.gif"/></p>

- <a href="https://b23.tv/guEbl9C">Click to view demo video👇</a>

<p style="text-align:center;"><a href="https://b23.tv/guEbl9C"><img src="https://github.com/YaoFANGUK/video-subtitle-remover/raw/main/design/demo.gif" alt="demo.gif"/></a></p>

## Source Code Usage Instructions

> **Do not use this project if you don't have an Nvidia graphics card**. Minimum configuration:
>
> **GPU**: GTX 1060 or higher graphics card
> 
> CPU: Supports AVX instruction set

#### 1. Download and Install Miniconda

- Windows: <a href="https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Windows-x86_64.exe">Miniconda3-py38_4.11.0-Windows-x86_64.exe</a>

- Linux: <a href="https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh">Miniconda3-py38_4.11.0-Linux-x86_64.sh</a>

#### 2. Create and Activate Virtual Environment

(1) Switch to the source code directory:
```shell
cd <source_code_directory>
```
> For example: If your source code is in the tools folder on drive D, and the source code folder name is video-subtitle-remover, enter ```cd D:/tools/video-subtitle-remover-main```

(2) Create and activate the conda environment
```shell
conda create -n videoEnv python=3.8
```

```shell
conda activate videoEnv
```

#### 3. Install Dependency Files

Please ensure you have installed python 3.8+. Use conda to create a project virtual environment and activate the environment (it is recommended to create a virtual environment to run, to avoid problems later).

- Install CUDA and cuDNN

  <details>
      <summary>Linux Users</summary>
      <h5>(1) Download CUDA 11.7</h5>
      <pre><code>wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run</code></pre>
      <h5>(2) Install CUDA 11.7</h5>
      <pre><code>sudo sh cuda_11.7.0_515.43.04_linux.run</code></pre>
      <p>1. Enter accept</p>
      <img src="https://i.328888.xyz/2023/03/31/iwVoeH.png" width="500" alt="">
      <p>2. Select CUDA Toolkit 11.7 (If you haven't installed the nvidia driver, select Driver. If you have already installed the nvidia driver, do not select driver), then select install, press Enter</p>
      <img src="https://i.328888.xyz/2023/03/31/iwVThJ.png" width="500" alt="">
      <p>3. Add environment variables</p>
      <p>Add the following content to ~/.bashrc</p>
      <pre><code># CUDA
  export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}
  export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}</code></pre>
      <p>Make it effective</p>
      <pre><code>source ~/.bashrc</code></pre>
      <h5>(3) Download cuDNN 8.4.1</h5>
      <p>Domestic: <a href="https://pan.baidu.com/s/1Gd_pSVzWfX1G7zCuqz6YYA">cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive.tar.xz</a> Extraction code: 57mg</p>
      <p>Overseas: <a href="https://github.com/YaoFANGUK/video-subtitle-extractor/releases/download/1.0.0/cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive.tar.xz">cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive.tar.xz</a></p>
      <h5>(4) Install cuDNN 8.4.1</h5>
      <pre><code> tar -xf cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive.tar.xz
   mv cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive cuda
   sudo cp ./cuda/include/* /usr/local/cuda-11.7/include/
   sudo cp ./cuda/lib/* /usr/local/cuda-11.7/lib64/
   sudo chmod a+r /usr/local/cuda-11.7/lib64/*
   sudo chmod a+r /usr/local/cuda-11.7/include/*</code></pre>
  </details>

  <details>
        <summary>Windows Users</summary>
        <h5>(1) Download CUDA 11.7</h5>
        <a href="https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_516.01_windows.exe">cuda_11.7.0_516.01_windows.exe</a>
        <h5>(2) Install CUDA 11.7</h5>
        <h5>(3) Download cuDNN v8.4.0 (April 1st, 2022), for CUDA 11.x</h5>
        <p><a href="https://github.com/YaoFANGUK/video-subtitle-extractor/releases/download/1.0.0/cudnn-windows-x86_64-8.4.0.27_cuda11.6-archive.zip">cudnn-windows-x86_64-8.4.0.27_cuda11.6-archive.zip</a></p>
        <h5>(4) Install cuDNN 8.4.0</h5>
        <p>
           Extract cuDNN, then copy the files under the bin, include, lib directories inside the cuda folder to the corresponding directories under C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\
        </p>
    </details>


- Install GPU version of Paddlepaddle:

  - windows:

      ```shell 
      python -m pip install paddlepaddle-gpu==2.4.2.post117 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
      ```

  - Linux:

      ```shell
      python -m pip install paddlepaddle-gpu==2.4.2.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
      ```

- Install GPU version of Pytorch:
      
  ```shell
  conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
  ```
  Alternatively use
  ```shell
  pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
  ```

- Install other dependencies:

  ```shell
  pip install -r requirements.txt
  ```


#### 4. Run the Program

- Run Graphical Interface (GUI)

```shell
python gui.py
```

- Run Command Line Version (CLI)

```shell
python ./backend/main.py
```

## Common Problems
1. What to do if the extraction speed is slow?

Modify the parameters in backend/config.py, which can significantly increase the removal speed
```python
MODE = InpaintMode.STTN  # Set to STTN algorithm
STTN_SKIP_DETECTION = True # Skip subtitle detection, skipping may cause subtitles that need to be removed to be missed or mistakenly damage video frames without subtitles
```

2. What to do if the video removal effect is not good?

Modify the parameters in backend/config.py, try different removal algorithms. Algorithm introduction:

> - InpaintMode.STTN algorithm: Works better for real-person videos, fast, can skip subtitle detection
> - InpaintMode.LAMA algorithm: Best effect for images, good effect for animated videos, moderate speed, cannot skip subtitle detection
> - InpaintMode.PROPAINTER algorithm: Requires a large amount of VRAM, slower speed, better effect for videos with very intense motion

- Use STTN algorithm

```python
MODE = InpaintMode.STTN  # Set to STTN algorithm
# Number of neighboring frames, increasing it will increase VRAM usage and improve the effect
STTN_NEIGHBOR_STRIDE = 10
# Reference frame length, increasing it will increase VRAM usage and improve the effect
STTN_REFERENCE_LENGTH = 10
# Set the maximum number of frames STTN algorithm can process simultaneously, setting it larger slows down the speed but improves the effect
# Ensure STTN_MAX_LOAD_NUM is greater than STTN_NEIGHBOR_STRIDE and STTN_REFERENCE_LENGTH
STTN_MAX_LOAD_NUM = 30
```
- Use LAMA algorithm
```python
MODE = InpaintMode.LAMA  # Set to LAMA algorithm # Note: Original comment said STTN, assuming LAMA was intended here based on context
LAMA_SUPER_FAST = False  # Ensure effect quality
```

> If you are not satisfied with the model's subtitle removal effect, you can check the training methods in the design folder, use the code in backend/tools/train to train, and then replace the old model with the trained model.

3. CondaHTTPError

Place the .condarc file from the project in the user directory (C:/Users/<your_username>). If the file already exists in the user directory, overwrite it.

Solution: https://zhuanlan.zhihu.com/p/260034241

4. 7z file extraction error

Solution: Upgrade the 7-zip extraction program to the latest version.

5. 4090 cannot run with CUDA 11.7

Solution: Use CUDA 11.8 instead

```shell
pip install torch==2.1.0 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

## Sponsorship

<img src="https://github.com/YaoFANGUK/video-subtitle-extractor/raw/main/design/sponsor.png" width="600">

| Donor                     | Cumulative Donation Amount | Sponsorship Tier |
|---------------------------|--------------------------|------------------|
| 坤V                        | 400.00 RMB               | Gold Sponsor     |
| Jenkit                    | 200.00 RMB               | Gold Sponsor     |
| 子车松兰                    | 188.00 RMB               | Gold Sponsor     |
| 落花未逝                    | 100.00 RMB               | Gold Sponsor     |
| 张音乐                    | 100.00 RMB               | Gold Sponsor     |
| 麦格                      | 100.00 RMB               | Gold Sponsor     |
| 无痕                      | 100.00 RMB               | Gold Sponsor     |
| wr                        | 100.00 RMB               | Gold Sponsor     |
| 陈                        | 100.00 RMB               | Gold Sponsor     |
| TalkLuv                   | 50.00 RMB                | Silver Sponsor   |
| 陈凯                      | 50.00 RMB                | Silver Sponsor   |
| Tshuang                   | 20.00 RMB                | Silver Sponsor   |
| 很奇异                     | 15.00 RMB                | Silver Sponsor   |
| 郭鑫                       | 12.00 RMB                | Silver Sponsor   |
| 生活不止眼前的苟且            | 10.00 RMB                | Bronze Sponsor   |
| 何斐                      | 10.00 RMB                | Bronze Sponsor   |
| 老猫                      | 8.80 RMB                 | Bronze Sponsor   |
| 伍六七                    | 7.77 RMB                 | Bronze Sponsor   |
| 长缨在手                    | 6.00 RMB                 | Bronze Sponsor   |
| 无忌                      | 6.00 RMB                 | Bronze Sponsor   |
| Stephen                   | 2.00 RMB                 | Bronze Sponsor   |
| Leo                       | 1.00 RMB                 | Bronze Sponsor   |
```
