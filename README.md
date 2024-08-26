Comparison for image upscaling models and algorithms

| Model/Algorithm | Pros | Cons                                                                   | Example | 
|-----------------|------|------------------------------------------------------------------------|---------|
| NAFNet          |      | Pre-trained models only for 4x, 2x<br/>Not effective for small images  |         |
|                 |      |                                                                        |         |
|                 |      |                                                                        |         |

# Install
<details open>
    <summary>
        <b>Expand</b>
    </summary>

## Repository
```shell
$ git clone https://github.com/isbicf/image_upscaling.git
$ cd image_upscaling
$ git flow init
$ git branch
```

## Install Python and Packages
```shell
# Python 3.12
$ sudo add-apt-repository ppa:deadsnakes/ppa
$ sudo apt update
$ sudo apt install python3.12 python3.12-venv python3.12-dev

# Virtual environment
$ python3.12 -m venv venv
$ source venv/bin/activate

# Packages
(venv) $ pip install --upgrade pip
(venv) $ pip install numpy
(venv) $ pip install opencv-python
(venv) $ pip install pillow
(venv) $ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
(venv) $ pip install psutil
(venv) $ pip install gpustat
(venv) $ pip install tqdm
(venv) $ pip install PyYAML
```
</details>

# NAFNet
- An image restoration model
<details open>
    <summary>
        <b>Expand</b>
    </summary>

- [GitHub](https://github.com/megvii-research/NAFNet)
- Licence: MIT

## Install 
### Download 
- Download the source code from [NAFNet GitHub](https://github.com/megvii-research/NAFNet) \
  <img src="./upscaling/nafnet/Download NAFNet.png">

- Decompress the zip file and move to <project home>/upscaling\
  i.e. $ mv ~/Download/NAFNet_main /projects/image_upscaling/NAFNet

### Install Packages
```shell
(venv) $ pip install lmdb
(venv) $ pip install scipy
(venv) $ pip install scikit-image
```

### Clean (Uninstall) Packages
```shell
(venv) $ pip uninstall lmdb -y
(venv) $ pip uninstall scipy -y

# sckikit-image
(venv) $ pip uninstall imageio -y
(venv) $ pip uninstall lazy_loader -y
(venv) $ pip uninstall packaging -y
(venv) $ pip uninstall scikit-image -y
(venv) $ pip uninstall tifffile -y
```

## Download Pretrained Models
### Restoration Model
1. Download [NAFNet-REDS-width64.pth](https://drive.google.com/file/d/14D4V4raNYIOhETfcuuLI3bGLB-OYIv6X/view)
2. Move the pretrained model to <NAFNetHome>/experiments/pretrained_models\
   i.e. $ mv ~/Downloads/NAFNet-REDS-width64.pth /projects/image_upscaling/NAFNet/experiments/pretrained_models

### Super-Resolution Models (4x)
1. Download [NAFSSR-L_4x.pth](https://drive.google.com/file/d/1TIdQhPtBrZb2wrBdAp9l8NHINLeExOwb/view)
2. Move the pretrained model to <NAFNetHome>/experiments/pretrained_models\
   i.e. $ mv ~/Downloads/NAFNet-REDS-width64.pth /projects/image_upscaling/NAFNet/experiments/pretrained_models

## Usage 
### De-blur
```python
import cv2
from upscaling.nafnet.nafnet import Debluring

img = cv2.imread('<file path>')
nafnet_deblurer = Debluring()
deblured = nafnet_deblurer.deblur(img)
```

### Super-resolution
```python
import cv2
from upscaling.nafnet.nafnet import SuperResolution

img = cv2.imread('<file path>')
# todo;
```

### Testing for customised module (upscaling.nafnet) 
- Run a test
  ```shell
  (venv) $ python -m upscaling.nafnet.tester
  (venv) $ ls -l data/nafnet
  ```
- View the output images and verify \
  i.e.\
  ![]()
  - The first image is the original image
  - The second is a 4x-resized image by OpenCV
  - The third is a de-blurred image of the second
  - The last is a 4x-resized image by the NAFNet super-resolution model

</details>

# Testing Device
<details open>
    <summary>
        <b>Expand</b>
    </summary>

</details>
