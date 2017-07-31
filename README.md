# TF_Deformable_Net

This is an tensorflow implementation of [Deformable Convolutional Network](https://arxiv.org/abs/1703.06211) in Faster R-CNN fashion. This project is largely built on [TFFRCNN](https://github.com/CharlesShang/TFFRCNN), the [original implementation in mxnet](https://github.com/msracver/Deformable-ConvNets) and many other upstream projects. This repository is only on test phase right now, any contributions helping with bugs and compatibility issues are welcomed.

* [TF_Deformable_Net](#tf_deformable_net)
* [TODO](#todo)
* [Requirements: software](#requirements-software)
* [Requirements: Hardware](#requirements-hardware)
* [Installation (sufficient for the demo)](#installation-sufficient-for-the-demo)
* [Demo](#demo)
   * [Download list](#download-list)
* [Training](#training)
* [Testing](#testing)
* [FAQ](#faq)
## TODO

- [x] Faster R-CNN
- [x] Trained on ResNet-50
- [x] More Networks
- [ ] Potential bugs


- [ ] R-FCN

     ​

## Requirements: software

Python 3 (Insufficient compatibility guaranteed for Python 2, though TFFRCNN is built on python 2, I use [ilovin's refactored version](https://github.com/ilovin/TFFRCNN_Python3) as base, and add some `__future__` imports, so any report on compatibility issues welcomed)

Tensorflow(1.0+) (Build from source recommended, or else you might need to check `./lib/cuda_config.h ` to fill in some options.)

matplotlib

python-opencv

easydict

scikit-image

cython

g++ 4.9(For gcc5 users, you should check make.sh and modify it as told.)

Cuda 8.0

## Requirements: Hardware

Any NVIDIA GPUs with at least 4GB memory should be OK.(Only single gpu mode supported, if you encounter any memory issue on a multi-gpu machine, try `export $CUDA_VISIBLE_DEVICE=$(the gpu id you want to use)`).

## Installation (sufficient for the demo)

1. Clone this repository
    ```Shell
    git clone https://github.com/Zardinality/TF_Deformable_Net.git
    cd TF_Deformable_Net
    ```

2. setup all the Cython module and gpu kernels (you might want to set -arch depending on the device you run on, note that `row_pooling` and `psroi_pooling` are not neccesserily needed in every sense, because `deformable_psroi_pooling` can work as them in many ways, also, tensorflow already has `tf.image.crop_and_resize` as a faster version for `row_pooling`).

    ```Shell
    cd ./lib
    make
    ```

## Demo

*After successfully completing [basic installation](#installation-sufficient-for-the-demo)*, you'll be ready to run the demo.

To run the demo
```Shell
cd $TF_Deformable_Net
python ./faster_rcnn/demo.py --model model_path
# e.g. python faster_rcnn/demo.py --model ~/tf_deformable_net/restore_output/Resnet50_iter_145000.ckpt
```
Also, for many people work on remote machine, an ipython notebook version demo and test are provided in case visually debug is needed. Except for that all arguments are made in a easydict object, the ipython notebook version demo and test are the same as the script version ones.

The demo performs detection using a ResNet50 network trained for detection on PASCAL VOC 2007.
, where model can be download below. Note that since TF 0.12 the checkpoint must contains more than one file, so you need to unzip the downloaded model to a folder whose path is model_path. Also, when you restore, be sure to set the checkpoint name to be the name you save them(no matter what suffix the checkpoint files now have).

### Download list

1. [VGG16 trained on ImageNet](https://drive.google.com/open?id=0ByuDEGFYmWsbNVF5eExySUtMZmM)
2. [Resnet50 trained on ImageNet](https://drive.google.com/file/d/0B_xFdh9onPagSWU1ZTAxUTZkZTQ/view?usp=sharing)
3. [Resnet50 Model(map@0.5 66%)](https://drive.google.com/file/d/0B6rLC-vrlfKFbHk1Si05YVZ0d3c/view?usp=sharing)

## Training

1. Download the training, validation, test data and VOCdevkit

    ```Shell
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
    ```

2. Extract all of these tars into one directory named `VOCdevkit`

    ```Shell
    tar xvf VOCtrainval_06-Nov-2007.tar
    tar xvf VOCtest_06-Nov-2007.tar
    tar xvf VOCdevkit_08-Jun-2007.tar
    ```

3. It should have this basic structure

    ```Shell
    $VOCdevkit/                           # development kit
    $VOCdevkit/VOCcode/                   # VOC utility code
    $VOCdevkit/VOC2007                    # image sets, annotations, etc.
    # ... and several other directories ...
    ```

4. Create symlinks for the PASCAL VOC dataset

    ```Shell
    cd $TF_Deformable_Net/data
    ln -s $VOCdevkit VOCdevkit2007
    ```

5. Download pre-trained model [VGG16](https://drive.google.com/open?id=0ByuDEGFYmWsbNVF5eExySUtMZmM) or [Resnet50](https://drive.google.com/file/d/0B_xFdh9onPagSWU1ZTAxUTZkZTQ/view?usp=sharing) and put it in the path `./data/pretrain_model`

6. Run training scripts 

    ```Shell
    cd $TF_Deformable_Net
    # for resnet-50
    python ./faster_rcnn/train_net.py --gpu 0 --weights ./data/pretrain_model/Resnet50.npy --imdb voc_2007_trainval --iters 70000 --cfg  ./experiments/cfgs/faster_rcnn_end2end_resnet.yml --network Resnet50_train --set EXP_DIR exp_dir
    # for vggnet
    python ./faster_rcnn/train_net.py --gpu 0 --weights ./data/pretrain_model/VGG_imagenet.npy --imdb voc_2007_trainval --iters 70000 --cfg  ./experiments/cfgs/faster_rcnn_end2end.yml --network VGGnet_train --set EXP_DIR exp_dir
    ```
    Or equivalently, edit scripts in `./experiments/scripts`, and start training by running shell scripts. For example:

    ```shell
    # for resnet-50
    ./experiments/scripts/faster_rcnn_voc_resnet.sh
    # for vggnet
    ./experiments/scripts/faster_rcnn_vggnet.sh
    ```

7. Run a profiling

    ```Shell
    cd $TF_Deformable_Net
    # install a visualization tool
    sudo apt-get install graphviz  
    ./experiments/profiling/run_profiling.sh 
    # generate an image ./experiments/profiling/profile.png
    ```

## Testing

After training, you could run scripts in `./experiments/eval`to evaluate on VOC2007. Or by running `./faster_rcnn/test_net.py` directly.

```shell
# for resnet-50
./experiments/eval/voc2007_test_res.sh
# for vggnet
./experiments/scripts/voc2007_test_vgg.sh
```

## FAQ

1. cudaCheckError() failed : invalid device function. 
   Check ./lib/make.sh and change the -arch flag accordingly. (Credit to [here](https://github.com/smallcorgi/Faster-RCNN_TF/issues/19))

2. undefined symbol: _ZN10tensorflow8internal21CheckOpMessageBuilder9NewStringB5cxx11Ev
   If you use gcc5 to build, modify `make.sh` to gcc5 version(simply adding a `-D_GLIBCXX_USE_CXX11_ABI=0` flag as pointed out in [this issue](https://github.com/tensorflow/tensorflow/issues/1569)).
