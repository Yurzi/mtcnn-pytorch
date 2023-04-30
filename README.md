# MTCNN-PyTorch

A pytorch implementation for mtcnn. It is based on the paper *Zhang, K et al.(2016)*[[ZHANG2016]](#Reference)

## Preparing the Environment

### Use virtual environment 

First, I recommend you to use virtual environment. So you need to install conda on your PC. If you don’t want it to eat a lot of your disk space, [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is better.

Step follow to create and activate a virtual environment with conda.
```shell
# python=3.10 will install latest 3.10.* python, you can try other python version.
conda create --name mtcnn python=3.10
```

Access to virtual environment named mtcnn.
```shell
conda activate mtcnn
```
Then you will be ready to proceed actual installation for environment.

### Install dependencies

First. You need to install PyTorch. Recommendly, using conda to do it.
```shell
# ref: https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Second. Install poetry with conda.
```shell
# ref: https://anaconda.org/conda-forge/poetry
conda install -c conda-forge poetry
```

*Additional:* 
*1. if you are stuck in conda slowly installation process, your can try mamba to boost it.*
*2. poetry is project mamager to help dependencies management, but it can not handle multi-version python*


Finally. Use poetry to install all rest dependencies.
```shell
poetry install
```

## Dataset

### Dataset Structure

You need to follow the appropriate structure to make the code run correctly.

**Actually**, you can just provide `raw` folder with `annotations.txt`, the other folder will be generate automatically, but you can also generate them from `raw` manually by using script under the `tools` folder

You can configure the partition ratio by using a configuration file(described below) when you generate the dataset from `raw`.

```shell
dataset
    ├── eval                (can be generated)
    │   ├── annotations.txt
    │   └── images
    ├── onet                (can be generated)
    │   ├── annotations.txt
    │   └── images
    ├── pnet                (can be generated)
    │   ├── annotations.txt
    │   └── images
    ├── raw
    │   ├── annotations.txt
    │   ├── eval.txt        (can be generated)
    │   ├── images
    │   ├── test.txt        (can be generated)
    │   └── train.txt       (can be generated)
    ├── rnet                (can be generated)
    │   ├── annotations.txt
    │   └── images
    └── test                (can be generated)
        ├── annotations.txt
        └── images
```
### Generate From Raw Manually

You can use the tool script to complete the dataset manually.

The script will manage to finish successfully with fallback mechanism
- if raw folder don't has partition the script will fallback to use configs/config_name.py's settings
- if --config is not set the script will fallback to default.py under configs folder
- if --path is not seth the script will fallback to config.py 's settings
- if default.py is not existed, a exception will be raised

```shell
python tools/dataset/completion.py [--config config_name] [--path path/to/dataset_dir]
```

### Dataset Annotation File Detail

All annotation files follow the similar format.

For raw.

If there are multi object in a picture, list them in multi line. If a picture don't have boundingbox or landmark, please left the place empty

```shell
# annotations.txt
# image_name bbox[] landmark[]
# boundingbox[]
# left_top_x/y normalized by raw picture's width and height.
# width and height are also normalized by raw picture's width and height.
# landmark[]
# l1_x/y is relative to bbox's left_top and normalized by bbox's width and height.
# l[2-5]_x/y is offset relative to l1_x/y and normalized by bbox's width and height .

xxxx.jpg left_top_x left_top_y width height l1_x l1_y l2_x l2_y l3_x l3_y l4_x l4_y l5_x l5_y
xxxx.jpg left_top_x left_top_y width height l1_x l1_y l2_x l2_y l3_x l3_y l4_x l4_y l5_x l5_y
yyyy.jpg left_top_x left_top_y width height l1_x l1_y l2_x l2_y l3_x l3_y l4_x l4_y l5_x l5_y
zzzz.jpg

```

For p|o|rnet.
```shell
# annotations.txt
# image_name classification(0|1|2) gt_bbox[] gt_landmark[]
# classification(0|1|2)
# 0 = negative; 1 = positive; 2 = part
# if classification is negative, the gt_bbox[] and gt_landmark[] will be ignored.
# if classification is part, the gt_landmark[] will be ignored.
# gt_bbox[]
# left_top_x/y is relative offset to cropped picture's left_top normalized by raw picture's width and height.
# width and height are normalized by raw picture's width and height.
# landmark[]
# l1_x/y is relative to gt_bbox's left_top and normalized by gt_bbox's width and height.
# l[2-5]_x/y is offset relative to l1_x/y and normalized by gt_bbox's width and height.


xxxx.jpg 2 left_top_x left_top_y width height l1_x l1_y l2_x l2_y l3_x l3_y l4_x l4_y l5_x l5_y
yyyy.jpg 1 left_top_x left_top_y width height l1_x l1_y l2_x l2_y l3_x l3_y l4_x l4_y l5_x l5_y
zzzz.jpg 0 left_top_x left_top_y width height l1_x l1_y l2_x l2_y l3_x l3_y l4_x l4_y l5_x l5_y
```

## Train
If your dataset has been perpared, you can use train.py to train all three net  sequentially.
```shell
python mtcnn/train.py --config config_name [--resume]
```
Or, you can use tran_(p|r|o)net.py to train echo net seprately.

```shell
python mtcnn/train_(p|r|o)net.py --config config_name [--resume]
```

## Evaluation

Use eval.py to eval your model.
```shell
python mtcnn/eval.py --config config_name
```

## Inference
Use inference.py to get a predication result
```shell
python mtcnn/inference.py --config config_name
```
## Reference

| KEY         | INFO                                                         |
| ----------- | ------------------------------------------------------------ |
| [ZHANG2016] | Zhang, K., Zhang, Z., Li, Z., and Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. IEEE  Signal Processing Letters, 23(10):1499–1503. [arXiv:1604.02878](https://arxiv.org/abs/1604.02878) |
