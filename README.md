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


## Train


## Evaluation


## Inference


## Reference

| KEY         | INFO                                                         |
| ----------- | ------------------------------------------------------------ |
| [ZHANG2016] | Zhang, K., Zhang, Z., Li, Z., and Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. IEEE  Signal Processing Letters, 23(10):1499–1503. [arXiv:1604.02878](https://arxiv.org/abs/1604.02878) |
