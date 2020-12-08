# Region Proportion Regularized Inference (RePRI) for Few-Shot Segmentation

In this repo, we provide the code for our paper : "Few-Shot Segmentation Without Meta-Learning: A Good Transductive Inference Is All You Need?", available at :

<img src="figures/intro_image.png" width="800" height="400"/>


## Getting Started

### Minimum requirements

1. Software :
+ torch==1.7.0
+ numpy==1.18.4
+ cv2==4.2.0
+ pyyaml==5.3.1

For both training and testing, metrics monitoring is done through **visdom_logger** (https://github.com/luizgh/visdom_logger). To install this package, simply clone the repo and install it with pip:

 ```
 git clone https://github.com/luizgh/visdom_logger.git
 pip install -e visdom_logger
 ```

 2. Hardware : A 11 GB+ CUDA-enabled GPU

### Download data

We provide the versions of Pascal-VOC 2012 and MS-COCO 2017 used in this work at https://drive.google.com/file/d/1Lj-oBzBNUsAqA9y65BDrSQxirV8S15Rk/view?usp=sharing. You can download the full .zip and directly extract it at the root of this repo.

The train/val splits are directly provided in lists/. How they were obtained is explained at https://github.com/Jia-Research-Lab/PFENet

### Download pre-trained models

#### Pre-trained backbones
First, you will need to download the ImageNet pre-trained backbones at https://drive.google.com/drive/folders/1Hrz1wOxOZm4nIIS7UMJeL79AQrdvpj6v and put them under initmodel/. These will be used if you decide to train your models from scratch.

#### Pre-trained models
We directly provide the full pre-trained models at https://drive.google.com/file/d/1iuMAo5cJ27oBdyDkUI0JyGIEH60Ln2zm/view?usp=sharing. You can download them and directly extract them at the root of this repo. This includes Resnet50 and Resnet101 backbones on Pascal-5i, and Resnet50 on Coco-20i.

## Overview of the repo

Data are located in data/. All the code is provided in src/. Default configuration files can be found in config_files/. Training and testing scripts are located in scripts/. Lists/ contains the train/validation splits for each dataset.


## Training (optional)

If you want to use the pre-trained models, this step is optional. Otherwise, you can train your own models from scratch with the scripts/train.sh script, as follows.

```python
bash scripts/train.sh {data} {fold} {[gpu_ids]} {layers}
```
For instance, if you want to train a Resnet50-based model on the fold-0 of Pascal-5i on GPU 1, use:
```python
bash scripts/train.sh pascal 0 [1] 50
```

Note that this code supports distributed training. If you want to train on multiple GPUs, you may simply replace [1] in the previous examples with the list of gpus_id you want to use.


## Testing

To test your models, use the scripts/test.sh script, the general synthax is:
```python
bash scripts/test.sh {data} {shot} {[gpu_ids]} {layers}
```
This script will test successively on all folds of the current dataset. Below are presented specific commands for several experiments.


### Pascal-5i

Results :
|(1 shot/5 shot)|   Arch     | Fold-0 	   | Fold-1 	 | Fold-2 	   | Fold-3      | Mean 		|
| 	   ---      |    ---     |      ---    |	   ---   |	   ---     |    ---      |  ---  		|
| RePRI       	| Resnet-50  | 59.8 / 64.6 | 68.3 / 71.4 | 62.1 / 71.1 | 48.5 / 59.3 | 59.7 / 66.6	|
| Oracle-RePRI	| Resnet-50  | 72.4 / 75.1 | 78.0 / 80.8 | 77.1 / 81.4 | 65.8 / 74.4 | 73.3 / 77.9  |
| RePRI       	| Resnet-101 | 59.6 / 66.2 | 68.3 / 71.4 | 62.2 / 67.0 | 47.2 / 57.7 | 59.4 / 65.6	|
| Oracle-RePRI	| Resnet-101 | 73.9 / 76.8 | 79.7 / 81.7 | 76.1 / 79.5 | 65.1 / 74.5 | 73.7 / 78.1  |

Command:
```python
bash scripts/test.sh pascal 1 [0] 50  # 1-shot
bash scripts/test.sh pascal 5 [0] 50  # 5-shot
```

### Coco-20i

Results :
|(1 shot/5 shot)|   Arch     | Fold-0 	   | Fold-1 	 | Fold-2 	   | Fold-3      | Mean 		|
| 	   ---      |    ---     |      ---    |	   ---   |	   ---     |    ---      |  ---			|
| RePRI       	| Resnet-50  | 32.0 / 39.3 | 38.7 / 45.4 | 32.7 / 39.7 | 33.1 / 41.8 | 34.1/41.6    |
| Oracle-RePRI	| Resnet-50  | 49.3 / 51.5 | 51.4 / 60.8 | 38.2 / 54.7 | 41.6 / 55.2 |	45.1 / 55.5 |

Command :
```python
bash scripts/test.sh coco 1 [0] 50  # 1-shot
bash scripts/test.sh coco 5 [0] 50  # 5-shot
```

### Coco-20i -> Pascal-VOC


The folds used for cross-domain experiments are presented in the image below:
<img src="figures/coco2pascal.png" width="800" height="200"/>

Results :

|(1 shot/5 shot)|   Arch     | Fold-0 	   | Fold-1 	 | Fold-2 	   | Fold-3      | Mean 		|
| 	   ---      |    ---     |      ---    |	   ---   |	   ---     |    ---      |  --- 		|
| RePRI       	| Resnet-50  | 52.8 / 57.7 | 64.0 / 66.1 | 64.1 / 67.6 | 71.5 / 73.1 | 63.1 / 66.2  |
| Oracle-RePRI	| Resnet-50  | 69.6 / 73.5 | 71.7 / 74.9 | 77.6 / 82.2 | 86.2 / 88.1 | 76.2 / 79.7  |


Command :
```python
bash scripts/test.sh coco2pascal 1 [0] 50  # 1-shot
bash scripts/test.sh coco2pascal 5 [0] 50  # 5-shot
```



## Monitoring metrics

For both training and testing, you can monitor metrics using visdom_logger (https://github.com/luizgh/visdom_logger). To install this package, simply clone the repo and install it with pip:
 ```
 git clone https://github.com/luizgh/visdom_logger.git
 pip install -e visdom_logger
 ```
 Then, you need to start a visdom server with:
 ```
 python -m visdom.server -port 8098
 ```

Finally, add the line visdom_port 8098 in the options in scripts/train.sh or scripts/test.sh, and metrics will be displayed at this port. You can monitor them through your navigator.

## Contact

For further questions or details, please post an issue or directly reach out to Malik Boudiaf (malik.boudiaf.1@etsmtl.net)


## Acknowledgments

We gratefully thank the authors of https://github.com/Jia-Research-Lab/PFENet, as well as https://github.com/hszhao/semseg from which some parts of our code are inspired.


