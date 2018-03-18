# Codes for *PFNet: A Novel Part Fusion Network for Fine-grained Visual Categorization*
This repository holds the PyTorch(V0.3.0) code for PFNet.

## Introduction

We propose a novel and simple part fusion network (PFNet) to effectively use discriminative image parts for classification. It consists of a part feature extractor to get part features and a two-level loss to train part-level and image-level features simultaneously. The loss consists of part attention loss and image loss. Unlike previous attention mechanisms based on intermediary feature maps, part attention loss embeds attention in loss function according to different parts' characteristics. It enables the PFNet to make better use of easy parts, hard parts and background parts. Combination with image loss further improves accuracy. PFNet does not need extra annotations and can be trained end to end. It achieves high accuracies on three popular challenging datasets CUB-200-2011 (85.1\%), Stanford Cars (93.2\%) and FGVC-Aircraft (90.4\%), which are higher than or comparable with the best reported (without using bounding box or part annotations).

![alt text](https://github.com/MichaelLiang12/PFNet-FGVC/blob/master/pic/PFNet.jpg "visualization")

## Prepare Datasets

Prepare the corresponding datasets ([CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [Stanford Cars](http://ai.stanford.edu/~jkrause/cars/car_dataset.html) or [FGVC-Aircraft](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)) before training PFNet. For quick start, you can download the dataset [Stanford Cars](http://ai.stanford.edu/~jkrause/cars/car_dataset.html), proposed rois files [car_rois500.tar.gz](https://drive.google.com/open?id=18DWMrK2WVEMGzRdMpgqgNiRbWOTtRwnP) and prepared train/test split file [car_splits.tar.gz](https://drive.google.com/open?id=18DWMrK2WVEMGzRdMpgqgNiRbWOTtRwnP). Unzip these files and organize them in the current working directory as follows:
```
-car
--car_ims
---000001.jpg

--car_rois500
---car_ims
----000001.txt

--split
---Acura Integra Type R 2001_test.txt
```

For part proposal, we also provide codes for generating part proposals using [Selective Search Window](https://koen.me/research/selectivesearch/). Please refer to the guide provide in our `part proposal` directory.



## Usage

1, Download this repo recursively:
```shell
git clone --recursive https://github.com/MichaelLiang12/PFNet-FGVC.git
```
2, Build RoiPooling module

Please follow the instuctions in [pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn#installation). We use the RoiPooling module implemented by them. Note that if you also use `Ubuntu14.04+Cuda8.0+TitanX`, you might not need to compile again.


3, Run `PFNet_train_test.py`

You can modify fundamental parameters in the `main()` function. The training process might be like follows. By setting `args.evaluate = True`, you can download [our model](https://drive.google.com/open?id=18DWMrK2WVEMGzRdMpgqgNiRbWOTtRwnP) and test it directly. 

![alt text](https://github.com/MichaelLiang12/PFNet-FGVC/blob/master/pic/dog_loss_acc1.png "visualization")

## Citation
For Selective Search Window and RoiPooling module.
```
@article{uijlings2013selective,
  title={Selective search for object recognition},
  author={Uijlings, Jasper RR and Van De Sande, Koen EA and Gevers, Theo and Smeulders, Arnold WM},
  journal={International Journal of Computer Vision},
  volume={104},
  number={2},
  pages={154--171},
  year={2013},
  publisher={Springer}
}

@article{chen17implementation,
    Author = {Xinlei Chen and Abhinav Gupta},
    Title = {An Implementation of Faster RCNN with Study for Region Sampling},
    Journal = {arXiv preprint arXiv:1702.02138},
    Year = {2017}
}
```
