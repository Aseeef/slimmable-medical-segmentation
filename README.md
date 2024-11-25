# Introduction

PyTorch implementation of medical segmentation models, support multi-gpu training and validating, automatic mixed precision training, knowledge distillation, hyperparameter optimization using Optuna etc.  

# Requirements

torch == 1.8.1  
segmentation-models-pytorch  
torchmetrics  
albumentations  
loguru  
tqdm  
optuna == 4.0.0 (optional)  
optuna-integration == 4.0.0 (optional)  

If you find any version conflicts, see [requirements](./requirements.txt). This repo may also work with torch > 1.8.1, but it has not been verified yet.  

If you want a minimally reproducible environment, you may run
```
pip install -r requirements.txt
```

# Supported datasets

<details open><summary>Polyp</summary>

- [Kvasir](https://datasets.simula.no/kvasir-seg/)
- [CVC-ClinicDB](https://polyp.grand-challenge.org/CVCClinicDB/)
- [CVC-ColonDB](https://drive.google.com/drive/folders/1-gZUo1dgsdcWxSdXV9OAPmtGEbwZMfDY)
- [ETIS-LaribpolypDB](https://drive.google.com/drive/folders/10QXjxBJqCf7PAXqbDvoceWmZ-qF07tFi)

In order to obtain reproducible results, you may also download the train-val-test sets provided by DUCKNet's authors [here](https://drive.google.com/drive/folders/1kg9XImzrd9PpTtleQSz6l8uq82LV1sjV).  

</details>

# Supported models

<details><summary>DUCKNet</summary>

[Using DUCK-Net for polyp image segmentation](https://www.nature.com/articles/s41598-023-36940-5) [[codes](models/ducknet.py)]  

> Abstract: This paper presents a novel supervised convolutional neural network architecture, “DUCK-Net”, capable of effectively learning and generalizing from small amounts of medical images to perform accurate segmentation tasks. Our model utilizes an encoder-decoder structure with a residual downsampling mechanism and a custom convolutional block to capture and process image information at multiple resolutions in the encoder segment. We employ data augmentation techniques to enrich the training set, thus increasing our model's performance. While our architecture is versatile and applicable to various segmentation tasks, in this study, we demonstrate its capabilities specifically for polyp segmentation in colonoscopy images. We evaluate the performance of our method on several popular benchmark datasets for polyp segmentation, Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB, and ETIS-LARIBPOLYPDB showing that it achieves state-of-the-art results in terms of mean Dice coefficient, Jaccard index, Precision, Recall, and Accuracy. Our approach demonstrates strong generalization capabilities, achieving excellent performance even with limited training data.

</details>

<details><summary>UNet</summary>

[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) [[codes](models/unet.py)]  

> Abstract: There is large consent that successful training of deep networks requires many thousand annotated training samples. In this paper, we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. We show that such a network can be trained end-to-end from very few images and outperforms the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. Using the same network trained on transmitted light microscopy images (phase contrast and DIC) we won the ISBI cell tracking challenge 2015 in these categories by a large margin. Moreover, the network is fast. Segmentation of a 512x512 image takes less than a second on a recent GPU. The full implementation (based on Caffe) and the trained networks are available at [this http URL](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net).

</details>

<br>

If you want to use encoder-decoder structure with pretrained encoders, you may refer to: segmentation-models-pytorch[^smp]. This repo also provides easy access to SMP. Just modify the [config file](configs/my_config.py) to (e.g. if you want to train UNet with ResNet-101 backbone as teacher model to perform knowledge distillation)  

```
self.model = 'smp'
self.encoder = 'resnet101'
self.decoder = 'unet'
```

or use [command-line arguments](configs/parser.py)  

```
python main.py --model smp --encoder resnet101 --decoder unet
```

Details of the configurations can also be found in this [file](configs/parser.py).  

[^smp]: [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)  

# Knowledge Distillation

Currently only support the original knowledge distillation method proposed by Geoffrey Hinton.[^kd]  

[^kd]: [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)  

# Hyperparameter Optimization

This repo also supports hyperparameter optimization using Optuna.[^optuna] For example, if you have enough computing power and want to search hyperparameters using UNet, you may simply run

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 optuna_search.py
```

[^optuna]: [Optuna: A hyperparameter optimization framework](https://github.com/optuna/optuna)

# How to use

## DDP training (recommend)

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py
```

## DP training

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py
```

# Performances and checkpoints

## Kvasir

| Model					| Pretrained  | Params(M) | Dice(paper)<sup>3</sup> | mIoU(paper)<sup>3</sup> | Dice(my)<sup>4</sup> <br> val/test | mIoU(my)<sup>4</sup> <br> val/test | 
| --------------------- |:-----------:|:---------:|:-----------------------:|:-----------------------:|:----------------------------------:|:----------------------------------:|
| DUCKNet-17			| None		  | 40.1	  | 0.9343					| 0.8769				  | 0.9227/0.9515					   | 0.8612/0.9094						|
| DUCKNet-34			| None		  | 160.28	  | 0.9502					| 0.9051				  | 0.9230/0.9573					   | 0.8618/0.9198						|
| UNet-32<sup>1</sup>	| None		  | 8.63	  | 0.8655					| 0.7629				  | 0.9107/0.9444					   | 0.8421/0.8972						|
| smp-UNet<sup>2</sup>  | None		  | 14.33	  | n.a.					| n.a.					  | 0.9267/0.9556					   | 0.8677/0.9166						|
| smp-UNet				| ImageNet	  | 14.33	  | n.a.					| n.a.					  | 0.9341/0.9543					   | 0.8799/0.9144						|

## CVC-ClinicDB

| Model      | Pretrained | Params(M) | Dice(paper) | mIoU(paper) | Dice(my) <br> val/test | mIoU(my) <br> val/test | 
| ---------- |:----------:|:---------:|:-----------:|:-----------:|:----------------------:|:----------------------:|
| DUCKNet-17 | None		  | 40.1	  | 0.9450		| 0.8952	  | 0.9607/0.9434		   | 0.9262/0.8966			|
| DUCKNet-34 | None		  | 160.28	  | 0.9478		| 0.9009	  | 0.9674/0.9451		   | 0.9382/0.8995			|
| UNet-32	 | None		  | 8.63	  | 0.7631		| 0.6169	  | 0.9537/0.9424		   | 0.9140/0.8949			|
| smp-UNet	 | None		  | 14.33	  | n.a.		| n.a.		  | 0.9662/0.9503		   | 0.9361/0.9082			|
| smp-UNet	 | ImageNet	  | 14.33	  | n.a.		| n.a.		  | 0.9737/0.9566		   | 0.9497/0.9190			|

## CVC-ColonDB

| Model      | Pretrained | Params(M) | Dice(paper) | mIoU(paper) | Dice(my) <br> val/test | mIoU(my) <br> val/test | 
| ---------- |:----------:|:---------:|:-----------:|:-----------:|:----------------------:|:----------------------:|
| DUCKNet-17 | None		  | 40.1	  | 0.9353		| 0.8785	  | 0.9432/0.9357		   | 0.8968/0.8847			|
| DUCKNet-34 | None		  | 160.28	  | 0.9230		| 0.8571	  | 0.9390/0.9322		   | 0.8899/0.8790			|
| UNet-32	 | None		  | 8.63	  | 0.8032		| 0.7037	  | 0.9125/0.8966		   | 0.8486/0.8255			|
| smp-UNet	 | None		  | 14.33	  | n.a.		| n.a.		  | 0.9545/0.9498		   | 0.9157/0.9078			|
| smp-UNet	 | ImageNet	  | 14.33	  | n.a.		| n.a.		  | 0.9676/0.9658		   | 0.9388/0.9356			|

## ETIS-LaribpolypDB

| Model      | Pretrained | Params(M) | Dice(paper) | mIoU(paper) | Dice(my) <br> val/test | mIoU(my) <br> val/test | 
| ---------- |:----------:|:---------:|:-----------:|:-----------:|:----------------------:|:----------------------:|
| DUCKNet-17 | None		  | 40.1	  | 0.9324		| 0.8734	  | 0.8939/0.9013		   | 0.8223/0.8323			|
| DUCKNet-34 | None		  | 160.28	  | 0.9354		| 0.8788	  | 0.8805/0.8884		   | 0.8040/0.8142			|
| UNet-32	 | None		  | 8.63	  | 0.7984		| 0.6969	  | 0.8294/0.8218		   | 0.7403/0.7296			|
| smp-UNet	 | None		  | 14.33	  | n.a.		| n.a.		  | 0.9386/0.8997		   | 0.8896/0.8300			|
| smp-UNet	 | ImageNet	  | 14.33	  | n.a.		| n.a.		  | 0.9740/0.9706		   | 0.9504/0.9442			|

[<sup>1</sup>UNet-32 stands for vanilla UNet with base_channel=32. Please note that the original base_channel for UNet is 64. To reduce computation complexity and VRAM usuage, I used base_channel=32 in my experiments.]  
[<sup>2</sup>smp-UNet stands for smp model with decoder UNet. For simplicity, ResNet-18 is chosen as the decoder in my experiments.]  
[<sup>3</sup>These results were obtained from DUCKNet's paper]  
[<sup>4</sup>These results were obtained by training 400 epochs with crop-size 320x320]  

## Knowledge distillation

| Dataset			| Model      | kd_training | Dice <br> val/test | mIoU <br> val/test | 
| ----------------- |:----------:|:-----------:|:------------------:|:------------------:|
| Kvasir			| smp-UNet	 | teacher	   | 0.9341/0.9543		| 0.8799/0.9144	 	 |
|					| UNet-32	 | False	   | 0.9107/0.9444		| 0.8421/0.8972	 	 |
|					| UNet-32	 | True		   | 0.9144/0.9458		| 0.8478/0.8995	 	 |
| CVC-ClinicDB		| smp-UNet	 | teacher	   | 0.9737/0.9566		| 0.9497/0.9190	 	 |
|					| UNet-32	 | False	   | 0.9537/0.9424		| 0.9140/0.8949	 	 |
|					| UNet-32	 | True		   | 0.9570/0.9468		| 0.9197/0.9023	 	 |
| CVC-ColonDB		| smp-UNet	 | teacher	   | 0.9676/0.9658		| 0.9388/0.9356	 	 |
|					| UNet-32	 | False	   | 0.9125/0.8966		| 0.8486/0.8255	 	 |
|					| UNet-32	 | True		   | 0.9289/0.9131		| 0.8738/0.8496		 |
| ETIS-LaribpolypDB | smp-UNet	 | teacher	   | 0.9740/0.9706		| 0.9504/0.9442	 	 |
|					| UNet-32	 | False	   | 0.8294/0.8218		| 0.7403/0.7296	 	 |
|					| UNet-32	 | True		   | 0.8547/0.7988		| 0.7706/0.7023	 	 |

## Hyperparameter Optimization

| Model      | Dataset			 | Dice(paper) | Dice(random) <br> val/test | Dice(Optuna) <br> val/test | 
| ---------- |:-----------------:|:-----------:|:--------------------------:|:--------------------------:|
| UNet-32	 | Kvasir			 | 0.8655	   | 0.9107/0.9444				| 0.9235/0.9483				 |
| 			 | CVC-ClinicDB		 | 0.7631	   | 0.9537/0.9424				| 0.9679/0.9601				 |
| 			 | CVC-ColonDB		 | 0.8032	   | 0.9125/0.8966				| 0.9601/0.9529				 |
| 			 | ETIS-LaribpolypDB | 0.7984	   | 0.8294/0.8218				| 0.9653/0.9411				 |
| smp-UNet	 | Kvasir			 | n.a.		   | 0.9341/0.9543				| 0.9368/0.9624				 |
| 			 | CVC-ClinicDB		 | n.a.		   | 0.9737/0.9566				| 0.9769/0.9727				 |
| 			 | CVC-ColonDB		 | n.a.		   | 0.9676/0.9658				| 0.9758/0.9634				 |
| 			 | ETIS-LaribpolypDB | n.a.		   | 0.9740/0.9706				| 0.9796/0.9700				 |

[When using random search, the hyperparameters were chosen from the default config. For Optuna search, each experiment was performed 100 trials.]  

# Prepare the dataset

```
PolypDataset/
├── Kvasir-SEG/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── validation/
│   └── test/
├── CVC-ClinicDB/
├── CVC-ColonDB/
└── ETIS-LaribPolypDB/
```

# References