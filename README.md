# DeepLabv3Plus-Pytorch

Modification of the [work](https://github.com/VainF/DeepLabV3Plus-Pytorch) by [Gongfan Fang](https://github.com/VainF) and [work](https://github.com/andreas-apg/deeplabv3-custom-dataset).


### 1. Available Architectures
Specify the model architecture with '--model ARCH_NAME' and set the output stride using '--output_stride OUTPUT_STRIDE'.

| DeepLabV3    |  DeepLabV3+        |
| :---: | :---:     |
|deeplabv3_resnet50|deeplabv3plus_resnet50|
|deeplabv3_resnet101|deeplabv3plus_resnet101|
|deeplabv3_mobilenet|deeplabv3plus_mobilenet ||
|deeplabv3_hrnetv2_48 | deeplabv3plus_hrnetv2_48 |
|deeplabv3_hrnetv2_32 | deeplabv3plus_hrnetv2_32 |


### 3. Training With a Custom Dataset

Pass the directories for the original images and the segmentation masks when calling train.py, with the arguments:
* `--train_dir` and `--train_seg_dir` for the training dataset;
* `--val_dir` and `--val_seg_dir` for the validation dataset.

The number of classes passed in the `--num_classes` argument should be the same as the n in your get_labels function: the number of objects of interest plus background.

For example, if you want to identify cats and dogs, you would have **3** classes: **background**, **cat** and **dog**.

In order to fix our masking for segmentation images you must run fixsegments.ipynb with the correct input directory and your desired output directory for all segmented images.

To train, you would use something like this:
Fresh training
```bash
python train.py --model deeplabv3plus_mobilenet --gpu_id 0 --lr 0.01 --crop_val --crop_size 255 --batch_size 16 --output_stride 16 --train_dir 'pathtotrainingdata' --train_seg_dir 'pathtosegmentedtrainingdata' --val_dir 'pathtovalidationdata' --val_seg_dir 'pathtosegmentedvalidationdata' --save_val_results --num_classes 9 --dataset custom --model_name 'nameofmodel' --loss_type focal_loss  --total_epochs 100
```
Train frm checkpoint
```bash
python train.py --model deeplabv3plus_mobilenet --gpu_id 0 --lr 0.01 --crop_val --crop_size 255 --batch_size 16 --output_stride 16 --train_dir datasets/data/noseg --train_seg_dir datasets/data/segfix --val_dir datasets/validdata/noseg --val_seg_dir datasets/validdata/segfix --save_val_results --num_classes 9 --dataset custom --model_name focalfix --loss_type focal_loss --ckpt checkpoints/best_deeplabv3plus_mobilenet_custom_os16_focalfix.pth --total_epochs 100
```
The .pth weight file will be saved under the `checkpoints` directory. The argument passed in --model_name will be concatenated along with the name of the model passed in the `--model` argument and the `--dataset` argument, with both a latest and a best weight files: 
* latest_deeplabv3plus_mobilenet_custom_cats-and-dogs.pth
* best_deeplabv3plus_mobilenet_custom_cats-and-dogs.pth



#### 3.2. Testing

Results will be saved at ./results.

```bash
python main.py --model deeplabv3plus_mobilenet --enable_vis --vis_port 28333 --gpu_id 0 --year 2012_aug --crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 --ckpt checkpoints/best_deeplabv3plus_mobilenet_voc_os16.pth --test_only --save_val_results
```

### 4. Prediction
You load the weight file using the `--ckpt` argument. Be sure to also pass the respective `--dataset` and `--model` that were used during training to generate those weights.

Single image:
```bash
python predict.py --input datasets/data/noseg/0593.png --model deeplabv3plus_mobilenet --ckpt checkpoints/best_deeplabv3plus_mobilenet_custom_os16_focalfix.pth --save_dir test_results --num_classes 9
```

Image folder:
```bash
python predict.py --input datasets/data/noseg --model deeplabv3plus_mobilenet --ckpt checkpoints/best_deeplabv3plus_mobilenet_custom_os16_focalfix.pth --save_dir test_results --num_classes 9
```
