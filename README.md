# ESPNetv2

This is one of the best semantic segmentation model, works pretty well on the Camvid dataset with 11 classes

## How to train ?

`CUDA_VISIBLE_DEVICES=0 python train_segmentation.py --model espnetv2 --s 2.0 --dataset sample --data_path ~/EdgeNet/vision_datasets/sample_dataset/ --batch-size 1 --crop_size 512 256 --model espnetv2 --s 1.5 --lr 0.009 --scheduler hybrid --clr-max 61 --epochs 100`

- Input: vision_datasets/sample_dataset/images
- Target: vision_datasets/sample_dataset/annotations
- First run the convert_to_gray.py file which will convert all the annotations into grayscale images
- Then add the image names and respective grayscale annotations pair in vision_datasets/sample_dataset/train.txt file
- Do the same for val.txt file
- Start running the training process

## How to test ?

1. `python test_seg.py`

This file run on the live webcam feed

2. `python tester.py`

This runs on the images on the images in the input folder

### References

The original work is done by Sachin Mehta, the repository can be found [here](https://github.com/sacmehta/EdgeNets)
