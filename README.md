# Hotdog Object Detection

This project will fine tune the YOLOv8 model on a custom hotdog dataset using Ultralytics' Python API. The dataset provided consists of ~2000 images of hotdogs. Training was performed using Google Colab's Nvidia T4 GPU and took approximately 50 minutes. The project is designed as the application task for aUToronto Perception Subteam -- 2D Vision.

[YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
[A list of YOLOv8 Performance Metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics/)

**Please note that the model is only trained on 50 epochs for time efficiency while the recommended by Ultralytics is 300 epochs. This could be the cause for certain performance issues.**

## Overview

It is assumed that the user has the dataset prepared. The folder inclues a script to fine-tune the pre-trained model and to make inference calls on a particular image.

Using a dataset of only 2000 images, it is impossible to train an object detection model of any architecture from scratch. Therefore, fine-tuning on custom dataset is a more preferably and efficient technique when approaching this particular vision project.

YOLOv8 is the state of the art model for object detection, instance segmentation, and classification based on the original YOLO (you only look once) in 2015. We will use the pre-trained weights on the COCO dataset containing 80 classes.

To use YOLOv8, you must first install the Ultralytics package either using pip or conda:

```
pip install ultralytics
conda install -c conda-forge ultralytics
```

If you are working on a CUDA enabled environment, it is a good practice to install `ultralytics`, `pytorch`, and `pytorch-cuda` together to resolve any conflicts:

```
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
```

## Training

The training configuration (i.e. path to the train and validation dataset) is defined in `hotdog.yaml`, and this file should be modified based on the actual data file path on yoru file. _Note that by default it is assumed the dataset contains data/train and data/valid. Please modify this if it is not the case._ In order to do so, you can either use a text editor such as `vim` or `nano` or in zsh/bash shell run the following commands:

```
sed -i.bak 's|^path: /content/data|path: <your-path-to-dataset>|' hotdog.yaml
```

Then, you can run the following to load a YOLOv8 with pre-trained weights and fine tune it on the specified dataset:

```
python3 finetune.py
```

Training the model takes considerable amount of time, therefore it is recommended to do inference directly based on the already fine-tuned model that will be automatically loaded.

> If you wish to train it on Google Colab using their GPU, a Jupyter Notebook `yolov8.ipynb` has been attached so it can be directly uploaded.

For your convenience, the trained weights has been included in `weights/best.pt`. Other training output (e.g. images) are displayed in the PDF. Please note that if you wish to train it the weights will be created in `hotdog_detection/hotdog_model/weights/`

## Inference

Making inference calls are very easy with your desired image:

```
python3 inference.py --image=<path-to-image> --output=<path-to-output>
```

Currently, `--image` is limited to image path (i.e. no video) but it can easily be scaled up since YOLO supports video predictions as well. The command line arguments are processed by `argparse`. The predicted result (image) will be shown and detailed results will be printed (or logged). The result will also be saved in the `inferences/` directory or a specified path.

## Evaluation

This will obtain evaluation metrics based on the validation dataset in `data/val/images`. Ideally, the functionality should be included in `finetune.py`, but since the training script will take **very** long to run this is provided as an quicker standalone alternative.

```
python3 evaluate.py
```

> For detail regarding evaluation metrics, please refer to the attached metrics.pdf!

## Known Issues

It has come to our attention that **datapath for training and validation images are often prone to causing error when training**. When installing the ultralytics package, the default root path appended to your specified path will be set in `settings.yaml`. In the case of an error message (that looks like the one below), please modify the above path so that the final path is correct.

_Sample output of error message_:

```
Dataset 'hotdot.yaml' images not found ⚠️, missing path '/Users/xxx/Documents/xxx/datasets/data/valid/images'
Note dataset download directory is '/Users/xxx/Documents/xxx/datasets'. You can update this in '/Users/xxx/Library/Application Support/Ultralytics/settings.yaml'
```

## Next Steps

Comparing the results of this model against various other architecture such as Faster R-CNN would allow us to assess the strength and weakness of each design and figure out the best way to optimize for hotdog detection.

Since YOLOv8 also supports instance segmentation, we could consider creating new training sets to enable it to perform additional tasks than detection.

Additionally, adversarial attack such as FGSM can be implemented to test the robustness of the model, for example [as described in this papaer](https://arxiv.org/abs/2202.04781).
