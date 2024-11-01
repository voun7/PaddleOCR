---
comments: true
---

# SRN

## 1. Introduction

Paper:
> [Towards Accurate Scene Text Recognition with Semantic Reasoning Networks](https://arxiv.org/abs/2003.12294#)
> Deli Yu, Xuan Li, Chengquan Zhang, Junyu Han, Jingtuo Liu, Errui Ding
> CVPR,2020

Using MJSynth and SynthText two text recognition datasets for training, and evaluating on IIIT, SVT, IC03, IC13, IC15,
SVTP, CUTE datasets, the algorithm reproduction effect is as follows:

| Model | Backbone        | config                                                                                                     | Acc    | Download link                                                                           |
|-------|-----------------|------------------------------------------------------------------------------------------------------------|--------|-----------------------------------------------------------------------------------------|
| SRN   | Resnet50_vd_fpn | [rec_r50_fpn_srn.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/rec/rec_r50_fpn_srn.yml) | 86.31% | [train model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r50_vd_srn_train.tar) |

## 2. Environment

Please refer to ["Environment Preparation"](../../ppocr/environment.en.md) to configure the PaddleOCR environment, and
refer to ["Project Clone"](../../ppocr/blog/clone.en.md)to clone the project code.

## 3. Model Training / Evaluation / Prediction

Please refer to [Text Recognition Tutorial](../../ppocr/model_train/recognition.en.md). PaddleOCR modularizes the code,
and training different recognition models only requires **changing the configuration file**.

### Training

Specifically, after the data preparation is completed, the training can be started. The training command is as follows:

```bash linenums="1"
# Single GPU training (long training period, not recommended)
python3 tools/train.py -c configs/rec/rec_r50_fpn_srn.yml

# Multi GPU training, specify the gpu number through the --gpus parameter
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_r50_fpn_srn.yml
```

### Evaluation

```bash linenums="1"
# GPU evaluation
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/rec_r50_fpn_srn.yml -o Global.pretrained_model={path/to/weights}/best_accuracy
```

### Prediction

```bash linenums="1"
# The configuration file used for prediction must match the training
python3 tools/infer_rec.py -c configs/rec/rec_r50_fpn_srn.yml -o Global.pretrained_model={path/to/weights}/best_accuracy Global.infer_img=doc/imgs_words/en/word_1.png
```

## 4. Inference and Deployment

### 4.1 Python Inference

First, the model saved during the SRN text recognition training process is converted into an inference
model. ( [Model download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r50_vd_srn_train.tar) ), you can use
the following command to convert:

```bash linenums="1"
python3 tools/export_model.py -c configs/rec/rec_r50_fpn_srn.yml -o Global.pretrained_model=./rec_r50_vd_srn_train/best_accuracy  Global.save_inference_dir=./inference/rec_srn
```

For SRN text recognition model inference, the following commands can be executed:

```bash linenums="1"
python3 tools/infer/predict_rec.py --image_dir="./doc/imgs_words/en/word_1.png" --rec_model_dir="./inference/rec_srn/" --rec_image_shape="1,64,256" --rec_char_type="ch" --rec_algorithm="SRN" --rec_char_dict_path="ppocr/utils/ic15_dict.txt" --use_space_char=False
```

### 4.2 C++ Inference

Not supported

### 4.3 Serving

Not supported

### 4.4 More

Not supported

## 5. FAQ

## Citation

```bibtex
@article{Yu2020TowardsAS,
  title={Towards Accurate Scene Text Recognition With Semantic Reasoning Networks},
  author={Deli Yu and Xuan Li and Chengquan Zhang and Junyu Han and Jingtuo Liu and Errui Ding},
  journal={2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020},
  pages={12110-12119}
}
```
