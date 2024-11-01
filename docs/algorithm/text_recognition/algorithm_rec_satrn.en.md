---
comments: true
---

# SATRN

## 1. Introduction

论文信息：
> [On Recognizing Texts of Arbitrary Shapes with 2D Self-Attention](https://arxiv.org/abs/1910.04396)
> Junyeop Lee, Sungrae Park, Jeonghun Baek, Seong Joon Oh, Seonghyeon Kim, Hwalsuk Lee
> CVPR, 2020
> Using MJSynth and SynthText two text recognition datasets for training, and evaluating on IIIT, SVT, IC03, IC13, IC15,
> SVTP, CUTE datasets, the algorithm reproduction effect is as follows:

| Model | Backbone   | config | Acc                                                                                                        | Download link                                                    |
|-------|------------|--------|------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| SATRN | ShallowCNN | 88.05% | [configs/rec/rec_satrn.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/rec/rec_satrn.yml) | [训练模型](https://pan.baidu.com/s/10J-Bsd881bimKaclKszlaQ?pwd=lk8a) |

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
python3 tools/train.py -c configs/rec/rec_satrn.yml
# Multi GPU training, specify the gpu number through the --gpus parameter
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_satrn.yml
```

### Evaluation

```bash linenums="1"
# GPU evaluation
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/rec_satrn.yml -o Global.pretrained_model={path/to/weights}/best_accuracy
```

### Prediction

```bash linenums="1"
# The configuration file used for prediction must match the training
python3 tools/infer_rec.py -c configs/rec/rec_satrn.yml -o Global.pretrained_model={path/to/weights}/best_accuracy Global.infer_img=doc/imgs_words/en/word_1.png
```

## 4. Inference and Deployment

### 4.1 Python Inference

First, the model saved during the SATRN text recognition training process is converted into an inference
model. ( [Model download link](https://pan.baidu.com/s/10J-Bsd881bimKaclKszlaQ?pwd=lk8a) ), you can use the following
command to convert:

```bash linenums="1"
python3 tools/export_model.py -c configs/rec/rec_satrn.yml -o Global.pretrained_model=./rec_satrn_train/best_accuracy  Global.save_inference_dir=./inference/rec_satrn
```

For SATRN text recognition model inference, the following commands can be executed:

```bash linenums="1"
python3 tools/infer/predict_rec.py --image_dir="./doc/imgs_words/en/word_1.png" --rec_model_dir="./inference/rec_satrn/" --rec_image_shape="3, 48, 48, 160" --rec_algorithm="SATRN" --rec_char_dict_path="ppocr/utils/dict90.txt" --max_text_length=30 --use_space_char=False
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
@article{lee2019recognizing,
      title={On Recognizing Texts of Arbitrary Shapes with 2D Self-Attention},
      author={Junyeop Lee and Sungrae Park and Jeonghun Baek and Seong Joon Oh and Seonghyeon Kim and Hwalsuk Lee},
      year={2019},
      eprint={1910.04396},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
