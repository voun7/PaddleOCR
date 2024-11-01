---
comments: true
---

# CRNN

## 1. Introduction

Paper:
> [An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717)

> Baoguang Shi, Xiang Bai, Cong Yao

> IEEE, 2015

Using MJSynth and SynthText two text recognition datasets for training, and evaluating on IIIT, SVT, IC03, IC13, IC15,
SVTP, CUTE datasets, the algorithm reproduction effect is as follows:

| Model | Backbone    | ACC    | config                                                                                                                                       | Download link                                                                                     |
|-------|-------------|--------|----------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| ---   | ---         | ---    | ---                                                                                                                                          | ---                                                                                               |
| CRNN  | Resnet34_vd | 81.04% | [configs/rec/rec_r34_vd_none_bilstm_ctc.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/rec/rec_r34_vd_none_bilstm_ctc.yml) | [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_none_bilstm_ctc_v2.0_train.tar) |
| CRNN  | MobileNetV3 | 77.95% | [configs/rec/rec_mv3_none_bilstm_ctc.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/rec/rec_mv3_none_bilstm_ctc.yml)       | [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_none_bilstm_ctc_v2.0_train.tar)    |

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
python3 tools/train.py -c configs/rec/rec_r34_vd_none_bilstm_ctc.yml

# Multi GPU training, specify the gpu number through the --gpus parameter
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_r34_vd_none_bilstm_ctc.yml
```

### Evaluation

```bash linenums="1"
# GPU evaluation
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/rec_r34_vd_none_bilstm_ctc.yml -o Global.pretrained_model={path/to/weights}/best_accuracy
```

### Prediction

```bash linenums="1"
# The configuration file used for prediction must match the training
python3 tools/infer_rec.py -c configs/rec/rec_r34_vd_none_bilstm_ctc.yml -o Global.pretrained_model={path/to/weights}/best_accuracy Global.infer_img=doc/imgs_words/en/word_1.png
```

## 4. Inference and Deployment

### 4.1 Python Inference

First, the model saved during the CRNN text recognition training process is converted into an inference
model. ( [Model download link](https://paddleocr.bj.bcebos.com/dygraph_v2.1/rec/rec_r31_CRNN_train.tar) ), you can use
the following command to convert:

```bash linenums="1"
python3 tools/export_model.py -c configs/rec/rec_r34_vd_none_bilstm_ctc.yml -o Global.pretrained_model=./rec_r34_vd_none_bilstm_ctc_v2.0_train/best_accuracy  Global.save_inference_dir=./inference/rec_crnn
```

For CRNN text recognition model inference, the following commands can be executed:

```bash linenums="1"
python3 tools/infer/predict_rec.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./inference/rec_crnn/" --rec_image_shape="3, 32, 100" --rec_char_dict_path="./ppocr/utils/ic15_dict.txt"
```

### 4.2 C++ Inference

With the inference model prepared, refer to the [cpp infer](../../ppocr/infer_deploy/cpp_infer.en.md) tutorial for C++
inference.

### 4.3 Serving

With the inference model prepared, refer to the [pdserving](../../ppocr/infer_deploy/paddle_server.en.md) tutorial for
service deployment by Paddle Serving.

### 4.4 More

More deployment schemes supported for CRNN:

- Paddle2ONNX: with the inference model prepared, please refer to
  the [paddle2onnx](../../ppocr/infer_deploy/paddle2onnx.en.md) tutorial.

## 5. FAQ

## Citation

```bibtex
@ARTICLE{7801919,
  author={Shi, Baoguang and Bai, Xiang and Yao, Cong},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={An End-to-End Trainable Neural Network for Image-Based Sequence Recognition and Its Application to Scene Text Recognition},
  year={2017},
  volume={39},
  number={11},
  pages={2298-2304},
  doi={10.1109/TPAMI.2016.2646371}}
```
