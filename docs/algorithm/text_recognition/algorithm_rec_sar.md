---
comments: true
---

# SAR

## 1. 算法简介

论文信息：
> [Show, Attend and Read: A Simple and Strong Baseline for Irregular Text Recognition](https://arxiv.org/abs/1811.00751)
> Hui Li, Peng Wang, Chunhua Shen, Guyu Zhang
> AAAI, 2019

使用MJSynth和SynthText两个文字识别数据集训练，在IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE数据集上进行评估，算法复现效果如下：

| 模型  | 骨干网络     | 配置文件                                                                                               | Acc    | 下载链接                                                                           |
|-----|----------|----------------------------------------------------------------------------------------------------|--------|--------------------------------------------------------------------------------|
| SAR | ResNet31 | [rec_r31_sar.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/rec/rec_r31_sar.yml) | 87.20% | [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/rec/rec_r31_sar_train.tar) |

注：除了使用MJSynth和SynthText两个文字识别数据集外，还加入了[SynthAdd](https://pan.baidu.com/share/init?surl=uV0LtoNmcxbO-0YA7Ch4dg)
数据（提取码：627x），和部分真实数据，具体数据细节可以参考论文。

## 2. 环境配置

请先参考[《运行环境准备》](../../ppocr/environment.md)配置PaddleOCR运行环境，参考[《项目克隆》](../../ppocr/blog/clone.md)
克隆项目代码。

## 3. 模型训练、评估、预测

请参考[文本识别教程](../../ppocr/model_train/recognition.md)。PaddleOCR对代码进行了模块化，训练不同的识别模型只需要*
*更换配置文件**即可。

### 训练

具体地，在完成数据准备后，便可以启动训练，训练命令如下：

```bash linenums="1"
#单卡训练（训练周期长，不建议）
python3 tools/train.py -c configs/rec/rec_r31_sar.yml

# 多卡训练，通过--gpus参数指定卡号
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_r31_sar.yml
```

### 评估

```bash linenums="1"
# GPU 评估， Global.pretrained_model 为待测权重
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/rec_r31_sar.yml -o Global.pretrained_model={path/to/weights}/best_accuracy
```

### 预测

```bash linenums="1"
# 预测使用的配置文件必须与训练一致
python3 tools/infer_rec.py -c configs/rec/rec_r31_sar.yml -o Global.pretrained_model={path/to/weights}/best_accuracy Global.infer_img=doc/imgs_words/en/word_1.png
```

## 4. 推理部署

### 4.1 Python推理

首先将SAR文本识别训练过程中保存的模型，转换成inference
model。（ [模型下载地址](https://paddleocr.bj.bcebos.com/dygraph_v2.1/rec/rec_r31_sar_train.tar) )，可以使用如下命令进行转换：

```bash linenums="1"
python3 tools/export_model.py -c configs/rec/rec_r31_sar.yml -o Global.pretrained_model=./rec_r31_sar_train/best_accuracy  Global.save_inference_dir=./inference/rec_sar
```

SAR文本识别模型推理，可以执行如下命令：

```bash linenums="1"
python3 tools/infer/predict_rec.py --image_dir="./doc/imgs_words/en/word_1.png" --rec_model_dir="./inference/rec_sar/" --rec_image_shape="3, 48, 48, 160" --rec_algorithm="SAR" --rec_char_dict_path="ppocr/utils/dict90.txt" --max_text_length=30 --use_space_char=False
```

### 4.2 C++推理

由于C++预处理后处理还未支持SAR，所以暂未支持

### 4.3 Serving服务化部署

暂不支持

### 4.4 更多推理部署

暂不支持

## 5. FAQ

## 引用

```bibtex
@article{Li2019ShowAA,
  title={Show, Attend and Read: A Simple and Strong Baseline for Irregular Text Recognition},
  author={Hui Li and Peng Wang and Chunhua Shen and Guyu Zhang},
  journal={ArXiv},
  year={2019},
  volume={abs/1811.00751}
}
```
