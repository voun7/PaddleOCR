# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import os
import sys

__dir__ = os.path.dirname(__file__)

sys.path.append(os.path.join(__dir__, ""))

import cv2
import logging
import numpy as np
from pathlib import Path
import base64
from io import BytesIO
import pprint
from PIL import Image


def _import_file(module_name, file_path, make_importable=False):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if make_importable:
        sys.modules[module_name] = module
    return module


tools = _import_file(
    "tools", os.path.join(__dir__, "tools/__init__.py"), make_importable=True
)
ppocr = importlib.import_module("ppocr", "paddleocr")
from ppocr.utils.logging import get_logger

from ppocr.utils.utility import (
    check_and_read,
    get_image_file_list,
    alpha_to_color,
    binarize_img,
)
from ppocr.utils.network import (
    maybe_download,
    download_with_progressbar,
    is_link,
    confirm_model_dir_url,
)
from tools.infer import predict_system
from tools.infer.utility import str2bool, check_gpu, init_args

logger = get_logger()

__all__ = [
    "PaddleOCR",
    "download_with_progressbar",
]

SUPPORT_DET_MODEL = ["DB"]
SUPPORT_REC_MODEL = ["CRNN", "SVTR_LCNet"]
BASE_DIR = os.path.expanduser("~/.paddleocr/")

DEFAULT_OCR_MODEL_VERSION = "PP-OCRv4"
SUPPORT_OCR_MODEL_VERSION = ["PP-OCR", "PP-OCRv2", "PP-OCRv3", "PP-OCRv4"]
MODEL_URLS = {
    "OCR": {
        "PP-OCRv4": {
            "det": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar",
                },
                "en": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar",
                },
                "ml": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar"
                },
            },
            "rec": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar",
                    "dict_path": "./ppocr/utils/ppocr_keys_v1.txt",
                },
                "en": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar",
                    "dict_path": "./ppocr/utils/en_dict.txt",
                },
                "korean": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/korean_PP-OCRv4_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/korean_dict.txt",
                },
                "japan": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/japan_PP-OCRv4_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/japan_dict.txt",
                },
                "chinese_cht": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/chinese_cht_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/chinese_cht_dict.txt",
                },
                "ta": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/ta_PP-OCRv4_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/ta_dict.txt",
                },
                "te": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/te_PP-OCRv4_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/te_dict.txt",
                },
                "ka": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/ka_PP-OCRv4_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/ka_dict.txt",
                },
                "latin": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/latin_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/latin_dict.txt",
                },
                "arabic": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/arabic_PP-OCRv4_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/arabic_dict.txt",
                },
                "cyrillic": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/cyrillic_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/cyrillic_dict.txt",
                },
                "devanagari": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/devanagari_PP-OCRv4_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/devanagari_dict.txt",
                },
            },
            "cls": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
                }
            },
        },
        "PP-OCRv3": {
            "det": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar",
                },
                "en": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar",
                },
                "ml": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar"
                },
            },
            "rec": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/ppocr_keys_v1.txt",
                },
                "en": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/en_dict.txt",
                },
                "korean": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/korean_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/korean_dict.txt",
                },
                "japan": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/japan_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/japan_dict.txt",
                },
                "chinese_cht": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/chinese_cht_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/chinese_cht_dict.txt",
                },
                "ta": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ta_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/ta_dict.txt",
                },
                "te": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/te_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/te_dict.txt",
                },
                "ka": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ka_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/ka_dict.txt",
                },
                "latin": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/latin_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/latin_dict.txt",
                },
                "arabic": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/arabic_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/arabic_dict.txt",
                },
                "cyrillic": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/cyrillic_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/cyrillic_dict.txt",
                },
                "devanagari": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/devanagari_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/devanagari_dict.txt",
                },
            },
            "cls": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
                }
            },
        },
        "PP-OCRv2": {
            "det": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar",
                },
            },
            "rec": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar",
                    "dict_path": "./ppocr/utils/ppocr_keys_v1.txt",
                }
            },
            "cls": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
                }
            },
        },
        "PP-OCR": {
            "det": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar",
                },
                "en": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_ppocr_mobile_v2.0_det_infer.tar",
                },
                "structure": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_det_infer.tar"
                },
            },
            "rec": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/ppocr_keys_v1.txt",
                },
                "en": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/en_dict.txt",
                },
                "french": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/french_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/french_dict.txt",
                },
                "german": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/german_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/german_dict.txt",
                },
                "korean": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/korean_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/korean_dict.txt",
                },
                "japan": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/japan_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/japan_dict.txt",
                },
                "chinese_cht": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/chinese_cht_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/chinese_cht_dict.txt",
                },
                "ta": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ta_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/ta_dict.txt",
                },
                "te": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/te_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/te_dict.txt",
                },
                "ka": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ka_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/ka_dict.txt",
                },
                "latin": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/latin_ppocr_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/latin_dict.txt",
                },
                "arabic": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/arabic_ppocr_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/arabic_dict.txt",
                },
                "cyrillic": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/cyrillic_ppocr_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/cyrillic_dict.txt",
                },
                "devanagari": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/devanagari_ppocr_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/devanagari_dict.txt",
                },
                "structure": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_rec_infer.tar",
                    "dict_path": "ppocr/utils/dict/table_dict.txt",
                },
            },
            "cls": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
                }
            },
        },
    }
}


def parse_args(mMain=True):
    import argparse

    parser = init_args()
    parser.add_help = mMain
    parser.add_argument("--lang", type=str, default="ch")
    parser.add_argument("--det", type=str2bool, default=True)
    parser.add_argument("--rec", type=str2bool, default=True)
    parser.add_argument("--type", type=str, default="ocr")
    parser.add_argument("--savefile", type=str2bool, default=False)
    parser.add_argument(
        "--ocr_version",
        type=str,
        choices=SUPPORT_OCR_MODEL_VERSION,
        default="PP-OCRv4",
        help="OCR Model version, the current model support list is as follows: "
             "1. PP-OCRv4/v3 Support Chinese and English detection and recognition model, and direction classifier model"
             "2. PP-OCRv2 Support Chinese detection and recognition model. "
             "3. PP-OCR support Chinese detection, recognition and direction classifier and multilingual recognition model.",
    )

    for action in parser._actions:
        if action.dest in [
            "rec_char_dict_path",
            "table_char_dict_path",
            "layout_dict_path",
            "formula_char_dict_path",
        ]:
            action.default = None
    if mMain:
        return parser.parse_args()
    else:
        inference_args_dict = {}
        for action in parser._actions:
            inference_args_dict[action.dest] = action.default
        return argparse.Namespace(**inference_args_dict)


def parse_lang(lang):
    latin_lang = [
        "af",
        "az",
        "bs",
        "cs",
        "cy",
        "da",
        "de",
        "es",
        "et",
        "fr",
        "ga",
        "hr",
        "hu",
        "id",
        "is",
        "it",
        "ku",
        "la",
        "lt",
        "lv",
        "mi",
        "ms",
        "mt",
        "nl",
        "no",
        "oc",
        "pi",
        "pl",
        "pt",
        "ro",
        "rs_latin",
        "sk",
        "sl",
        "sq",
        "sv",
        "sw",
        "tl",
        "tr",
        "uz",
        "vi",
        "french",
        "german",
    ]
    arabic_lang = ["ar", "fa", "ug", "ur"]
    cyrillic_lang = [
        "ru",
        "rs_cyrillic",
        "be",
        "bg",
        "uk",
        "mn",
        "abq",
        "ady",
        "kbd",
        "ava",
        "dar",
        "inh",
        "che",
        "lbe",
        "lez",
        "tab",
    ]
    devanagari_lang = [
        "hi",
        "mr",
        "ne",
        "bh",
        "mai",
        "ang",
        "bho",
        "mah",
        "sck",
        "new",
        "gom",
        "sa",
        "bgc",
    ]
    if lang in latin_lang:
        lang = "latin"
    elif lang in arabic_lang:
        lang = "arabic"
    elif lang in cyrillic_lang:
        lang = "cyrillic"
    elif lang in devanagari_lang:
        lang = "devanagari"
    assert (
            lang in MODEL_URLS["OCR"][DEFAULT_OCR_MODEL_VERSION]["rec"]
    ), "param lang must in {}, but got {}".format(
        MODEL_URLS["OCR"][DEFAULT_OCR_MODEL_VERSION]["rec"].keys(), lang
    )
    if lang == "ch":
        det_lang = "ch"
    elif lang == "structure":
        det_lang = "structure"
    elif lang in ["en", "latin"]:
        det_lang = "en"
    else:
        det_lang = "ml"
    return lang, det_lang


def get_model_config(type, version, model_type, lang):
    if type == "OCR":
        DEFAULT_MODEL_VERSION = DEFAULT_OCR_MODEL_VERSION
    else:
        raise NotImplementedError

    model_urls = MODEL_URLS[type]
    if version not in model_urls:
        version = DEFAULT_MODEL_VERSION
    if model_type not in model_urls[version]:
        if model_type in model_urls[DEFAULT_MODEL_VERSION]:
            version = DEFAULT_MODEL_VERSION
        else:
            logger.error(
                "{} models is not support, we only support {}".format(
                    model_type, model_urls[DEFAULT_MODEL_VERSION].keys()
                )
            )
            sys.exit(-1)

    if lang not in model_urls[version][model_type]:
        if lang in model_urls[DEFAULT_MODEL_VERSION][model_type]:
            version = DEFAULT_MODEL_VERSION
        else:
            logger.error(
                "lang {} is not support, we only support {} for {} models".format(
                    lang,
                    model_urls[DEFAULT_MODEL_VERSION][model_type].keys(),
                    model_type,
                )
            )
            sys.exit(-1)
    return model_urls[version][model_type][lang]


def img_decode(content: bytes):
    np_arr = np.frombuffer(content, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)


def check_img(img, alpha_color=(255, 255, 255)):
    """
    Check the image data. If it is another type of image file, try to decode it into a numpy array.
    The inference network requires three-channel images, So the following channel conversions are done
        single channel image: Gray to RGB R←Y,G←Y,B←Y
        four channel image: alpha_to_color
    args:
        img: image data
            file format: jpg, png and other image formats that opencv can decode, as well as gif and pdf formats
            storage type: binary image, net image file, local image file
        alpha_color: Background color in images in RGBA format
        return: numpy.array (h, w, 3) or list (p, h, w, 3) (p: page of pdf), boolean, boolean
    """
    flag_gif, flag_pdf = False, False
    if isinstance(img, bytes):
        img = img_decode(img)
    if isinstance(img, str):
        # download net image
        if is_link(img):
            download_with_progressbar(img, "tmp.jpg")
            img = "tmp.jpg"
        image_file = img
        img, flag_gif, flag_pdf = check_and_read(image_file)
        if not flag_gif and not flag_pdf:
            with open(image_file, "rb") as f:
                img_str = f.read()
                img = img_decode(img_str)
            if img is None:
                try:
                    buf = BytesIO()
                    image = BytesIO(img_str)
                    im = Image.open(image)
                    rgb = im.convert("RGB")
                    rgb.save(buf, "jpeg")
                    buf.seek(0)
                    image_bytes = buf.read()
                    data_base64 = str(base64.b64encode(image_bytes), encoding="utf-8")
                    image_decode = base64.b64decode(data_base64)
                    img_array = np.frombuffer(image_decode, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                except:
                    logger.error("error in loading image:{}".format(image_file))
                    return None, flag_gif, flag_pdf
        if img is None:
            logger.error("error in loading image:{}".format(image_file))
            return None, flag_gif, flag_pdf
    # single channel image array.shape:h,w
    if isinstance(img, np.ndarray) and len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # four channel image array.shape:h,w,c
    if isinstance(img, np.ndarray) and len(img.shape) == 3 and img.shape[2] == 4:
        img = alpha_to_color(img, alpha_color)
    return img, flag_gif, flag_pdf


class PaddleOCR(predict_system.TextSystem):
    def __init__(self, **kwargs):
        """
        paddleocr package
        args:
            **kwargs: other params show in paddleocr --help
        """
        params = parse_args(mMain=False)
        params.__dict__.update(**kwargs)
        assert (
                params.ocr_version in SUPPORT_OCR_MODEL_VERSION
        ), "ocr_version must in {}, but get {}".format(
            SUPPORT_OCR_MODEL_VERSION, params.ocr_version
        )
        params.use_gpu = check_gpu(params.use_gpu)

        if not params.show_log:
            logger.setLevel(logging.INFO)
        self.use_angle_cls = params.use_angle_cls
        lang, det_lang = parse_lang(params.lang)

        # init model dir
        det_model_config = get_model_config("OCR", params.ocr_version, "det", det_lang)
        params.det_model_dir, det_url = confirm_model_dir_url(
            params.det_model_dir,
            os.path.join(BASE_DIR, "whl", "det", det_lang),
            det_model_config["url"],
        )
        rec_model_config = get_model_config("OCR", params.ocr_version, "rec", lang)
        params.rec_model_dir, rec_url = confirm_model_dir_url(
            params.rec_model_dir,
            os.path.join(BASE_DIR, "whl", "rec", lang),
            rec_model_config["url"],
        )
        cls_model_config = get_model_config("OCR", params.ocr_version, "cls", "ch")
        params.cls_model_dir, cls_url = confirm_model_dir_url(
            params.cls_model_dir,
            os.path.join(BASE_DIR, "whl", "cls"),
            cls_model_config["url"],
        )
        if params.ocr_version in ["PP-OCRv3", "PP-OCRv4"]:
            params.rec_image_shape = "3, 48, 320"
        else:
            params.rec_image_shape = "3, 32, 320"
        # download model if using paddle infer
        if not params.use_onnx:
            maybe_download(params.det_model_dir, det_url)
            maybe_download(params.rec_model_dir, rec_url)
            maybe_download(params.cls_model_dir, cls_url)

        if params.det_algorithm not in SUPPORT_DET_MODEL:
            logger.error("det_algorithm must in {}".format(SUPPORT_DET_MODEL))
            sys.exit(0)
        if params.rec_algorithm not in SUPPORT_REC_MODEL:
            logger.error("rec_algorithm must in {}".format(SUPPORT_REC_MODEL))
            sys.exit(0)

        if params.rec_char_dict_path is None:
            params.rec_char_dict_path = str(
                Path(__file__).parent / rec_model_config["dict_path"]
            )

        logger.debug(params)
        # init det_model and rec_model
        super().__init__(params)
        self.page_num = params.page_num

    def ocr(
            self,
            img,
            det=True,
            rec=True,
            cls=True,
            bin=False,
            inv=False,
            alpha_color=(255, 255, 255),
            slice={},
    ):
        """
        OCR with PaddleOCR

        Args:
            img: Image for OCR. It can be an ndarray, img_path, or a list of ndarrays.
            det: Use text detection or not. If False, only text recognition will be executed. Default is True.
            rec: Use text recognition or not. If False, only text detection will be executed. Default is True.
            cls: Use angle classifier or not. Default is True. If True, the text with a rotation of 180 degrees can be recognized. If no text is rotated by 180 degrees, use cls=False to get better performance.
            bin: Binarize image to black and white. Default is False.
            inv: Invert image colors. Default is False.
            alpha_color: Set RGB color Tuple for transparent parts replacement. Default is pure white.
            slice: Use sliding window inference for large images. Both det and rec must be True. Requires int values for slice["horizontal_stride"], slice["vertical_stride"], slice["merge_x_thres"], slice["merge_y_thres"] (See doc/doc_en/slice_en.md). Default is {}.

        Returns:
            If both det and rec are True, returns a list of OCR results for each image. Each OCR result is a list of bounding boxes and recognized text for each detected text region.
            If det is True and rec is False, returns a list of detected bounding boxes for each image.
            If det is False and rec is True, returns a list of recognized text for each image.
            If both det and rec are False, returns a list of angle classification results for each image.

        Raises:
            AssertionError: If the input image is not of type ndarray, list, str, or bytes.
            SystemExit: If det is True and the input is a list of images.

        Note:
            - If the angle classifier is not initialized (use_angle_cls=False), it will not be used during the forward process.
            - For PDF files, if the input is a list of images and the page_num is specified, only the first page_num images will be processed.
            - The preprocess_image function is used to preprocess the input image by applying alpha color replacement, inversion, and binarization if specified.
        """
        assert isinstance(img, (np.ndarray, list, str, bytes))
        if isinstance(img, list) and det == True:
            logger.error("When input a list of images, det must be false")
            exit(0)
        if cls == True and self.use_angle_cls == False:
            logger.warning(
                "Since the angle classifier is not initialized, it will not be used during the forward process"
            )

        img, flag_gif, flag_pdf = check_img(img, alpha_color)
        # for infer pdf file
        if isinstance(img, list) and flag_pdf:
            if self.page_num > len(img) or self.page_num == 0:
                imgs = img
            else:
                imgs = img[: self.page_num]
        else:
            imgs = [img]

        def preprocess_image(_image):
            _image = alpha_to_color(_image, alpha_color)
            if inv:
                _image = cv2.bitwise_not(_image)
            if bin:
                _image = binarize_img(_image)
            return _image

        if det and rec:
            ocr_res = []
            for img in imgs:
                img = preprocess_image(img)
                dt_boxes, rec_res, _ = self.__call__(img, cls, slice)
                if not dt_boxes and not rec_res:
                    ocr_res.append(None)
                    continue
                tmp_res = [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
                ocr_res.append(tmp_res)
            return ocr_res
        elif det and not rec:
            ocr_res = []
            for img in imgs:
                img = preprocess_image(img)
                dt_boxes, elapse = self.text_detector(img)
                if dt_boxes.size == 0:
                    ocr_res.append(None)
                    continue
                tmp_res = [box.tolist() for box in dt_boxes]
                ocr_res.append(tmp_res)
            return ocr_res
        else:
            ocr_res = []
            cls_res = []
            for img in imgs:
                if not isinstance(img, list):
                    img = preprocess_image(img)
                    img = [img]
                if self.use_angle_cls and cls:
                    img, cls_res_tmp, elapse = self.text_classifier(img)
                    if not rec:
                        cls_res.append(cls_res_tmp)
                rec_res, elapse = self.text_recognizer(img)
                ocr_res.append(rec_res)
            if not rec:
                return cls_res
            return ocr_res


def main(image_dir):
    """
    Main function for running PaddleOCR or PPStructure.

    This function takes command line arguments, processes the images, and performs OCR or structure analysis based on the specified type.
    """
    # for cmd
    args = parse_args(mMain=True)
    logger.info("for usage help, please use `paddleocr --help`")
    args.image_dir = image_dir
    if is_link(image_dir):
        os.remove("tmp.jpg") if os.path.exists("tmp.jpg") else None
        download_with_progressbar(image_dir, "tmp.jpg")
        image_file_list = ["tmp.jpg"]
    else:
        image_file_list = get_image_file_list(args.image_dir)
    if len(image_file_list) == 0:
        logger.error("no images find in {}".format(args.image_dir))
        return
    if args.type == "ocr":
        engine = PaddleOCR(**args.__dict__)
    else:
        raise NotImplementedError

    for img_path in image_file_list:
        img_name = os.path.basename(img_path).split(".")[0]
        logger.info("{}{}{}".format("*" * 10, img_path, "*" * 10))
        if args.type == "ocr":
            result = engine.ocr(
                img_path,
                det=args.det,
                rec=args.rec,
                cls=args.use_angle_cls,
                bin=args.binarize,
                inv=args.invert,
                alpha_color=args.alphacolor,
            )
            if result is not None:
                lines = []
                for res in result:
                    if res:
                        for line in res:
                            logger.info(line)
                            lines.append(pprint.pformat(line) + "\n")
                    else:
                        logger.info(result)
                if args.savefile:
                    if os.path.exists(args.output) is False:
                        os.mkdir(args.output)
                    outfile = args.output + "/" + img_name + ".txt"
                    with open(outfile, "w", encoding="utf-8") as f:
                        f.writelines(lines)


if __name__ == '__main__':
    main("doc/imgs")
