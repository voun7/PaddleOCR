import argparse
import os
import sys

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ModuleNotFoundError:
    ort = None  # optional if paddle is being used

try:
    import paddle
    from paddle import inference
except ModuleNotFoundError:
    paddle = None  # optional if onnx is being used

from ppocr.utils.logging import get_logger


def str2bool(v):
    return v.lower() in ("true", "yes", "t", "y", "1")


def str2int_tuple(v):
    return tuple([int(i.strip()) for i in v.split(",")])


def init_args():
    parser = argparse.ArgumentParser()
    # params for prediction engine
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--gpu_mem", type=int, default=500)
    parser.add_argument("--gpu_id", type=int, default=0)

    # params for text detector
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--det_algorithm", type=str, default="DB")
    parser.add_argument("--det_model_dir", type=str)
    parser.add_argument("--det_limit_side_len", type=float, default=960)
    parser.add_argument("--det_limit_type", type=str, default="max")
    parser.add_argument("--det_box_type", type=str, default="quad")

    # DB parmas
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.6)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.5)
    parser.add_argument("--max_batch_size", type=int, default=10)
    parser.add_argument("--use_dilation", type=str2bool, default=False)
    parser.add_argument("--det_db_score_mode", type=str, default="fast")

    # params for text recognizer
    parser.add_argument("--rec_algorithm", type=str, default="SVTR_LCNet")
    parser.add_argument("--rec_model_dir", type=str)
    parser.add_argument("--rec_image_inverse", type=str2bool, default=True)
    parser.add_argument("--rec_image_shape", type=str, default="3, 48, 320")
    parser.add_argument("--rec_batch_num", type=int, default=6)
    parser.add_argument("--max_text_length", type=int, default=25)
    parser.add_argument(
        "--rec_char_dict_path", type=str, default="./ppocr/utils/ppocr_keys_v1.txt"
    )
    parser.add_argument("--use_space_char", type=str2bool, default=True)
    parser.add_argument("--drop_score", type=float, default=0.5)

    # params for text classifier
    parser.add_argument("--use_angle_cls", type=str2bool, default=False)
    parser.add_argument("--cls_model_dir", type=str)
    parser.add_argument("--cls_image_shape", type=str, default="3, 48, 192")
    parser.add_argument("--label_list", type=list, default=["0", "180"])
    parser.add_argument("--cls_batch_num", type=int, default=6)
    parser.add_argument("--cls_thresh", type=float, default=0.9)

    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--cpu_threads", type=int, default=10)
    parser.add_argument("--use_pdserving", type=str2bool, default=False)
    parser.add_argument("--warmup", type=str2bool, default=False)

    # multi-process
    parser.add_argument("--use_mp", type=str2bool, default=False)
    parser.add_argument("--total_process_num", type=int, default=1)
    parser.add_argument("--process_id", type=int, default=0)

    parser.add_argument("--show_log", type=str2bool, default=True)
    parser.add_argument("--use_onnx", type=str2bool, default=False)
    parser.add_argument("--onnx_providers", nargs="+", type=str, default=False)
    parser.add_argument("--onnx_sess_options", type=list, default=False)

    # extended function
    parser.add_argument(
        "--return_word_box",
        type=str2bool,
        default=False,
        help="Whether return the bbox of each word (split by space) or chinese character. Only used in ppstructure for layout recovery",
    )

    parser.add_argument(
        "--invert",
        type=str2bool,
        default=False,
        help="Whether to invert image before processing",
    )
    parser.add_argument(
        "--binarize",
        type=str2bool,
        default=False,
        help="Whether to threshold binarize image before processing",
    )
    parser.add_argument(
        "--alphacolor",
        type=str2int_tuple,
        default=(255, 255, 255),
        help="Replacement color for the alpha channel, if the latter is present; R,G,B integers",
    )
    return parser


def parse_args():
    parser = init_args()
    return parser.parse_args()


def create_predictor(args, mode, logger):
    if mode == "det":
        model_dir = args.det_model_dir
    elif mode == "cls":
        model_dir = args.cls_model_dir
    elif mode == "rec":
        model_dir = args.rec_model_dir
    else:
        model_dir = args.e2e_model_dir

    if model_dir is None:
        logger.info("not find {} model file path {}".format(mode, model_dir))
        sys.exit(0)
    if args.use_onnx:
        if ort is None:
            raise ModuleNotFoundError("Onnxruntime module is not installed!")

        model_file_path = model_dir + "/model.onnx"
        if not os.path.exists(model_file_path):
            raise ValueError(f"not find model file path {model_file_path}")

        sess_options = args.onnx_sess_options or None

        if args.onnx_providers and len(args.onnx_providers) > 0:
            sess = ort.InferenceSession(
                model_file_path,
                providers=args.onnx_providers,
                sess_options=sess_options,
            )
        elif args.use_gpu:
            sess = ort.InferenceSession(
                model_file_path,
                providers=[
                    (
                        "CUDAExecutionProvider",
                        {"device_id": args.gpu_id, "cudnn_conv_algo_search": "DEFAULT"},
                    )
                ],
                sess_options=sess_options,
            )
        else:
            sess = ort.InferenceSession(
                model_file_path,
                providers=["CPUExecutionProvider"],
                sess_options=sess_options,
            )
        return sess, sess.get_inputs()[0], None, None

    else:
        file_names = ["model", "inference"]
        for file_name in file_names:
            model_file_path = "{}/{}.pdmodel".format(model_dir, file_name)
            params_file_path = "{}/{}.pdiparams".format(model_dir, file_name)
            if os.path.exists(model_file_path) and os.path.exists(params_file_path):
                break
        if not os.path.exists(model_file_path):
            raise ValueError(f"not find model.pdmodel or inference.pdmodel in {model_dir}")
        if not os.path.exists(params_file_path):
            raise ValueError(f"not find model.pdiparams or inference.pdiparams in {model_dir}")

        if paddle is None:
            raise ModuleNotFoundError("Paddle or Setuptools module is not installed!")

        config = inference.Config(model_file_path, params_file_path)

        if hasattr(args, "precision"):
            if args.precision == "fp16" and args.use_tensorrt:
                precision = inference.PrecisionType.Half
            elif args.precision == "int8":
                precision = inference.PrecisionType.Int8
            else:
                precision = inference.PrecisionType.Float32
        else:
            precision = inference.PrecisionType.Float32

        if args.use_gpu:
            gpu_id = get_infer_gpuid()
            if gpu_id is None:
                logger.warning(
                    "GPU is not found in current device by nvidia-smi. Please check your device or ignore it if run on jetson."
                )
            config.enable_use_gpu(args.gpu_mem, args.gpu_id)
            if args.use_tensorrt:
                config.enable_tensorrt_engine(
                    workspace_size=1 << 30,
                    precision_mode=precision,
                    max_batch_size=args.max_batch_size,
                    min_subgraph_size=args.min_subgraph_size,  # skip the minmum trt subgraph
                    use_calib_mode=False,
                )

                # collect shape
                trt_shape_f = os.path.join(model_dir, f"{mode}_trt_dynamic_shape.txt")

                if not os.path.exists(trt_shape_f):
                    config.collect_shape_range_info(trt_shape_f)
                    logger.info(f"collect dynamic shape info into : {trt_shape_f}")
                try:
                    config.enable_tuned_tensorrt_dynamic_shape(trt_shape_f, True)
                except Exception as E:
                    logger.info(E)
                    logger.info("Please keep your paddlepaddle-gpu >= 2.3.0!")
        else:
            config.disable_gpu()
            if args.enable_mkldnn:
                # cache 10 different shapes for mkldnn to avoid memory leak
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
                if args.precision == "fp16":
                    config.enable_mkldnn_bfloat16()
                if hasattr(args, "cpu_threads"):
                    config.set_cpu_math_library_num_threads(args.cpu_threads)
                else:
                    # default cpu threads as 10
                    config.set_cpu_math_library_num_threads(10)
        # enable memory optim
        config.enable_memory_optim()
        config.disable_glog_info()
        config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
        config.delete_pass("matmul_transpose_reshape_fuse_pass")
        if mode == "rec" and args.rec_algorithm == "SRN":
            config.delete_pass("gpu_cpu_map_matmul_v2_to_matmul_pass")
        if mode == "re":
            config.delete_pass("simplify_with_basic_ops_pass")
        if mode == "table":
            config.delete_pass("fc_fuse_pass")  # not supported for table
        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)

        # create predictor
        predictor = inference.create_predictor(config)
        input_names = predictor.get_input_names()
        if mode in ["ser", "re"]:
            input_tensor = []
            for name in input_names:
                input_tensor.append(predictor.get_input_handle(name))
        else:
            for name in input_names:
                input_tensor = predictor.get_input_handle(name)
        output_tensors = get_output_tensors(args, mode, predictor)
        return predictor, input_tensor, output_tensors, config


def get_output_tensors(args, mode, predictor):
    output_names = predictor.get_output_names()
    output_tensors = []
    if mode == "rec" and args.rec_algorithm in ["CRNN", "SVTR_LCNet", "SVTR_HGNet"]:
        output_name = "softmax_0.tmp_0"
        if output_name in output_names:
            return [predictor.get_output_handle(output_name)]
        else:
            for output_name in output_names:
                output_tensor = predictor.get_output_handle(output_name)
                output_tensors.append(output_tensor)
    else:
        for output_name in output_names:
            output_tensor = predictor.get_output_handle(output_name)
            output_tensors.append(output_tensor)
    return output_tensors


def get_infer_gpuid():
    """
    Get the GPU ID to be used for inference.

    Returns:
        int: The GPU ID to be used for inference.
    """
    logger = get_logger()
    if not paddle.device.is_compiled_with_rocm:
        gpu_id_str = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    else:
        gpu_id_str = os.environ.get("HIP_VISIBLE_DEVICES", "0")

    gpu_ids = gpu_id_str.split(",")
    logger.warning(
        "The first GPU is used for inference by default, GPU ID: {}".format(gpu_ids[0])
    )
    return int(gpu_ids[0])


def get_rotate_crop_image(img, points):
    """
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    """
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])
        )
    )
    pts_std = np.float32(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    )
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def get_minarea_rect_crop(img, points):
    bounding_box = cv2.minAreaRect(np.array(points).astype(np.int32))
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_a, index_b, index_c, index_d = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_a = 0
        index_d = 1
    else:
        index_a = 1
        index_d = 0
    if points[3][1] > points[2][1]:
        index_b = 2
        index_c = 3
    else:
        index_b = 3
        index_c = 2

    box = [points[index_a], points[index_b], points[index_c], points[index_d]]
    crop_img = get_rotate_crop_image(img, np.array(box))
    return crop_img


def slice_generator(image, horizontal_stride, vertical_stride, maximum_slices=500):
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    image_h, image_w = image.shape[:2]
    vertical_num_slices = (image_h + vertical_stride - 1) // vertical_stride
    horizontal_num_slices = (image_w + horizontal_stride - 1) // horizontal_stride

    assert (
            vertical_num_slices > 0
    ), f"Invalid number ({vertical_num_slices}) of vertical slices"

    assert (
            horizontal_num_slices > 0
    ), f"Invalid number ({horizontal_num_slices}) of horizontal slices"

    if vertical_num_slices >= maximum_slices:
        recommended_vertical_stride = max(1, image_h // maximum_slices) + 1
        assert (
            False
        ), f"Too computationally expensive with {vertical_num_slices} slices, try a higher vertical stride (recommended minimum: {recommended_vertical_stride})"

    if horizontal_num_slices >= maximum_slices:
        recommended_horizontal_stride = max(1, image_w // maximum_slices) + 1
        assert (
            False
        ), f"Too computationally expensive with {horizontal_num_slices} slices, try a higher horizontal stride (recommended minimum: {recommended_horizontal_stride})"

    for v_slice_idx in range(vertical_num_slices):
        v_start = max(0, (v_slice_idx * vertical_stride))
        v_end = min(((v_slice_idx + 1) * vertical_stride), image_h)
        vertical_slice = image[v_start:v_end, :]
        for h_slice_idx in range(horizontal_num_slices):
            h_start = max(0, (h_slice_idx * horizontal_stride))
            h_end = min(((h_slice_idx + 1) * horizontal_stride), image_w)
            horizontal_slice = vertical_slice[:, h_start:h_end]

            yield (horizontal_slice, v_start, h_start)


def calculate_box_extents(box):
    min_x = box[0][0]
    max_x = box[1][0]
    min_y = box[0][1]
    max_y = box[2][1]
    return min_x, max_x, min_y, max_y


def merge_boxes(box1, box2, x_threshold, y_threshold):
    min_x1, max_x1, min_y1, max_y1 = calculate_box_extents(box1)
    min_x2, max_x2, min_y2, max_y2 = calculate_box_extents(box2)

    if (
            abs(min_y1 - min_y2) <= y_threshold
            and abs(max_y1 - max_y2) <= y_threshold
            and abs(max_x1 - min_x2) <= x_threshold
    ):
        new_xmin = min(min_x1, min_x2)
        new_xmax = max(max_x1, max_x2)
        new_ymin = min(min_y1, min_y2)
        new_ymax = max(max_y1, max_y2)
        return [
            [new_xmin, new_ymin],
            [new_xmax, new_ymin],
            [new_xmax, new_ymax],
            [new_xmin, new_ymax],
        ]
    else:
        return None


def merge_fragmented(boxes, x_threshold=10, y_threshold=10):
    merged_boxes = []
    visited = set()

    for i, box1 in enumerate(boxes):
        if i in visited:
            continue

        merged_box = [point[:] for point in box1]

        for j, box2 in enumerate(boxes[i + 1:], start=i + 1):
            if j not in visited:
                merged_result = merge_boxes(
                    merged_box, box2, x_threshold=x_threshold, y_threshold=y_threshold
                )
                if merged_result:
                    merged_box = merged_result
                    visited.add(j)

        merged_boxes.append(merged_box)

    if len(merged_boxes) == len(boxes):
        return np.array(merged_boxes)
    else:
        return merge_fragmented(merged_boxes, x_threshold, y_threshold)


def check_gpu(use_gpu):
    return use_gpu and ((paddle and paddle.is_compiled_with_cuda()) or (ort and ort.get_device() == "GPU"))


if __name__ == "__main__":
    pass
