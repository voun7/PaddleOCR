import copy

__all__ = ["build_post_process"]

from .cls_postprocess import ClsPostProcess
from .db_postprocess import DBPostProcess, DistillationDBPostProcess
from .rec_postprocess import CTCLabelDecode, DistillationCTCLabelDecode


def build_post_process(config, global_config=None):
    support_dict = [
        "DBPostProcess",
        "DistillationDBPostProcess",
        "CTCLabelDecode",
        "DistillationCTCLabelDecode",
        "ClsPostProcess",
    ]

    config = copy.deepcopy(config)
    module_name = config.pop("name")
    if module_name == "None":
        return
    if global_config is not None:
        config.update(global_config)
    assert module_name in support_dict, Exception(f"post process only support {support_dict}")
    module_class = eval(module_name)(**config)
    return module_class
