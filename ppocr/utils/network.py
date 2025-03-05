import os
import os.path as osp
import shutil
import tarfile
import time
from pathlib import Path

from ppocr.utils.logging import get_logger

DOWNLOAD_RETRY_LIMIT = 3


def download_with_progressbar(url, save_path):
    logger = get_logger()
    if save_path and os.path.exists(save_path):
        logger.info(f"Path {save_path} already exists. Skipping...")
        return
    else:
        _download(url, save_path)


def _download(url, save_path):
    """
    Download from url, save to path.

    url (str): download url
    save_path (str): download to given path
    """
    import requests

    logger = get_logger()

    fname = osp.split(url)[-1]
    retry_cnt = 0

    while not osp.exists(save_path):
        if retry_cnt < DOWNLOAD_RETRY_LIMIT:
            retry_cnt += 1
        else:
            raise RuntimeError(f"Download from {url} failed. " "Retry limit reached")

        try:
            req = requests.get(url, stream=True)
        except Exception as e:  # requests.exceptions.ConnectionError
            logger.info(
                "Downloading {} from {} failed {} times with exception {}".format(fname, url, retry_cnt + 1, str(e)))
            time.sleep(1)
            continue

        if req.status_code != 200:
            raise RuntimeError(f"Downloading from {url} failed with code {req.status_code}!")

        # For protecting download interrupted, download to tmp_file firstly, move tmp_file to save_path
        # after download finished
        tmp_file = save_path + ".tmp"
        total_size = req.headers.get("content-length")
        with open(tmp_file, "wb") as f:
            if total_size:
                total, downloaded_size = int(total_size), 0
                for chunk in req.iter_content(chunk_size=1024):
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    print_progress(downloaded_size, total, "Downloading")
            else:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        shutil.move(tmp_file, save_path)

    return save_path


def print_progress(iteration: int, total: int, prefix: str = '', suffix: str = 'Complete', decimals: int = 3,
                   bar_length: int = 25) -> None:
    if not total:  # prevent error if total is zero.
        return
    format_str = "{0:." + str(decimals) + "f}"  # format the % done number string
    percents = format_str.format(100 * (iteration / float(total)))  # calculate the % done
    filled_length = int(round(bar_length * iteration / float(total)))  # calculate the filled bar length
    bar = '#' * filled_length + '-' * (bar_length - filled_length)  # generate the bar string
    print(f"\r{prefix} |{bar}| {percents}% {suffix}", end='', flush=True)  # prints progress on the same line
    if "100.0" in percents:  # prevent next line from joining previous line
        print()


def maybe_download(model_storage_directory, url, use_onnx):
    if Path(f"{model_storage_directory}/model.onnx").exists() and use_onnx:
        return
    # using custom model
    tar_file_name_list = [".pdiparams", ".pdiparams.info", ".pdmodel"]
    if not os.path.exists(
            os.path.join(model_storage_directory, "inference.pdiparams")
    ) or not os.path.exists(os.path.join(model_storage_directory, "inference.pdmodel")):
        assert url.endswith(".tar"), "Only supports tar compressed package"
        tmp_path = os.path.join(model_storage_directory, url.split("/")[-1])
        print("download {} to {}".format(url, tmp_path))
        os.makedirs(model_storage_directory, exist_ok=True)
        download_with_progressbar(url, tmp_path)
        with tarfile.open(tmp_path, "r") as tarObj:
            for member in tarObj.getmembers():
                filename = None
                for tar_file_name in tar_file_name_list:
                    if member.name.endswith(tar_file_name):
                        filename = "inference" + tar_file_name
                if filename is None:
                    continue
                file = tarObj.extractfile(member)
                with open(os.path.join(model_storage_directory, filename), "wb") as f:
                    f.write(file.read())
        os.remove(tmp_path)


def is_link(s):
    return s is not None and s.startswith("http")


def confirm_model_dir_url(model_dir, default_model_dir, default_url):
    url = default_url
    if model_dir is None or is_link(model_dir):
        if is_link(model_dir):
            url = model_dir
        file_name = url.split("/")[-1][:-4]
        model_dir = default_model_dir
        model_dir = os.path.join(model_dir, file_name)
    return model_dir, url
