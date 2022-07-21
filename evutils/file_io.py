# -*- coding: utf-8 -*-
import errno
import os
import tqdm
from six.moves import urllib

from mmcv.utils import get_logger
logger = get_logger('evutils')


def mkdir_p(dirname):
    """ Like "mkdir -p", make a dir recursively, but do nothing if the dir exists

    Args:
        dirname(str):
    """
    assert dirname is not None
    if dirname == '' or os.path.isdir(dirname):
        return
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e


def download(url, dir, filename=None, expect_size=None):
    """
    Download URL to a directory.
    Will figure out the filename automatically from URL, if not given.
    """
    mkdir_p(dir)
    if filename is None:
        filename = url.split('/')[-1]
    fpath = os.path.join(dir, filename)

    if os.path.isfile(fpath):
        if expect_size is not None and os.stat(fpath).st_size == expect_size:
            logger.info("File {} exists! Skip download.".format(filename))
            return fpath
        else:
            logger.warn("File {} exists. Will overwrite with a new download!".format(filename))

    def hook(t):
        last_b = [0]

        def inner(b, bsize, tsize=None):
            if tsize is not None:
                t.total = tsize
            t.update((b - last_b[0]) * bsize)
            last_b[0] = b
        return inner
    try:
        with tqdm.tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            fpath, _ = urllib.request.urlretrieve(url, fpath, reporthook=hook(t))
        statinfo = os.stat(fpath)
        size = statinfo.st_size
    except IOError:
        logger.error("Failed to download {}".format(url))
        raise
    assert size > 0, "Downloaded an empty file from {}!".format(url)

    if expect_size is not None and size != expect_size:
        logger.error("File downloaded from {} does not match the expected size!".format(url))
        logger.error("You may have downloaded a broken file, or the upstream may have modified the file.")

    # TODO human-readable size
    logger.info('Succesfully downloaded ' + filename + ". " + str(size) + ' bytes.')
    return fpath


def recursive_walk(rootdir):
    """
    Yields:
        str: All files in rootdir, recursively.
    """
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)


def read_dir(root):
    """
    ref: https://github.com/whai362/pan_pp.pytorch/blob/master/eval/ctw/file_util.py#L3
    """
	file_path_list = []
	for file_path, dirs, files in os.walk(root):
		for file in files:
			file_path_list.append(os.path.join(file_path, file).replace('\\', '/'))
	file_path_list.sort()
	return file_path_list


def get_image_file_list(img_file):
    """
    ref: https://github.com/PaddlePaddle/PaddleOCR/blob/bb77fcef6fdfc029f92d7876b714554389053699/ppocr/utils/utility.py#L57
    """
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif'}
    if os.path.isfile(img_file) and _check_image_file(img_file):
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if os.path.isfile(file_path) and _check_image_file(file_path):
                imgs_lists.append(file_path)
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    return imgs_lists
