# -*- coding: utf-8 -*-
from typing import List

import numpy as np
import mygenerator.util_image as util_image
from mygenerator.utils import log, log2, logw
import glob
import cv2

################################################################################
################################################################################
def image_padding_validate(final_image, min_padding, max_padding):
    """
    Args:
        final_image:
        min_padding:
        max_padding:

    Returns:

    """
    char_list, pad_list = image_padding_get(final_image, threshold=0)
    if (
        sum([t >= min_padding and t <= max_padding for t in pad_list]) != len(pad_list)
        or len(pad_list) == 0
    ):
        logw(f"Wrong Padding:  {final_image.shape}, {min_padding}, {max_padding}, {pad_list}")
        #logw("\n")
        return False
        # raise Exception('error wrong')
    return True


def image_padding_load(img_path, threshold=15) -> List[int]:
    """
    Args:
        img_path:
        threshold:

    Returns number of consecutive blank columns in the image.
    Example return value: [4, 8, 3] means that image contains
    3 blank columns. The size of each corresponding column in pixels
    is 4, 8 and 3.
    """
    img = util_image.image_read(str(img_path))
    char_list, pad_list = image_padding_get(img, threshold)
    return char_list, pad_list


def image_padding_get(img, threshold=0, inverse=True):
    """
    Args:
         img_path:
         threshold:
      Returns number of consecutive blank columns in the image.
      Example return value: [4, 8, 3] means that image contains
      3 blank columns. The size of each corresponding column in pixels
      is 4, 8 and 3.
    """
    if inverse:
        img = cv2.bitwise_not(img)
    height, width = img.shape[:2]
    xpad_list = []
    xchar_list = []
    xpad = 0
    xchar = 0
    smax = 1
    nchar_min = 4

    # print(img.shape)
    for xi in range(0, width):
        s = np.sum(img[:, xi])
        # print(s)

        if s >= smax:
            xchar += 1
            if xpad != 0:
                xpad_list.append(xpad)
                xpad = 0

        elif xpad >= 1:
            xpad += 1

        else:
            ### Minimal char width OR next nchar_min are blank
            if xchar >= nchar_min or np.sum(img[:, xi : (xi + nchar_min)]) < 1:
                xpad += 1
                if xchar != 0:
                    xchar_list.append(xchar)
                    xchar = 0
            else:
                xchar += 1

    if xpad != 0:
        xpad_list.append(xpad)

    if xchar != 0:
        xchar_list.append(xchar)

    return xchar_list, xpad_list
    # return  xpad_list


def run_image_padding_validate(
    min_spacing: int = 1,
    max_spacing: int = 1,
    image_width: int = 5,
    input_path: str = "",
    inverse_image: bool = True,
    config_file: str = "default",
    **kwargs,
):
    """
    Args:
        min_spacing:
        max_spacing:
        image_width:
        input_path:
        config_file:
        **kwargs:
    Returns: None

    """
    flist = sorted(glob.glob(input_path + "/*.png"))
    log("N files: ", len(flist))
    nerr = 0
    for fi in flist:
        img = util_image.image_read(str(fi))
        char_list, pad_list = image_padding_get(img, threshold=0, inverse=inverse_image)

        if img.shape[1] != image_width:
            logw("error witdth", fi, image_width, img.shape)
            nerr += 1

        if (
            sum([t >= min_spacing and t <= max_spacing for t in pad_list]) != len(pad_list)
            or len(pad_list) == 0
        ):
            logw("error padding", fi, min_spacing, max_spacing, pad_list)
            nerr += 1

    log("N errors", nerr)
