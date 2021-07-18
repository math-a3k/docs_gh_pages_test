# -*- coding: utf-8 -*-
import pathlib
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

import mygenerator.dataset as dataset
import mygenerator.util_image as util_image
from mygenerator.utils import logw
from mygenerator.validate import image_padding_validate


class ImageTransform:
    """
    Maps textual features of the input dataset to images of the text characters.
    Expects features of the input dataset to be strings.
    """

    def __init__(self):
        """
        Parameters
        ----------
        """

    def transform(self, ds: dataset.ImageDataset) -> dataset.ImageDataset:
        """
        Transform Dataset into Dataset
        Args:
            ds: dataset.ImageDataset
        Returns:dataset.ImageDataset
        """
        return ds

    def fit(self, ds: dataset.ImageDataset):
        """
        fit the transformation
        Args:
            ds: dataset.ImageDataset
        Returns: Object
        """
        return self

    def fit_transform(self, ds: dataset.ImageDataset) -> dataset.ImageDataset:
        """
        Updates internal parameters of this transformation and then
        applies it to the provided dataset.

        Args:
            ds: dataset.ImageDataset
        Returns:dataset.ImageDataset
        """
        return self.fit(ds).transform(ds)


class CharToImages:
    """
    Maps textual features of the input dataset to images of the text characters.
    Expects features of the input dataset to be strings.
    """

    def __init__(self, font: dataset.ImageDataset):
        """
        Parameters
        ----------
        font: dataset which contains images of characters. Images are features
        of the font dataset and characters are it's labels. Each character
        can have more than one image.
        """
        self.font_dataset = font

    def transform(self, ds: dataset.NlpDataset) -> dataset.ImageDataset:
        """
        Replaces features of the input dataset by mapping feature (string type)
        to a list of character images (List[np.array] type)
        """

        def _get_image_fn(idx):
            digits = ds.get_text_only(idx)
            img_list: List[np.ndarray] = []
            for dig in digits:
                dig_loc = self.font_dataset.get_label_list(dig)
                if len(dig_loc) == 0:
                    continue

                #### Random select one image
                idx = np.random.choice(dig_loc)

                #### Get the image
                char_img = self.font_dataset.get_image_only(idx)
                img_list.append(char_img)
            return img_list

        return dataset.ImageDataset(get_image_fn=_get_image_fn, meta=ds.meta)

    def fit(self, ds: dataset.NlpDataset):
        """
        Args:
            ds: dataset.NlpDataset
        Returns:dataset.ImageDataset
        """
        return self

    def fit_transform(self, ds: dataset.NlpDataset) -> dataset.ImageDataset:
        """
        Updates internal parameters of this transformation and then
        applies it to the provided dataset.
        """
        return self.fit(ds).transform(ds)


class RemoveWhitePadding(ImageTransform):
    """
    Removes surrounding white spaces in images of the input dataset.
    Images of the output dataset are cropped.
    """

    def transform(self, ds: dataset.ImageDataset) -> dataset.ImageDataset:
        # TBD: How to handle totally "transparent" images?

        def _get_image_fn(idx):
            img = ds.get_image_only(idx)
            img = self.transform_sample(img)

            return img

        return dataset.ImageDataset(get_image_fn=_get_image_fn, meta=ds.meta)

    def transform_sample(self, image: np.ndarray):
        """
        Remove surrounding white spaces in digit image

        Parameters
        ----------
        image: image of the digit

        returns
        -------
        crop: cropped image
        """
        # from mygenerator.util_image import image_remove_extra_padding
        return util_image.image_remove_extra_padding(image)


class CombineImagesHorizontally(ImageTransform):
    """
    Aligns images of the input dataset in a row. Each feature of the input
    dataset is expected to be list of images. Each feature of the output
    dataset is a single image.
    """

    def __init__(self, padding_range: Tuple[int, int], combined_width: int):
        """
        Parameters
        ----------
        spacing_range:
            a (minimum, maximum) int pair (tuple), representing the min and max spacing
            between digits. Unit should be pixel.
        image_width:
            specifies the width of the image in pixels.
        """
        super().__init__()
        self.padding_range = padding_range
        self.combined_width = combined_width

    def transform(self, ds: dataset.ImageDataset) -> dataset.ImageDataset:
        def _get_image_fn(idx):
            cropped_images = ds.get_image_only(idx)
            combined_image, padding_size = self.transform_sample(
                cropped_images, padding_range=self.padding_range, combined_width=self.combined_width
            )
            return combined_image

        return dataset.ImageDataset(get_image_fn=_get_image_fn, meta=ds.meta)

    def transform_sample(
        self,
        image_list: List[np.ndarray],
        padding_range=(1, 1),
        combined_width=10,
        min_image_width=2,
        validate=True,
    ):
        """
        Combine images of individual digits horizontally to make image of the complete number
        Parameters
        ----------
        image_list: list of np.ndarray containing images of each digit
        padding_range: (minimum space between two digits, maximum space between two digits)
        combined_width: total width of the image
        returns
        -------
        final_image: combined image of number
        padding_size: padding between each digits in a number
        """
        n_dim = image_list[0].ndim  ### Assume all image same size
        n_digits = len(image_list)
        n_padding = n_digits - 1
        min_padding = padding_range[0]
        max_padding = padding_range[1]

        ##### Check minial image size
        assert n_padding * max_padding + min_image_width * n_digits < combined_width, "Error: "

        ##### Padding setup
        padding_list = util_image.padding_generate(
            n_padding, min_padding=padding_range[0], max_padding=max_padding
        )

        # log2(f"Padding between digits: {padding_list}")
        total_padding = np.sum(padding_list)
        assert (
            combined_width > total_padding + min_image_width * n_digits
        ), "Error: total final Image > padding*n_digits + min_width*n_digits"

        #####  New image size
        image_single_width2 = int((combined_width - total_padding) / n_digits)
        # heights = [img.shape[0] for img in image_list]
        # image_single_height2 = np.max(heights)  ### Take highest height
        image_single_height2 = 28  ## SPECSh
        """
         original_image_witdh :  same : remove white padding, ratio 
              width_padding, with_no_padding.
              
          padding : Fixed
          
          target_width :   "quality"    
          
          1 : widht  Not size, same padding.
          
             total bigger than size,
             
             full resize number : visibility.
             
             total image > bigger target:
                 crop, resize, re-rrange the numbers in 2D.
                 
             left, right image  target iamge smaller.
             
             
             too small : padding.
             too big :  
         
         
         
         
         target_ratio
        
        
        """

        ##### Adjust for rounding numbers
        ##### TODO : Issues if number of digits goes to infinity : too big adjustment
        madjust = combined_width - image_single_width2 * n_digits - total_padding
        image_single_width3 = image_single_width2 + madjust
        n_adjust = 1

        assert (
            image_single_width3 > min_image_width
        ), "error adjustment is too big for a single img, lower n_digits"

        assert (
            image_single_width2 * (n_digits - 1) + image_single_width3 + total_padding
            == combined_width
        ), "error target_width is not fullfilled"

        new_img_list = []  # to append resize images
        for i, img in enumerate(image_list):
            try:
                size_new = (image_single_width2, image_single_height2)
                if i < n_adjust:
                    #####Adjust for rounding numbers on 1st n_adjust image
                    size_new = (image_single_width3, image_single_height2)

                # log2("shape Old/New", img.shape, size_new)
                image_resize = cv2.resize(
                    img,
                    size_new,
                    interpolation=cv2.INTER_AREA,
                )
                new_img_list.append(image_resize)

            except Exception as e:
                raise Exception("Error during resizing:", e)

        # create an empty array with a size large enough to contain all the images + padding between images
        final_image, padding_size = util_image.image_merge(
            new_img_list, n_dim, padding_list, image_single_height2, combined_width
        )
        final_image = cv2.bitwise_not(final_image)  ### Invert to black --> white

        assert final_image.shape[1] == combined_width, "mismatch target_width"

        #### Validate image
        if validate:
            isok = image_padding_validate(final_image, min_padding, max_padding)
            if isok:
                return final_image, padding_size
            else:
                logw("Raw Image has noise: Padding not sastistied: skipping")
                ### TODO: Use a better Error Flag
                return np.zeros((1, 1, 3), dtype=np.uint8), []
        else:
            return final_image, padding_size


class ScaleImage(ImageTransform):
    """
    Scales images of the input dataset to a target size.
    """

    def __init__(
        self, width: Optional[int] = None, height: Optional[int] = None, inter=cv2.INTER_AREA
    ):
        """
        width and height specify output image dimensions
        """
        super().__init__()
        self.width = width
        self.height = height
        self.inter = inter

    def transform(self, ds: dataset.ImageDataset) -> dataset.ImageDataset:
        def _get_image_fn(idx):
            image = ds.get_image_only(idx)
            return self.transform_sample(image, self.width, self.height, self.inter)

        return dataset.ImageDataset(get_image_fn=_get_image_fn, meta=ds.meta)

    def transform_sample(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        """
        Resizes a image and maintains aspect ratio.
        """
        return util_image.image_resize(image, width, height, inter=cv2.INTER_AREA)


class TextToImage:
    """
    Uses image font to generate images of the input text.
    Font is defined as an image dataset with character images. Each
    character in font can have more than one image.
    """

    def __init__(
        self, font_dir: Union[str, pathlib.Path], spacing_range: Tuple[int, int], image_width: int
    ):
        """
        Parameters
        ----------
        font_dir: directory with the font
        spacing_range:
            a (minimum, maximum) int pair (tuple), representing
            the min and max spacing between characters. Unit should be pixel.
        image_width:
            specifies the width of the output image in pixels.
        """

        font_ds = dataset.ImageDataset(path=font_dir)
        font_ds = RemoveWhitePadding().fit_transform(font_ds)
        self.trans_list = [  # input feature: str, input label: str
            CharToImages(font_ds),  #
            CombineImagesHorizontally(padding_range=spacing_range, combined_width=image_width),  #
        ]

    def transform(self, ds: dataset.NlpDataset) -> dataset.ImageDataset:
        for tr in self.trans_list:
            ds = tr.transform(ds)
        return ds

    def fit(self, ds: dataset.NlpDataset):
        return self

    def fit_transform(self, ds: dataset.NlpDataset) -> dataset.ImageDataset:
        """
        Updates internal parameters of this transformation and then
        applies it to the provided dataset.
        """
        return self.fit(ds).transform(ds)
