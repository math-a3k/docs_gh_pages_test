"""
Image related utils
https://pypi.org/project/pysie/#description

"""
import os, sys, io, pandas as pd, numpy as np
import typing
import urllib.request
import urllib.parse



def log(*s):
    print(s)


def deps():
    with open('dependencies/reqs_image.txt') as fp:
        txt = fp.readlines()
    print(txt)


###################################################################################################
def read_image(filepath_or_buffer: typing.Union[str, io.BytesIO]):
    """Read a file into an image object
    Args:
        filepath_or_buffer: The path to the file, a URL, or any object
            with a `read` method (such as `io.BytesIO`)
    """
    import cv2
    import tifffile.tifffile
    import validators

    if isinstance(filepath_or_buffer, np.ndarray):
        return filepath_or_buffer
    if hasattr(filepath_or_buffer, 'read'):
        image = np.asarray(bytearray(filepath_or_buffer.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    elif isinstance(filepath_or_buffer, str):
        if validators.url(filepath_or_buffer):
            return read(urllib.request.urlopen(filepath_or_buffer))
        assert os.path.isfile(filepath_or_buffer), \
            'Could not find image at path: ' + filepath_or_buffer
        if filepath_or_buffer.endswith('.tif') or filepath_or_buffer.endswith('.tiff'):
            image = tifffile.imread(filepath_or_buffer)
        else:
            image = cv2.imread(filepath_or_buffer)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# helper function for data visualization
def visualize_in_row(**images):
    """Plot images in one row."""
    import matplotlib.pyplot as plt
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


# Resizes a image and maintains aspect ratio
def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size and initialize dimensions
    import cv2
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image

    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)

#############################################################################
#############################################################################
