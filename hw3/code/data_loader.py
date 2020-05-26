"""Utility functions for accessing the photo data.
"""

from enum import Enum
import re
import numpy as np
import image_util
import matplotlib.pyplot as plt
from PIL import Image
import os

class Lighting(Enum):
    BRIGHT = 1
    LEFT_DARK = 2
    SLIGHT_DARK = 3

ALL_CROPPED_CODES = [
    "+000E+00",
    "+000E+20",
    "+000E+45",
    "+000E+90",
    "+000E-20",
    "+000E-35",
    "+005E+10",
    "+005E-10",
    "+010E+00",
    "+010E-20",
    "+015E+20",
    "+020E+10",
    "+020E-10",
    "+020E-40",
    "+025E+00",
    "+035E+15",
    "+035E+40",
    "+035E+65",
    "+035E-20",
    "+050E+00",
    "+050E-40",
    "+060E+20",
    "+060E-20",
    "+070E+00",
    "+070E+45",
    "+070E-35",
    "+085E+20",
    "+085E-20",
    "+095E+00",
    "+110E+15",
    "+110E+40",
    "+110E+65",
    "+110E-20",
    "+120E+00",
    "+130E+20"
    "-005E+10",
    "-005E-10",
    "-010E+00",
    "-010E-20",
    "-015E+20",
    "-020E+10",
    "-020E-10",
    "-020E-40",
    "-025E+00",
    "-035E+15",
    "-035E+40",
    "-035E+65",
    "-035E-20",
    "-050E+00",
    "-050E-40",
    "-060E+20",
    "-060E-20",
    "-070E+00",
    "-070E+45",
    "-070E-35",
    "-085E+20",
    "-085E-20",
    "-095E+00",
    "-110E+15",
    "-110E+40",
    "-110E+65",
    "-110E-20",
    "-120E+00",
    "-130E+20",
]

LIGHTING_TO_CROPPED_CODE = {
  Lighting.BRIGHT: "-005E-10",
  Lighting.LEFT_DARK: "-070E+00",
  Lighting.SLIGHT_DARK: "-005E+10"
}

IDENTITY_TRAIN_CODES = [
    "-005E-10",
    "-005E+10",
    "-010E-20",
    "-010E+00",
    "-015E+20",
    # Slight left dark
    "-060E-20",
    # Slight right dark
    "+070E-35"
]

IDENTITY_TEST_CODES = [
    # Left very dark
    "-095E+00",
    # Right very dark
    "+110E+15"
]

IDENTITY_PERSON_IDS = set([1,2,3,4,5,6,8,9,11,12])

LIGHTING_TO_UNCROPPED_CODE = {
  Lighting.BRIGHT: "centerlight",
  Lighting.LEFT_DARK: "rightlight",
}

SMALL_CROPPED_IMG_SHAPE = (38, 34)
# Rescale with 0.2
CROPPED_IMG_SHAPE = (192, 168)
UNCROPPED_IMG_SHAPE = (243, 320)

FEMALE_PERSON_IDS = set([5,15, 16, 24, 27, 28, 32, 34, 37])
# 9 test. 6 male, 3 female.
TEST_PERSON_IDS = [
    # Male
    1, 2, 3, 4, 6, 7,
    # Female
    5, 15, 16
]
# 29 train. 23 male, 6 female.
TRAIN_PERSON_IDS = [x for x in range(1, 40) if x not in TEST_PERSON_IDS]
# 14 is missing.
TRAIN_PERSON_IDS.remove(14)

ALL_IDS = TRAIN_PERSON_IDS + TEST_PERSON_IDS

def get_cropped_image_path(person_id, lighting_str):
  """
  Parameters:
  - person_id (int) from 1 to 39
  - lighting_str E.g. +130E+20
  Return:
  - Relative file path of the image
  """
  return f"data/yale_cropped/yaleB{person_id:02}/yaleB{person_id:02}_P00A" + \
     f"{lighting_str}.pgm"

def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html
    Taken from https://stackoverflow.com/a/7369986
    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))

def load_cropped_image(person_id, lighting):
  """
  See get_cropped_image_path.
  Return:
  - 2D np array representing the image.
  """
  return load_cropped_image_from_code(person_id, LIGHTING_TO_CROPPED_CODE[lighting])

def load_cropped_image_from_code(person_id, lighting_code):
  """
  See get_cropped_image_path.
  Return:
  - 2D np array representing the image.
  """
  return read_pgm(get_cropped_image_path(person_id, lighting_code))


def get_identity_train_matrix():
    """Get all the cropped images for training identity recognition task.
    See imgs_to_matrix() for the format.
    """
    imgs = []
    for lighting_code in IDENTITY_TRAIN_CODES:
        for person_id in IDENTITY_PERSON_IDS:
            imgs.append(image_util.img_to_vector(
                load_cropped_image_from_code(person_id, lighting_code)))
    return image_util.imgs_to_matrix(imgs)

def get_identity_train_labels():
    """Get the identity labels corresponding to get_identity_train_matrix
    Return:
    - 1D int array, each entry from 1 to 5, the person id.
    """

    return np.array(list(range(1, 1 + len(IDENTITY_PERSON_IDS))) * len(IDENTITY_TRAIN_CODES))


def get_identity_test_matrix():
    """Get all the cropped images for testing identity recognition task.
    See imgs_to_matrix() for the format.
    """
    imgs = []
    for lighting_code in IDENTITY_TEST_CODES:
        for person_id in IDENTITY_PERSON_IDS:
            imgs.append(image_util.img_to_vector(
                load_cropped_image_from_code(person_id, lighting_code)))
    return image_util.imgs_to_matrix(imgs)


def get_identity_test_labels():
    """Get the identity labels corresponding to get_identity_test_matrix
    Return:
    - 1D int array, each entry from 1 to 5, the person id.
    """
    return np.array(list(range(1, 1 + len(IDENTITY_PERSON_IDS))) * len(IDENTITY_TEST_CODES))

def get_all_cropped_matrix():
    """Get all the cropped images for SVD-ing.
    See imgs_to_matrix() for the format.
    """
    imgs = []
    for person_id in ALL_IDS:
      base_dir = f"data/yale_cropped/yaleB{person_id:02}/"
      for fname in os.listdir(base_dir):
        fpath = os.path.join(base_dir, fname)
        img = np.array(Image.open(fpath))
        imgs.append(image_util.img_to_vector(img))
    return image_util.imgs_to_matrix(imgs)

def get_gender_train_matrix():
    """Get the train data matrix for gender classification task.
    See imgs_to_matrix() for the format.
    """
    imgs = [image_util.img_to_vector(load_cropped_image(person_id, Lighting.BRIGHT)) \
            for person_id in TRAIN_PERSON_IDS]
    return image_util.imgs_to_matrix(imgs)

def get_gender_train_labels():
    """
    Get the gender annotation corresponding to get_gender_train_matrix
    Return:
    - 1D boolean array. True = female. False = male
    """
    return np.array([person_id in FEMALE_PERSON_IDS for person_id in TRAIN_PERSON_IDS])

def get_gender_test_matrix():
    """Get the test data matrix for gender classification task.
    See imgs_to_matrix() for the format.
    """
    imgs = [image_util.img_to_vector(load_cropped_image(person_id, Lighting.BRIGHT)) \
            for person_id in TEST_PERSON_IDS]
    return image_util.imgs_to_matrix(imgs)

def get_gender_test_labels():
    """
    Get the gender annotation corresponding to get_gender_test_matrix
    Return:
    - 1D boolean array. True = female. False = male
    """
    return np.array([person_id in FEMALE_PERSON_IDS for person_id in TEST_PERSON_IDS])

def get_uncropped_image_path(person_id, lighting):
  """
  Parameters:
  - person_id (int) from 1 to 15.
  - lighting (Lighting enum) E.g. Lighting.BRIGHT
  Return:
  - Relative file path of the image
  """
  return f"data/yale_uncropped/subject{person_id:02}.{LIGHTING_TO_UNCROPPED_CODE[lighting]}"

def load_uncropped_image(person_id, lighting):
  """
  See get_cropped_image_path.
  Return:
  - 2D np array representing the image.
  """
  return plt.imread(get_uncropped_image_path(person_id, lighting))

def get_all_uncropped_matrix():
    """Get all the uncropped images for SVD-ing.
    See imgs_to_matrix() for the format.
    """
    imgs = []
    base_dir = f"data/yale_uncropped/"
    for fname in os.listdir(base_dir):
      fpath = os.path.join(base_dir, fname)
      img = np.array(Image.open(fpath))
      imgs.append(image_util.img_to_vector(img))
    return image_util.imgs_to_matrix(imgs)

def get_clustering_data_matrix():
  imgs = [image_util.img_to_vector(load_cropped_image(person_id, Lighting.LEFT_DARK)) \
          for person_id in ALL_IDS]
  imgs.extend(
    [image_util.img_to_vector(load_cropped_image(person_id, Lighting.BRIGHT)) \
          for person_id in ALL_IDS])
  return image_util.imgs_to_matrix(imgs)

def get_clustering_data_label():
  # The natural clusters are based off lighting.
  # 0 is bright, 1 is slight dark.
  return [0] * 38 + [1] * 38
