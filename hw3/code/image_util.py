""" Helper functions for dealing with images
"""
import numpy as np
import matplotlib.pyplot as plt

def show_image(ax, img):
    """"
    :param img: 2d numpy matrix representing the image.
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

def img_to_vector(img):
    """
    Parameters
    - img (2d np matrix).
    Return
    - A 1d vector, the flattened image, row by row.
    """
    return img.reshape((img.shape[0] * img.shape[1], 1))

def vector_to_img(vec, img_shape):
    return vec.reshape(img_shape)

def imgs_to_matrix(img_lst):
    """
    Parameters
    - img_lst (list of 2d np matrices).
    Return
    - A matrix where each column is a flattened image.
    """
    return np.column_stack(img_lst).astype(np.int16)

def get_reconstructed_face(u, rank, original_vec):
  """ Get a low-rank reconstruction of a face.
  Parameters:
  - u - the eigenface matrix. The columns are eigenfaces.
  - rank - the low rank to do reconstruction in.
  - original_vec - the vector representation of a face. Mean is already subtracted.
  Return:
  Low-rank reconstruction of original_vec.
  """
  ur = u[:,:rank]
  return ur @ (ur.T @ original_vec)
