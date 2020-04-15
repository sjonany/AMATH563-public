from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
import struct as st
import time

def read_images(image_path):
  # Adapted from https://medium.com/@mannasiladittya/converting-mnist-data-in-idx-format-to-python-numpy-array-5cb9126f99f1
  with open(image_path,'rb') as image_file:
    image_file.seek(0)
    magic = st.unpack('>4B',image_file.read(4))
    num_images = st.unpack('>I',image_file.read(4))[0] 
    num_rows = st.unpack('>I',image_file.read(4))[0]
    num_cols = st.unpack('>I',image_file.read(4))[0]

    images = np.zeros((num_images, num_rows, num_cols))
    total_bytes = num_images * num_rows * num_cols
    images_array = np.asarray(st.unpack('>'+'B'*total_bytes, \
      image_file.read(total_bytes))).reshape((num_images, num_rows, num_cols))
    return images_array

def read_labels(label_path):
  with open(label_path,'rb') as label_file:
    label_file.seek(0)
    magic = st.unpack('>4B',label_file.read(4))
    num_labels = st.unpack('>I',label_file.read(4))[0] 
    total_bytes = num_labels
    labels_array = np.asarray(st.unpack('>'+'B'*num_labels, label_file.read(num_labels)))
    return labels_array

def convert_labels_to_binary_vector(labels):
  """
  Convert the digit labels into binary arrays
  """
  labels = labels.reshape((1, len(labels)))
  binary_mat = np.zeros((labels.size, labels.max()+1))
  binary_mat[np.arange(labels.size), labels] = 1
  return binary_mat

def convert_regressed_vec_to_digit(vec):
  """
  Convert the regression result to a single digit prediction.
  We do this by finding an element that is closest to 1.
  """
  return np.argmin(np.abs(vec - 1))

def get_predictions(model, images):
  regress_vecs = model.predict(images)
  # Convert each row of regressed vector to a digt
  preds = np.apply_along_axis(convert_regressed_vec_to_digit, 1, regress_vecs)
  return preds

def get_errors(predictions, labels):
  return np.argwhere(labels != predictions).flatten()

def get_accuracy(num_error, num_samples):
  return 1.0 * (num_samples - num_error ) / num_samples

def plot_important_pixels(ax, pixel_ids):
  num_row = 28
  num_col = 28
  flat_img = np.zeros(num_row * num_col)
  flat_img[pixel_ids] = 1
  img = flat_img.reshape((num_row, num_col))
  ax.imshow(img, interpolation='nearest', cmap='Greys')

def subsample_pixels(images, pixel_indices):
  return images[:, pixel_indices]

def get_kfold_accuracy(model, train_flat_images, train_binary_labels, train_labels):
  num_fold = 5
  kf = KFold(n_splits=num_fold, random_state = 1, shuffle = True)
  splits = kf.split(train_flat_images)
  fold = 1
  total_acc = 0
  for train_indices, test_indices in splits:
    start_time = time.time()
    fold_train_images = train_flat_images[train_indices, :]
    fold_test_images = train_flat_images[test_indices, :]
    fold_train_binary_labels = train_binary_labels[train_indices, :]
    fold_test_labels = train_labels[test_indices]
    model.fit(fold_train_images, fold_train_binary_labels)
    preds = get_predictions(model, fold_test_images)
    errs = get_errors(preds, fold_test_labels)
    acc = get_accuracy(len(errs), len(fold_test_labels))
    print("Fold %d takes %d seconds, accuracy = %.2f" %
          (fold, time.time() - start_time, acc))
    fold += 1
    total_acc += acc
  return 1.0 * total_acc / num_fold