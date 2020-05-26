"""Non-trivial analysis codes.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import image_util
import data_loader
import pdb

def plot_energy_from_singular_values(data):
  U, singular_vals, Vh = np.linalg.svd(data, full_matrices = False)
  squared_singular_vals = np.power(singular_vals, 2)
  energy_values = squared_singular_vals / np.sum(squared_singular_vals)
  fig, ax = plt.subplots(figsize = (5, 5))
  ax.plot(range(1 , len(energy_values) + 1), energy_values, linestyle='--', marker='o', color='b')
  ax.set_xlabel("Singular value order")
  ax.set_ylabel("Energy")
  ax.xaxis.label.set_fontsize(15)
  ax.yaxis.label.set_fontsize(15)
  ax.set_title("Singular value energy distributions", fontdict={'fontsize': 15})

  energy_cumsum = np.cumsum(energy_values)
  for energy_percent in [90, 95, 99, 99.9]:
    # Index of first item greater than percent.
    mode = np.argmax(energy_cumsum > energy_percent / 100) + 1
    print(f"Need {mode} modes to cover {energy_percent} energy")

  return fig

def get_kfold_accuracy(model, train_mat, train_labels):
  """
  Get classification accuracy score from k-fold validation.
  :param model:
  :param train_mat:
  :param train_labels:
  :return:
  """
  num_fold = 3
  kf = StratifiedKFold(n_splits=num_fold, random_state = 1, shuffle = True)
  splits = kf.split(train_mat.T, train_labels)
  fold = 1
  total_acc = 0
  for train_indices, test_indices in splits:
    fold_train_images = train_mat[:, train_indices].T
    fold_test_images = train_mat[:, test_indices].T
    fold_train_labels = train_labels[train_indices]
    fold_test_labels = train_labels[test_indices]
    model.fit(fold_train_images, fold_train_labels)
    preds = model.predict(fold_test_images)
    acc =  1.0 * np.sum(preds == fold_test_labels) / len(preds)
    fold += 1
    total_acc += acc
  return 1.0 * total_acc / num_fold

def get_kfold_f1(model, train_mat, train_labels):
  """
  Get F1 score from k-fold validation.
  We use stratified k-fold to deal with the imbalanced class distribution.
  """
  num_fold = 2
  kf = StratifiedKFold(n_splits=num_fold, random_state = 1, shuffle = True)
  splits = kf.split(train_mat.T, train_labels)
  fold = 1
  total_f1 = 0
  for train_indices, test_indices in splits:
    fold_train_images = train_mat[:, train_indices].T
    fold_test_images = train_mat[:, test_indices].T
    fold_train_labels = train_labels[train_indices]
    fold_test_labels = train_labels[test_indices]
    model.fit(fold_train_images, fold_train_labels)
    preds = model.predict(fold_test_images)
    f1 = f1_score(fold_test_labels, preds)
    fold += 1
    total_f1 += f1
  return 1.0 * total_f1 / num_fold

def gender_analyze_errors(labels, preds, images):
  # TN = males correctly identified
  # FP = males misclassified as females
  # FN = Females misclassified as males
  # TP = Females correcly classified.
  tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
  print(f"Number of correctly identified males = {tn}")
  print(f"Number of correctly identified females = {tp}")
  print(f"Number of males misidentified as females = {fp}")
  print(f"Number of females misidentified as males  = {fn}")

  fig, axes = plt.subplots(ncols = 3, figsize=(9,3))
  ax_i = 0
  for g1 in [0, 1]:
    for g2 in [1, 0]:
      # argmax stops at first index of true
      img_i = np.argmax(np.logical_and(labels == g1, preds == g2))
      if labels[img_i] == g1 and preds[img_i] == g2:
        print(f"{g1}-> {g2}")

        image_util.show_image(axes[ax_i], image_util.vector_to_img(images[:, img_i],
                                                             data_loader.CROPPED_IMG_SHAPE))
        ax_i += 1
  return fig

def score_cluster(preds, labels):
  """
  Score 2-cluster results by classification accuracy, picking the class mapping
  that gives the higher score.
  """
  score1 = 0
  score2 = 0
  for i in range(len(preds)):
    if preds[i] == labels[i]:
      score1 += 1
    else:
      score2 += 1
  return 1.0 * max(score1, score2) / len(labels)