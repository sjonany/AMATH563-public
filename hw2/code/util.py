"""
Utility functions for plotting or data parsing.
"""
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
import time

FIRST_YEAR = 1845
LAST_YEAR = 1903
NEW_DT = 0.1
ORIGINAL_TIMEPOINTS = np.arange(FIRST_YEAR, LAST_YEAR+1, 2)
INTERPOLATED_TIMEPOINTS = np.arange(FIRST_YEAR, LAST_YEAR + NEW_DT, NEW_DT)
POP_FIG_SIZE = (5, 3)

def get_interpolated_one_row(data_row):
  tck = interpolate.splrep(ORIGINAL_TIMEPOINTS, data_row)
  ynew = interpolate.splev(INTERPOLATED_TIMEPOINTS, tck)
  return INTERPOLATED_TIMEPOINTS, ynew

def get_interpolated_data(data):
  """
  Use cubic-spline interpolation to expand out original data's 30 time points.
  """
  xnew, y0new = get_interpolated_one_row(data[0])
  xnew, y1new = get_interpolated_one_row(data[1])
  return np.array([y0new, y1new])

def compare_pred_with_data_one_pop(ax, data, preds, row, animal):
  color = get_color(animal)
  times = INTERPOLATED_TIMEPOINTS
  
  ax.plot(times, data[row, :], color = color, alpha = 0.5, linestyle = 'dashed')
  ax.plot(times, preds[row, :], color = color, label = animal)
  ax.set_xlabel("Year")
  ax.set_ylabel("Population")
  ax.xaxis.label.set_fontsize(15)
  ax.yaxis.label.set_fontsize(15)
  ax.legend()
  
  print("%s L2 norm discrepancy = %.2f " % (animal, np.linalg.norm(data[row, :] - preds[row,:])))

def compare_pred_with_data(data, preds):
  fig, ax = plt.subplots(figsize = POP_FIG_SIZE)
  compare_pred_with_data_one_pop(ax, data, preds, row = 0, animal = "Hare")
  compare_pred_with_data_one_pop(ax, data, preds, row = 1, animal = "Lynx")
  return fig

def score_population_pred(data, preds):
  """
  Score multiple time series fits by taking frobenius norm of the deviation.
  """
  return np.linalg.norm(data - preds)

def get_color(animal):
  if animal == "Hare":
    return 'blue'
  elif animal == "Lynx":
    return 'red'
  raise Exception(f'Unknown animal {animal}')

def compare_interpolation_with_data_one_pop(ax, data, interpolated_data, row, animal):
  color = get_color(animal)
  ax.plot(ORIGINAL_TIMEPOINTS, data[row, :], label = animal, color=color)
  ax.plot(INTERPOLATED_TIMEPOINTS, interpolated_data[row, :],
          color=color, linestyle='dashed')
  ax.set_xlabel("Year")
  ax.set_ylabel("Population")
  ax.xaxis.label.set_fontsize(15)
  ax.yaxis.label.set_fontsize(15)
  ax.legend()

def compare_interpolation_with_data(data, interpolated):
  fig, ax = plt.subplots(figsize = POP_FIG_SIZE)
  compare_interpolation_with_data_one_pop(ax, data, interpolated, row = 0, animal = "Hare")
  compare_interpolation_with_data_one_pop(ax, data, interpolated, row = 1, animal = "Lynx")
  return fig

def visualize_video(mat):
  """
  Visualize matrix as a video.
  Matrix[t, y, x] is the ordering of the axes.
  """
  fig, ax = plt.subplots(figsize=(5,5))
  time_points = mat.shape[0]
  for t in range(time_points):
    frame = mat[t,:,:]
    ax.pcolormesh(frame, shading='interp')
    display.clear_output(wait=True)
    display.display(plt.gcf())
    time.sleep(0.1)
  return fig

def plot_sindy_coefs(dx_library_coefs, dy_library_coefs, f_lib):
  fig, ax = plt.subplots(figsize=POP_FIG_SIZE)
  N = len(dx_library_coefs)
  # the x locations for the groups
  ind = np.arange(N)  
  # the width of the bars
  width = 0.5
  ax.bar(ind, dx_library_coefs, width, color='royalblue', label="dx/dt")
  ax.bar(ind + width , dy_library_coefs, width, color='seagreen', label="dy/dt")
  ax.set_xticks(ind + width / 2)
  ax.set_xticklabels(get_term_names_from_f_lib(f_lib), fontsize=10)
  ax.set_ylabel("Coefficient")
  ax.yaxis.label.set_fontsize(15)
  ax.legend()
  return fig

def get_term_names_from_f_lib(f_lib):
  return [name_f[0] for name_f in f_lib]

def trim_sindy_coefs(dx_library_coefs, dy_library_coefs, coef_threshold = 0.1):
  dx_library_coefs[np.abs(dx_library_coefs) < coef_threshold] = 0
  dy_library_coefs[np.abs(dy_library_coefs) < coef_threshold] = 0

def get_sindy_expression_str(coefs, f_lib):
  nonzero_terms = []
  term_names = get_term_names_from_f_lib(f_lib)
  for i, name in enumerate(term_names):
    if coefs[i] != 0:
      nonzero_terms.append(f'{coefs[i]} {term_names[i]}')
  return (" + ".join(nonzero_terms))

def pretty_print_sindy_coefs(dx_library_coefs, dy_library_coefs, f_lib):
  print("dx/dt = " + get_sindy_expression_str(dx_library_coefs, f_lib))
  print("dy/dt = " + get_sindy_expression_str(dy_library_coefs, f_lib))