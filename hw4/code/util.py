from pymatreader import read_mat
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import cv2
import time
from mpl_toolkits import mplot3d

def load_ks_data(data_path):
  data = read_mat(data_path)
  # x[i] gives you the fixed spatial locations
  # t[j] gives you the fixed time points
  # uu[i,j] gives you u(x[i], t[j])
  return data['x'], data['tt'], data['uu']

def load_lorenz_data(data_path):
  data = read_mat(data_path)
  # ts[ti] are the timepoints
  # ys[ti, dim] stores x[t], y[t], z[t]
  return data['t'], data['y']

def plot_u(x_list, t_list, u_mat):
  fig, ax = plt.subplots(figsize=(5,5))
  t_min = t_list[0]
  t_max = t_list[-1]
  x_min = x_list[0]
  x_max = x_list[-1]
  extent = [t_min , t_max, x_min , x_max]
  ax.imshow(u_mat, aspect=1.0, extent = extent)
  ax.set_xlabel("Time (s)")
  ax.set_ylabel("Position")
  return fig

def visualize_video(mat):
  """
  Visualize matrix as a video.
  Matrix[x, y, t] is the ordering of the axes.
  Example usage for reaction_diffusion.
    ts = list(range(0,50,5))
    util.visualize_video(u_mat[::10,::10,ts])
  """
  fig, ax = plt.subplots(figsize=(5,5))
  time_points = mat.shape[2]
  for t in range(time_points):
    frame = mat[:,:,t]
    ax.pcolormesh(frame, shading='interp')
    display.clear_output(wait=True)
    display.display(plt.gcf())
    time.sleep(0.1)
  return fig

def downsample_img(img):
  return cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)

def plot_w(w_vec):
  """
  w_vec is a slice from core.uv_to_wmat
  That is, it is flattened u and v concatenated together.
  """
  fig, axes = plt.subplots(ncols = 2, figsize = (6,3))
  r = int(np.sqrt(len(w_vec) / 2))
  c = r
  u = w_vec[:r*c].reshape((r,c))
  v = w_vec[r*c:].reshape((r,c))
  axes[0].imshow(u)
  axes[1].imshow(v)
  for ax in axes:
    ax.set_axis_off()
  return fig

def plot_lorenz_2ds(ts, ys):
  """Plot lorenz trajectory using three 2D graphs, one of x,y and z.
  See the format from load_lorenz_data()
  """
  fig, axes = plt.subplots(nrows = 3, figsize=(7, 3 * 3))
  dim_names = ['x','y','z']
  for dim in range(3):
    ax = axes[dim]
    ax.plot(ts, ys[:, dim])
    ax.set_title(dim_names[dim])
  return fig

def plot_lorenz_3d(ts,ys):
  fig = plt.figure(figsize=(21,7))
  ax = fig.add_subplot(1, 2, 1, projection='3d')
  ax.set_xlabel("X", fontsize=10)
  ax.set_ylabel("Y", fontsize=10)
  ax.set_zlabel("Z", fontsize=10)
  ax.plot(ys[:,0], ys[:,1], ys[:,2])
  return fig

def plot_lorenz_2ds_compare(ts, ys_gold, ys_pred):
  """Plot lorenz trajectory using three 2D graphs, one of x,y and z.
  See the format from load_lorenz_data()
  """
  fig, axes = plt.subplots(nrows = 3, figsize=(7, 3 * 3))
  dim_names = ['X','Y','Z']
  for dim in range(3):
    ax = axes[dim]
    ax.plot(ts, ys_gold[:, dim], label="Gold", c="orange")
    ax.plot(ts, ys_pred[:, dim], label="Prediction", c="blue")
    ax.set_title(dim_names[dim])
    ax.legend(fontsize=15)
  return fig

def plot_lorenz_3d_compare(ts, ys_gold, ys_pred):
  fig = plt.figure(figsize=(21,7))
  ax = fig.add_subplot(1, 2, 1, projection='3d')
  ax.set_xlabel("X", fontsize=20)
  ax.set_ylabel("Y", fontsize=20)
  ax.set_zlabel("Z", fontsize=20)
  ax.plot(ys_gold[:,0], ys_gold[:,1], ys_gold[:,2], label="Gold", c="orange")
  ax.plot(ys_pred[:,0], ys_pred[:,1], ys_pred[:,2], label="Prediction", c="blue")
  ax.legend(fontsize=20)
  return fig
