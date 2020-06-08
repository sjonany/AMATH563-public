from sklearn.model_selection import KFold
import numpy as np
import time
import pdb
import matplotlib.pyplot as plt

def u_mat_to_regression(u_mat):
  """
  Convert u(x,t) matrix to multi-variate regression inputs and outputs
  Each sample is an (input, output) tuple.
  input is u(x,t), of size |X|, one value for each x for the fixed t.
  output is u(x, t+dt), also of size |X|, one value for each x of the fixed t + dt.
  """
  num_time = u_mat.shape[1]
  num_space = u_mat.shape[0]
  input_mat = np.zeros((num_time - 1, num_space))
  output_mat = np.zeros((num_time - 1, num_space))
  for t in range(num_time - 1):
    input_mat[t, :] = u_mat[:, t]
    output_mat[t, :] = u_mat[:, t+1]
  return input_mat, output_mat

def lorenz_mat_to_regression(ys, rho):
  """
  Given ys from util.load_lorenz_data, where ys[ti, dim] stores x[t], y[t], z[t],
  and rho, one of the Lorenz parameter,
  convert this into a regression matrix for training NN.
  """
  input_mat, output_mat = u_mat_to_regression(ys)
  # We add rho as an additional feature so the NN knows how to advance different rhos
  # differently.
  r = input_mat.shape[0]
  input_mat = np.hstack((input_mat, (rho*np.ones(r)).reshape((r,1))))
  return input_mat, output_mat

def create_pred_u_mat(init_u, model, num_time):
  """
  Create a prediction of u(x,t) given u(x,t=0)
  where t goes from 0 to num_time - 1
  We do this by having model repeatedly predicting u(x, t+dt)
  and we keep feeding the previous step's output as an input for the next step.
  """
  # u(x,t)
  num_space = len(init_u)
  pred_u = np.zeros((num_space, num_time))

  ut = init_u
  pred_u[:, 0] = ut.T
  start_time = time.time()
  for t in range(1, num_time):
    if t % 10 == 0:
      print(f"t = {t}")
    # Use NN to advance the trajectory in time.
    ut = model.predict(ut.reshape((1, num_space)))
    pred_u[:,t] = ut
  print(f"NN prediction of u(x,t) takes {time.time() - start_time} s")
  return pred_u

def create_pred_lorenz(init_y, rho, model, num_time):
  """
  See create_pred_u_mat(), except that this is for Lorenz, where we accept rho
  as another input.
  """
  num_space = len(init_y)
  pred_y = np.zeros((num_space, num_time))

  yt = init_y
  pred_y[:, 0] = yt.T
  start_time = time.time()
  for t in range(1, num_time):
    if t % 10 == 0:
      print(f"t = {t}")
    # Use NN to advance the trajectory in time.
    nn_input = yt.reshape((1, num_space))
    # Need to attach the extra rho feature.
    nn_input = np.append(nn_input, rho)
    nn_input = nn_input.reshape((1, len(nn_input)))
    yt = model.predict(nn_input)
    pred_y[:,t] = yt
  print(f"NN prediction of full lorenz trajectory " +
    f"takes {time.time() - start_time} s")
  return pred_y

def get_kfold_mse(model, input_mat, output_mat):
  """
  Get the kfold mse for the provided model
  """

  num_fold = 3
  kf = KFold(n_splits=num_fold, random_state = 1, shuffle = True)
  splits = kf.split(input_mat)
  fold = 1
  total_mse = 0
  for train_indices, test_indices in splits:
    fold_train_in = input_mat[train_indices, :]
    fold_test_in = input_mat[test_indices, :]
    fold_train_out = output_mat[train_indices]
    fold_test_out = output_mat[test_indices]

    model.fit(fold_train_in, fold_train_out, epochs=100, batch_size=1000, shuffle=True)
    mse = model.evaluate(fold_test_in, fold_test_out)[0]
    fold += 1
    total_mse += mse
  return 1.0 * total_mse / num_fold

def uv_to_wmat(u, v):
  """
  Convert u[x,y,t], v[x,y,t]
  To w[i,t], where each column of w is a flattened u, followed by flattened v.
  """
  # u[x,y,t]
  R = u.shape[0]
  C = u.shape[1]
  T = u.shape[2]
  w_len = 2 * R * C
  # w is a 2D matrix where each column is a flattened u and v.
  w_mat = np.zeros((w_len, T))
  for t in range(T):
    w_mat[:R*C, t] = u[:,:,t].reshape((R*C,))
    w_mat[R*C:, t] = v[:,:,t].reshape((R*C,))
  return w_mat

def mat_to_uv(w_mat): 
  """
  The inverse of uv_to_wmat()
  """
  T = w_mat.shape[1]
  R = int(np.sqrt(w_mat.shape[0] / 2))
  C = R
  u = np.zeros((C, R, T))
  v = np.zeros((C, R, T))
  for t in range(T):
    u[:,:,t] = w_mat[:R*C, t].reshape((R, C))
    v[:,:,t] = w_mat[R*C:, t].reshape((R, C))
  return u, v

def plot_energy_from_singular_values(singular_vals):
  """
  Plot energy distribution. singular_vals is the 's' from u,s,v = svd(mat)
  """
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
