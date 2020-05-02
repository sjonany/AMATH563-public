"""
Important mathy functions here.
"""
from matplotlib import pyplot as plt
import numpy as np
import pdb
import scipy.integrate as integrate
import scipy.stats
from sklearn import linear_model
import util


"""
3 default function libraries, representing polynomials of degree 1, 2 and 3.
Each library is a list of (name, function) tuple.
"""
F_LIB_3 = [
  ("x", lambda s: s[0]), 
  ("y", lambda s: s[1]),
  ("x2", lambda s: s[0] ** 2),
  ("xy", lambda s: s[0] * s[1]),
  ("y2", lambda s: s[1] ** 2),
  ("x3", lambda s: s[0] ** 3),
  ("x2y", lambda s: s[0] ** 2 * s[1]),
  ("xy2", lambda s: s[0] * s[1] ** 2),
  ("y3", lambda s: s[1] ** 3)
]
F_LIB_2 = F_LIB_3[:5]
F_LIB_1 = F_LIB_3[:2]

def dmd(data, r):
  """
  Perform dynamic mode decomposition on the data matrix.
  Parameters:
  - data (n x m+1 matrix) Rows are spatial locations and columns are time points.
    That is, there are n spatial locations and m time points.
  - r (int) The low-rank to perform DMD in.
  Returns:
  - Phi (n x r matrix) The DMD (spatial) modes.
  - Lambda (r x r matrix) The diagonal eigenvalue matrix.
  - b (r x 1 matrix). The low-rank initial condition 
  """
  X0 = data[:, :-1]
  X1 = data[:, 1:]

  # X0 is n x m 
  # U is n x n
  # S is n x n, a diagonal matrix of the singular values.
  # Vh is m x m
  U, singular_vals, Vh = np.linalg.svd(X0)
  S = np.diag(singular_vals)
  V = Vh.conj().T
  # Ur is n x r
  Ur = U[:,:r]
  # Sr is r x r
  Sr = S[:r, :r]
  # Vr is m x r
  Vr = V[:,:r]
  Sr_inv = np.linalg.inv(Sr)
  # Atilde is r x r
  Atilde = Ur.conj().T @ X1 @ Vr @ Sr_inv

  # Both W and Lambda are r x r 
  eigenvalues, W = np.linalg.eig(Atilde)
  Lambda = np.diag(eigenvalues)

  # Phi is n x r, the DMD spatial models
  Phi = X1 @ (Vr @ Sr_inv) @ W
  initial_condition = X0[:, 0]
  b = np.linalg.pinv(Phi) @ initial_condition
  
  return Phi, Lambda, b

def get_xt_from_dmd(num_time, Phi, Lambda, b):
  """
  Predict x_t given the DMD results.
  Parameters:
  - num_time. Number of time points to predict states for.
    We will predict x1, ..., x_num_time, where x1 is the predicted initial condition.
  - Phi, Lambda, b - See dmd()
  Return:
  - A (N x num_time) matrix, where N is the number of spatial coordinates.
    mat[:1] will give you the approximation to x1, the initial condition.
  """
  n = Phi.shape[0]
  r = Phi.shape[1]
  result = np.zeros((n, num_time))
  
  # x(t) = Phi Lambda^{t-1} b
  phi_times_lambda = Phi
  for t in range(num_time):
  	# Sometimes the eigenvalues are complex. We only care about the real part.
    result[:, t] = np.real(phi_times_lambda @ b)
    phi_times_lambda = phi_times_lambda @ Lambda
  return result

def get_sliding_dmd_predictions(data, r, start_predict_time, dmd_window = -1):
  """
  For each future timestep, obtain the DMD modes based on some of the previous timesteps,
  then predict just the 1 future timestep.
  The number of past timesteps to look at is exactly start_predict_time. 
  Parameters:
  - data (n x m matrix) Rows are spatial locations and columns are time points.
    That is, there are n spatial locations and m time points.
  - r (int) The low-rank to perform DMD in.
  - start_predict_time (int) Which time step (zero-based) to start prediction on.
  Return:
  - predicted x(t), (n x (m-start_predict_time+1 matrix). Each column i of this matrix is the
    predicted x(t), based by obtaining DM modes on a window of past time steps, then predicting x(t = i).
  """
  n = data.shape[0]
  m = data.shape[1]
  if dmd_window < 0:
    dmd_window = start_predict_time
  result = np.zeros((n, m - start_predict_time))
  for t in range(start_predict_time, m):
    # Perform DMD on 'dmd_window' time points before t
    Phi, Lambda, b = dmd(data[:, t-dmd_window:t], r)
    # Predict x(0..t)
    xts = get_xt_from_dmd(dmd_window + 1, Phi, Lambda, b)
    # Store just the predicted x(t)
    result[:, t - start_predict_time] = xts[:, dmd_window]
  return result

def time_delay_embed(data, k):
  """
  Create a time delay embedding of the data matrix, with k offsets.
  Parameters:
  - data (n x m matrix) Rows are spatial locations and columns are time points.
  - k (int) The number of embeddings to make. k = 1 means no embedding.
  Return:
  - matrix with (k*n rows, m-k+1 columns)
    The first n rows are the first k columns of data ~ data[all_rows, t = 1 -> k]
    The following n rows are the second k columns of data ~ data[all_rows, t = 2 -> k + 1]
    And so on
  """
  n = data.shape[0]
  m = data.shape[1]
  result = np.zeros((n*k, m-k+1))
  for delay_counter in range(k):
    start_row = n * delay_counter
    end_row_exclusive = start_row + n
    start_col = delay_counter
    end_col_exclusive = delay_counter + m-k+1
    result[start_row:end_row_exclusive, :] = data[:, start_col:end_col_exclusive]
  return result

def get_sliding_time_delayed_dmd_predictions(data, r, training_window, embedding_count, start_predict_time):
  """
  For each future timestep, obtain the DMD modes based on time-delayed embedding
  of a window the previous timesteps, then predict just the 1 future timestep.
  Parameters:
  - data (n x m matrix) Rows are spatial locations and columns are time points.
    That is, there are n spatial locations and m time points.
  - r (int) The low-rank to perform DMD in.
  - training_window (int) The number of past histories to use for prediction.
  - embedding_count (int) The number of time delayed embedding.
    Space dimension will be multiplied by this much.
  - start_predict_time (int) Which time step (zero-based) to start prediction on.
  Return:
  - predicted x(t), (n x (m-start_predict_time+1 matrix). Each column i of this matrix is the
    predicted x(t), for t = start_predict_time to m, except that the column starts from 0.
  """
  n = data.shape[0]
  m = data.shape[1]
  w = training_window
  k = embedding_count 
  result = np.zeros((n, m - start_predict_time))
  for t in range(start_predict_time, m):
    # The matrix to train against is just the past 'w' time points,
    # Excluding the t-th column, which we want to predict.
    train_mat = data[:, t-w:t]
    embedded_train_mat = time_delay_embed(train_mat, k)
    # Perform DMD on the embedded train matrix.
    print("Doing DMD..")
    Phi, Lambda, b = dmd(embedded_train_mat, r)
    print("Done doing dmd!!")
    """
    This is the column in the embedded space whose last N rows contains
      the predicted train_mat[column=w], which is the predicted X(t).
    This is because each column i (zero-based) captures time = i to i+k-1
      , which implies column w-k+1 captures time = w-k+1 to w
    """
    relevant_embedded_col = w - k + 1
    print("Getting xt...")
    embedded_prediction = get_xt_from_dmd(relevant_embedded_col + 1, Phi, Lambda, b)
    print("Done getting xt!")
    # The predicted train_mat[column = w] is just the last n rows.
    predicted_xt = embedded_prediction[-n:, relevant_embedded_col]
    # Store just the predicted x(t)
    result[:, t - start_predict_time] = predicted_xt
  return result

def plot_energy_from_singular_values(data):
  U, singular_vals, Vh = np.linalg.svd(data)
  squared_singular_vals = np.power(singular_vals, 2)
  energy_values = squared_singular_vals / np.sum(squared_singular_vals)
  fig, ax = plt.subplots(figsize = (5, 5))
  ax.plot(range(1 , len(energy_values) + 1), energy_values, linestyle='--', marker='o', color='b')
  ax.set_xlabel("Singular value order")
  ax.set_ylabel("Energy")
  ax.xaxis.label.set_fontsize(15)
  ax.yaxis.label.set_fontsize(15)
  ax.set_title("Singular value energy distributions", fontdict={'fontsize': 15})
  print("Percent energy covered: %s" % (np.cumsum(energy_values)))
  return fig

def plot_population_modes(data, top_k):
  U, singular_vals, Vh = np.linalg.svd(data)
  fig1, ax = plt.subplots(figsize = (5, 5))
  for k in range(top_k):
    mode = U[:,k]
    ax.plot(range(len(mode)), mode, label = f"Mode {k + 1}")
    ax.set_xlabel("Time-embedded population offset")
    ax.set_ylabel("Time-embedded year offset")
  ax.xaxis.label.set_fontsize(15)
  ax.yaxis.label.set_fontsize(15)
  ax.set_title("Time modes", fontdict={'fontsize': 15})
  ax.legend()

  fig2, ax = plt.subplots(figsize = (5, 5))
  for k in range(top_k):
    mode = Vh[k, :]
    ax.plot(range(len(mode)), mode, label = f"Mode {k + 1}")
    ax.set_xlabel("Time-embedded year offset")
    ax.set_ylabel("Time-embedded population offset")
  ax.xaxis.label.set_fontsize(15)
  ax.yaxis.label.set_fontsize(15)
  ax.set_title("Population modes", fontdict={'fontsize': 15})
  ax.legend()
  return fig1, fig2

def get_dx_dts(xts):
  """
  Use centered finite difference to approximate dx/dt, given x(t)'s
  
  Parameters:
  - xts (n x 1 column vector). The x(t) over time.
  Return:
  - dx_dts ((n-2) x 1 column vector). The time in front and end are removed.
  """
  dt = util.NEW_DT
  dx_dts = np.zeros(len(xts) - 2)
  for i in range(len(dx_dts)):
    dx_dts[i] = (xts[i+2] - xts[i]) / (2.0 * dt)
  return dx_dts

def gen_library_matrix(measurements, f_lib):
  """
  Generate a matrix of library functions evaluated against the measurements.
  Parameters:
  - measurements (n x m matrix). n is the dimension of each state variable,
    m is the number of timepoints the measurements were taken.
  - f_lib: list((name, fun(x_vec)). A list of 'L' library functions,
    where each function accepts a column from 'measurements' matrix, that is,
    a single state variable vector, snapshotted in time.
  Return:
    m x L matrix, where each column is a single library function evaluated across
    the m measurements.
  """
  n = measurements.shape[0]
  m = measurements.shape[1]
  L = len(f_lib)
  library_mat = np.zeros((m, L))
  
  for i, name_fun in enumerate(f_lib):
    fun = name_fun[1]
    # Apply fun to each column, which represents x(t)
    library_mat[:,i] = np.apply_along_axis(fun, 0, measurements)
  return library_mat

def get_best_fit_lotka_volterra_params(data):
  """
  Use linear regression to obtain the parameters for the ODEs below that best fit the data.
  dx/dt = (b-py)x
  dy/dt = (rx-d)y
  
  Parameters:
  - data (n x m matrix) Rows are spatial locations and columns are time points.
  
  Return:
  - b, p, r, d, the ODE parameters.
  """
  # Generate the RHS of Ax = b
  dx_dts = get_dx_dts(data[0,:])
  dy_dts = get_dx_dts(data[1,:])
  
  # Generate the LHS, A of Ax = b
  # Remove the first and last time samples, because finite difference removes them too.
  data_without_ends = data[:, 1:-1]
  # For dx/dt, our library is [x, -xy]
  # The shape is 28 (time) x 2 (number of library functions)
  x_library = gen_library_matrix(data_without_ends, [
    ("x", lambda x: x[0]),
    ("-xy", lambda x: -x[0] * x[1])
  ])
  # For dy/dt, our library is [-y, xy]
  y_library = gen_library_matrix(data_without_ends, [
    ("-y", lambda x: -x[1]),
    ("xy", lambda x: x[0] * x[1])
  ])
  model = linear_model.LinearRegression()
  
  model.fit(x_library, dx_dts)
  x_library_coefs = model.coef_
  b = x_library_coefs[0]
  p = x_library_coefs[1]
  model.fit(y_library, dy_dts)
  y_library_coefs = model.coef_
  d = y_library_coefs[0]
  r = y_library_coefs[1]
  return b, p, r, d

def generate_predictions_from_lotka_volterra(b, p, r, d):
  """
  Given the ODE as below, where time is in the scale of years,
    dx/dt = (b-py)x
    dy/dt = (rx-d)y
  return the predicted population formatted as the original data.
  That is, from 1845 to 1903 in increment of 2 years, a 2 x 30 matrix
  """
  def dyn(t, states):
    x = states[0]
    y = states[1]
    dx = (b - p * y) * x 
    dy = (r * x - d) * y
    return np.array([dx, dy])

  # 1845 to 1903 spans 58 years. Each unit time is 1 year. Let t = 0 be year 1845.
  t_max = 58
  t_span = [0, t_max]
  t_eval = np.arange(0, t_max + util.NEW_DT, util.NEW_DT)
  # Hare and lynx population in 1845
  init_conds = [20, 32]
  sol = integrate.solve_ivp(dyn, t_span, init_conds, method = 'RK45', t_eval = t_eval)
  preds = sol.y
  return preds

def get_sindy_params(data, regression_model, f_lib):
  """
  Use sparse regression to obtain the parameters for the ODEs below that best fit the data.
  dx/dt, and dy/dt are linear combinations of (x, y, x2, xy, y, x3, x2y, xy2, y3)
  Note that there are 9 terms.
  That is, we are doing SINDy, but with library functions of polynomials up to degree 3.
  
  Parameters:
  - data (n x m matrix) Rows are spatial locations and columns are time points.
  - regression_model. A scipy regression model to solve Ax = b
  Return:
  - an array of 9 numbers, which are the weights for (x, y, x2, xy, y2, x3, x2y, xy2, y3)
  """
  # Generate the RHS of Ax = b
  dx_dts = get_dx_dts(data[0,:])
  dy_dts = get_dx_dts(data[1,:])
  
  # Generate the LHS, A of Ax = b
  # Remove the first and last time samples, because finite difference removes them too.
  data_without_ends = data[:, 1:-1]
  # The shape is 28 (time) x 2 (number of library functions)
  x_library = gen_library_matrix(data_without_ends, f_lib)
  y_library = gen_library_matrix(data_without_ends, f_lib)
  
  regression_model.fit(x_library, dx_dts)
  dx_library_coefs = regression_model.coef_

  regression_model.fit(y_library, dy_dts)
  dy_library_coefs = regression_model.coef_
  return dx_library_coefs, dy_library_coefs

def generate_predictions_from_sindy(x_library_coefs, y_library_coefs, f_lib):
  """
  Generate the predicted population formatted as the original data
  by using ODE generated by SINDy. Please see get_sindy_params.
  That is, from 1845 to 1903 in increment of 2 years, a 2 x 30 matrix
  Parameters:
  - poly_weights. The return value of get_sindy_params.
  Return:
  - scipy's integrate sol obj. sol.y gives you the predictions.
    sol.success tells you if the integration was successful.
  """
  def dyn(t, states):
    dx = 0
    dy = 0
    for i, name_f in enumerate(f_lib):
      f = name_f[1]
      f_res = f(states)
      dx += x_library_coefs[i] * f_res
      dy += y_library_coefs[i] * f_res
    return np.array([dx, dy])

  # 1845 to 1903 spans 58 years. Each unit time is 1 year. Let t = 0 be year 1845.
  t_max = 58
  t_span = [0, t_max]
  t_eval = np.arange(0, t_max + util.NEW_DT, util.NEW_DT)
  # Hare and lynx population in 1845
  init_conds = [20, 32]
  sol = integrate.solve_ivp(dyn, t_span, init_conds, method = 'RK45', t_eval = t_eval)
  return sol

def get_population_joint_distribution(population_data, pseudocount = 0.001):
  """
  Parameters
  - data (2 x m matrix) Rows are population types and columns are time points.
  - pseudocount. Laplacian smoothing parameter for never-seen samples.
  Returns
  - 200 x 200 float matrix where mat[i, j] is P(hare = i, lynx = j) at any time point.
    The joint probability is guaranteed to add up to 1 when summed over the entire sample space.
  """
  counts = np.zeros((200, 200))
  # Additive smoothing. For unseen, assume a pseudocount.
  counts += pseudocount
  for t in range(population_data.shape[1]):
    hare = int(population_data[0, t])
    lynx = int(population_data[1, t])
    counts[hare, lynx] += 1
  counts /= np.sum(counts)
  return counts

def get_kl_divergence(gold_distribution, pred_distribution):
  """
  Compute KL divergence of two probability distributions produced by
  get_population_joint_distribution()
  The sample space is all possible integer (hare, lynx) pairs from 0 to 200.
  """
  result = 0
  for hare in range(gold_distribution.shape[0]):
    for lynx in range(gold_distribution.shape[1]):
      p = gold_distribution[hare, lynx]
      q = pred_distribution[hare, lynx]
      result += p * np.log(p/q)
  return result

def get_aic_bic(k, population_data, predictions):
  """
  Compute AIC and BIC for the provided gold population data and the given
  set of predictions.
  For the likelihood, we assume that each sample in time (a hare and lynx population)
  is independently from a time-invariant joint probability distribution interpolated from
  predictions.
  Parameters
  - k. The number of optimizable free parameters of the model.
  - population_data (2 x m matrix) Rows are population types and columns are time points.
    This is the gold standard
  - predictions (2 x m matrix) The predictions
  """
  n = population_data.shape[1]
  pred_distribution = get_population_joint_distribution(predictions)
  ll_tot = 0
  # Log likelihood of entire datast is the sum of log likelihood of each time step,
  # assumed to be independent
  for t in range(n):
    hare = int(population_data[0,t])
    lynx = int(population_data[1,t])
    # This is the probability of observing this one time point.
    p = pred_distribution[hare, lynx]
    ll_tot += np.log(p)
  
  aic = 2 * k - 2 * ll_tot
  bic = np.log(n) * k - 2 * ll_tot
  return aic, bic

def from_video_to_data_matrix(video_data):
  """
  Convert a video matrix to a data matrix that we can do DMD on.
  Parameters:
  - video_data: 3D array with axes: [time][y][x]
  Return:
  - data matrix (n x m) Each column is the image frame at that time, but
  flattened out in row order. That is, there are n pixel values and m time points.
  """
  T = video_data.shape[0]
  NUM_ROW = video_data.shape[1]
  NUM_COL = video_data.shape[2]
  return video_data.reshape(T, (NUM_ROW * NUM_COL)).transpose()

def from_data_matrix_to_video(data_matrix):
  """
  The inverse of from_video_to_data_matrix
  """
  T = data_matrix.shape[1]
  NUM_ROW = int(np.sqrt(data_matrix.shape[0]))
  NUM_COL = NUM_ROW
  return data_matrix.transpose().reshape(T, NUM_ROW, NUM_COL)

def score_prediction_video(predictions, gold):
  """
  Get the average mean squared error from each pixel.
  Parameters:
  - predictions n x m matrix, where there are n time
  """
  return np.mean(np.power(predictions - gold, 2))
