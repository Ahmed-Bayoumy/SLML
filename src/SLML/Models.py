# ------------------------------------------------------------------------------------#
#  Statistical Learning Models ibrary - SLML                                          #
#                                                                                     #
#  Author: Ahmed H. Bayoumy                                                           #
#  email: ahmed.bayoumy@mail.mcgill.ca                                                #
#                                                                                     #
#  This program is free software: you can redistribute it and/or modify it under the  #
#  terms of the GNU Lesser General Public License as published by the Free Software   #
#  Foundation, either version 3 of the License, or (at your option) any later         #
#  version.                                                                           #
#                                                                                     #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY    #
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A    #
#  PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.   #
#                                                                                     #
#  You should have received a copy of the GNU Lesser General Public License along     #
#  with this program. If not, see <http://www.gnu.org/licenses/>.                     #
#                                                                                     #
#  You can find information on SLML        at                                         #
#  https://github.com/Ahmed-Bayoumy/SLML                                              #
# ------------------------------------------------------------------------------------#


import os
from typing import Callable, Dict, Any
import copy
from sklearn import linear_model, gaussian_process
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
from scipy.spatial.distance import cdist
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture as GMM

import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pyDOE2 import lhs

from scipy.spatial.distance import squareform, cdist, pdist
from matplotlib import pyplot as plt
from scipy.optimize import minimize, rosen, rosen_der
import copy
from ._common import MODEL_TYPE
from SLML.Dataset import DataSet
from ._modelFactory import modelFactory

@dataclass
class MOE(ABC):
  Exps: Dict[str, modelFactory] = None
  gmm: GMM = None

  def find_clusters(self, X, n_clusters, rseed=2):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    
    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)
        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])
        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return centers, labels
  
  def find_GMMs(self, nc, X, rstate=42, cov_type='full'):
    self.gmm = GMM(n_components=nc, covariance_type=cov_type, random_state=rstate).fit(X)
    labels = self.gmm.predict(X)
    probs = self.gmm.predict_proba(X)

    return labels, probs
  
  def plot_kmeans(self, labels, centers, X):

    # plot the input data
    fig, ax = plt.subplots(1)
    ax.scatter(X[:, 0], X[:, 1], c=labels,
                s=50, cmap='viridis')
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)

    # plot the representation of the KMeans model
    radii = [cdist(X[labels == i], [center]).max()
             for i, center in enumerate(centers)]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))
    
    plt.show()

  def draw_ellipse(self, position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
  def plot_gmm(self, X, labels, label=True, ax=None):
      ax = ax or plt.gca()
      # ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
      if label:
          ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
      else:
          ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
      ax.axis('equal')
      
      w_factor = 0.2 / self.gmm.weights_.max()
      for pos, covar, w in zip(self.gmm.means_, self.gmm.covariances_, self.gmm.weights_):
          self.draw_ellipse(pos, covar, alpha=w * w_factor, ax=ax)
      
      plt.show()

@dataclass
class RBF(modelFactory):
  extmodel: Callable = None
  # Initialize the RBF hyperparameters dict
  def __init__(self, name: str = "Exp_RBF_",x: np.ndarray = None, y: np.ndarray = None, xfile: str=None, yfile: str=None, options: Dict = None, HP: Dict = None):
    # Set initial values for theta and p
    super().__init__(name=name, x=x, y=y, xfile=xfile,
                     yfile=yfile, options=options, HP=HP)
    self.model_type = MODEL_TYPE.RBF
    if self.HP is None or not isinstance(self.HP, dict):
      self.HP = {}
      self.HP['gamma'] = {"lb": 0.1, "ub": 1., "value": 1.99}  # 0 < theta
      self.HP["rbf_func"] = {"var_type": "D_S1", "var_sets": {"S1": ['multiquadratic', 'Gaussian', 'inverse_multiquadratic', 'absolute', 'linear', 'cubic', 'thin_plate']}, "value": "cubic", "lb": 0, "ub": 6}

    if 'gamma' not in self.HP:
      self.HP['gamma'] = {"lb": 0.1, "ub": 1., "value": 1.99}  # 0 < theta
    
    if 'rbf_func' not in self.HP:
      self.HP["rbf_func"] = {"var_type": "D_S1", "var_sets": {"S1": ['multiquadratic', 'Gaussian', 'inverse_multiquadratic', 'absolute', 'linear', 'cubic', 'thin_plate']}, "value": "multiquadratic", "lb": 0, "ub": 6}

  def _multiquadric(self, r: DataSet) -> np.ndarray:
    return np.sqrt(np.power(r.points, 2) + self.HP['gamma']['value'] ** 2)

  # Inverse Multiquadratic
  def _inverse_multiquadric(self, r: DataSet) -> np.ndarray:
    return 1.0 / np.sqrt(np.power(r.points, 2) + self.HP['gamma']['value'] ** 2)

  # Absolute value
  def _absolute_value(self, r: DataSet) -> np.ndarray:
    return np.abs(r.points)

  # Standard Gaussian
  def _gaussian(self, r: DataSet) -> np.ndarray:
    return np.exp(-(self.HP['gamma']['value'] * r.points) ** 2)

  # Linear
  def _linear(self, r: DataSet) -> np.ndarray:
    return r.points

  # Cubic
  def _cubic(self, r: DataSet) -> np.ndarray:
    return (np.power(r.points, 3))

  # Thin Plate
  def _thin_plate(self, r: DataSet) -> np.ndarray:
    return np.power(r.points, 2) * np.log(np.abs(r.points))

  def _evaluate_kernel(self, r: DataSet) -> np.ndarray:
    rbf_dict = {
        "multiquadratic": self._multiquadric,
        "inverse_multiquadratic": self._inverse_multiquadric,
        "absolute": self._absolute_value,
        "Gaussian": self._gaussian,
        "linear": self._linear,
        "cubic": self._cubic,
        "thin_plate": self._thin_plate
    }
    return rbf_dict[self.HP["rbf_func"]["value"]](r)

  def _evaluate_radial_distance(self, a, b=None):
    # 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulczynski1', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'.
    if b is not None:
      return cdist(a, b, 'euclidean')
    else:
      return squareform(pdist(a, 'euclidean'))

  def calc_distance(self):
    pass

  def train(self, xt=None, yt=None):
    if xt is None and yt is None:
      self.scale.fit(self.xt.points)
      xstd = self.scale.transform(self.xt.points)
      y = self.yt.points
    else:
      self.scale.fit(xt)
      xstd = self.scale.transform(xt)
      y = yt

    r: DataSet = DataSet(name="r", nrows=1, ncols=1)

    r.points = self._evaluate_radial_distance(xstd)
    N = self._evaluate_kernel(r)
    self.weights = np.linalg.solve(N, y)
    self.is_trained = True
    self._compute_perf_metrics()
    self._display_acc_metrics()

  def predict(self, x, standardize: bool = False):
    if not self.is_trained:
      raise RuntimeError(f'The model {self.name} has not been trained yet! Please fit the model before using it for output predictions!')
    if standardize:
      self.scale.fit(x)
      xstd = self.scale.transform(x)
    else:
      xstd = x
    r: DataSet = DataSet(name="r", nrows=1, ncols=1)
    r.points = self._evaluate_radial_distance(
        self.xt.points, xstd)
    N = self._evaluate_kernel(r)
    return np.matmul(N.T, self.weights)

@dataclass
class LS(modelFactory):
  """ Least square """
  extmodel: Callable = linear_model.LinearRegression
  def __init__(self, name: str = "Exp_LS_",x: np.ndarray = None, y: np.ndarray = None, xfile: str=None, yfile: str=None, options: Dict = None, HP: Dict = None):
    # Set initial values for theta and p
    super().__init__(name=name, x=x, y=y, xfile=xfile,
                     yfile=yfile, options=options, HP=HP)
    self.model_type = MODEL_TYPE.LS
    if HP is None or not isinstance(self.HP, dict):
      self.HP = {}
      self.HP['type'] = {"var_type": "D_S1", "var_sets": {"S1": ['linear', 'polynomial', 'ridge', 'lasso', 'elastic']}, "value": "polynomial", "lb": 0, "ub": 4}
      self.HP['degree'] = {"var_type": "I", "lb": 1., "ub": 30., "value": 2.}  # 0 < theta
      self.HP["alpha"] = {"var_type": "R", "lb": 0.05, "ub": 0.5, "value": 1.}  # 0 < theta

    if 'type' not in self.HP:
      self.HP['type'] = {"var_type": "D_S1", "var_sets": {"S1": ['linear', 'polynomial', 'ridge', 'lasso', 'elastic']}, "value": "polynomial", "lb": 0, "ub": 4}
    if 'degree' not in self.HP:
      self.HP['degree'] = {"var_type": "I", "lb": 1., "ub": 30., "value": 2}  # 0 < theta
    if 'alpha' not in self.HP:
      self.HP['alpha'] = {"var_type": "R", "lb": 0.05, "ub": 0.5, "value": 1.}  # 0 < theta

  def train(self, xt: np.ndarray=None, yt:np.ndarray=None):
    if xt is None or yt is None:
      self.scale.fit(self.xt.points)
      xstd = self.scale.transform(self.xt.points)
      y = self.yt.points
    else:
      self.scale.fit(xt)
      xstd = self.scale.transform(xt)
      y = yt
    if self.HP["type"]["value"].lower() == 'polynomial':
      poly = PolynomialFeatures(degree=self.HP["degree"]["value"], include_bias=False)
      poly_features = poly.fit_transform(xstd)
      self.extmodel = linear_model.LinearRegression()
      self.extmodel.fit(poly_features, y)
    elif self.HP["type"]["value"].lower() == 'linear':
      self.extmodel = linear_model.LinearRegression()
      self.extmodel.fit(xstd, y)
    elif self.HP["type"]["value"].lower() == 'ridge':
      self.extmodel = linear_model.Ridge(alpha=self.HP["alpha"]["value"])
      self.extmodel.fit(xstd, y)
    elif self.HP["type"]["value"].lower() == 'lasso':
      self.extmodel = linear_model.Lasso(alpha=self.HP["alpha"]["value"])
      self.extmodel.fit(xstd, y)
    elif self.HP["type"]["value"].lower() == 'elastic':
      self.extmodel = linear_model.ElasticNet(alpha=self.HP["alpha"]["value"])
      self.extmodel.fit(xstd, y)
    else:
      raise IOError("Could not recognize the least square model name.")
    
    self.is_trained = True
    self._compute_perf_metrics()
    self._display_acc_metrics()

  def predict(self, x, standardize: bool = False):
    if standardize:
      self.scale.fit(x)
      xstd = self.scale.transform(x)
    else:
      xstd = x
    if self.HP["type"]["value"].lower() == "polynomial":
      poly = PolynomialFeatures(degree=self.HP["degree"]["value"], include_bias=False)
      if xstd.shape[0]>1:
        return self.extmodel.predict(poly.fit_transform(xstd))
      elif (xstd.shape[1]<=1):
        return self.extmodel.predict(poly.fit_transform(xstd.reshape(-1,1)))
      else:
        return self.extmodel.predict(poly.fit_transform(xstd.reshape(1,-1)))
    else:
      return self.extmodel.predict(xstd).reshape(-1,1)
  
  def predict_derivatives(self, x, kx):
    # Initialization
    self.scale.fit(x)
    xstd = self.scale.transform(x)
    n_eval, n_features_x = xstd.shape
    y = np.ones((n_eval, self.yt.ncols)) * self.extmodel.coef_[:, kx]
    return y

  def calc_distance(self):
    pass
      
@dataclass
class Kriging(modelFactory):
  r_inv: DataSet = None
  extmodel: Callable = gaussian_process
  # Initialize the Kriging hyperparameters dict
  def __init__(self, name: str = "Exp_kriging_",x: np.ndarray = None, y: np.ndarray = None, xfile: str=None, yfile: str=None, options: Dict = None, HP: Dict = None):
    # Set initial values for theta and p
    super().__init__(name=name, x=x, y=y, xfile=xfile,
                     yfile=yfile, options=options, HP=HP)
    self.model_type = MODEL_TYPE.KRIGING
    if self.HP is None or not isinstance(self.HP, dict):
      self.HP = {}
      self.HP['theta'] = {"lb": 0.01, "ub": 10., "value": 0.5}  # 0 < theta
      self.HP['p'] = {"lb": 0.1, "ub": 1., "value": 1.99}  # 0 < theta
      self.HP["scf_func"] = 'standard'
      self.HP["beta"] = {"value": 0.5}

    if 'theta' not in self.HP:
      self.HP['theta'] = {"lb": 0.01, "ub": 10., "value": 0.5}  # 0 < theta
    if 'p' not in self.HP:
      self.HP['p'] = {"lb": 0.1, "ub": 1., "value": 1.99}  # 0 < theta
    if 'beta' not in self.HP:
      self.HP['beta'] = {"value": 0.5} 

    if 'scf_func' not in self.HP:
      self.HP["scf_func"] = 'standard'

  def _compute_b(self, x:np.ndarray = None, y: np.ndarray=None) -> np.ndarray:
    if x is not None and y is not None:
      Xtemp = x
      Ytemp = y
    else:
      Xtemp = self.xt.points
      Ytemp = self.yt.points

    # Create a matrix of ones for calculating beta
    o = np.ones((Xtemp.shape[0], 1))
    beta = np.linalg.multi_dot(
        [o.T, self.r_inv.points, o, o.T, self.r_inv.points, Ytemp])
    return beta

  def _scf_compute(self, dist):
    if self.HP["scf_func"] == 'standard':
      try:
        r = np.exp(-1 * self.HP['theta']["value"] * dist ** self.HP['p']["value"])
        return r
      except:
        raise RuntimeError("Could not compute SCF!")
    else:
      raise ValueError("Not a currently coded SCF")

  def calc_distance(self, a, b=None):
    if b is not None:
      return cdist(a, b, 'minkowski', p=2)
    else:
      return squareform(pdist(a, 'minkowski', p=2))

  def _compute_r_inv(self, a, b=None) -> DataSet:
    dist = self.calc_distance(a, b)
    r: DataSet = DataSet(name="r_inv", nrows=1, ncols=1)
    r.points = self._scf_compute(dist)  # Calc SCF and return R inv
    r.points = np.linalg.inv(r.points)
    return r

  def _maximum_likelihood_estimator(self, x) -> np.ndarray:
    self.HP['theta']["value"] = x[0]
    self.HP['p']["value"] = x[1]
    self.cross_validate()
    y = self.yv.points
    x = self.xv.points

    self.scale.fit(x)
    x = self.scale.transform(x)
    self.scale.fit(y)
    y = self.scale.transform(y)

    self.r_inv: DataSet = DataSet(name="r_inv", nrows=1, ncols=1)
    self.r_inv = self._compute_r_inv(x)
    self.HP['r_inv'] = self.r_inv.points
    self.HP['beta']["value"] = self._compute_b(x, y)
    n = x.shape[0]
    y_b: np.ndarray = y - \
        np.matmul(np.ones((y.shape[0], 1)), self.HP['beta']["value"])
    sigma_sq: np.ndarray = (1 / n) * np.matmul(np.matmul(y_b.T,
                                                         self.r_inv.points), y_b)
    # TODO: use our coded cholesky
    mle = n * np.log(np.linalg.det(sigma_sq)) + \
        np.log(np.linalg.det(np.linalg.inv(self.r_inv.points)))
    if not isinstance(mle, complex):
      return [mle.tolist(), [0]]
    else:
      if mle.real == -np.inf:
        return [-100000000., [0.]]
      elif mle.real == np.inf:
        return [100000000., [0.]]
      return [mle.real, [0.]]

  def train(self, xt=None, yt=None):
    if xt is None and yt is None:
      self.scale.fit(self.xt.points)
      xstd = self.scale.transform(self.xt.points)
      y = self.yt.points
    else:
      self.scale.fit(xt)
      xstd = self.scale.transform(xt)
      y = yt
    self.r_inv = self._compute_r_inv(xstd)
    self.HP['r_inv'] = self.r_inv.points
    self.HP['beta']["value"] = self._compute_b(xstd, y)
    self.is_trained = True
    self._compute_perf_metrics()
    self._display_acc_metrics()

  def predict(self, x, standardize: bool = False):
    if not self.is_trained:
      raise RuntimeError(f'The model {self.name} has not been trained yet! Please fit the model before using it for output predictions!')
    if standardize:
      self.scale.fit(x)
      xstd = self.scale.transform(x)
    else:
      xstd = x
    r = self._scf_compute(self.calc_distance(self.xt.points, xstd)).T
    y_b = self.yt.points - \
        np.matmul(np.ones((self.yt.points.shape[0], 1)), self.HP['beta']['value'])
    return self.HP['beta']['value'] + np.linalg.multi_dot([r, self.r_inv.points, y_b])

def main():
  return

if __name__ == "__main__":
  main()
# IN PROGRESS: develop MOE
# IN PROGRESS: Link to BO
