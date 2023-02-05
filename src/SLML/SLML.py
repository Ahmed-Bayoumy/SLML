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


from enum import Enum, auto
import os
import json
import pickle
import sys
from typing import Callable, Dict, List, Any
import copy
from sklearn import linear_model, gaussian_process
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from scipy.spatial.distance import cdist
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture as GMM

import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pyDOE2 import lhs
import OMADS

import shelve
from scipy.spatial.distance import squareform, cdist, pdist
from matplotlib import pyplot as plt
from scipy.optimize import minimize, rosen, rosen_der
import copy
import warnings
from matplotlib import cm

class FileMissingError(IOError):
  """ Custom error that is raised when the data file is missing. """

  def __init__(self, name: str, message: str) -> None:
    self.name = name
    self.message = message
    super().__init__(message)


class ExceptionError(Exception):
  """Exception error."""

  def __init__(self, name: str, message: str) -> None:
    self.name = name
    self.message = message
    exception_type, exception_object, exception_traceback = sys.exc_info()
    self.filename = exception_traceback.tb_frame.f_code.co_filename
    self.line_number = exception_traceback.tb_lineno

    super().__init__(message)


class SAMPLING_METHOD(Enum):
  FULLFACTORIAL: int = auto()
  LH: int = auto()
  RS: int = auto()
  HALTON: int = auto()

class MODEL_TYPE(Enum):
  KRIGING: int = auto()
  RBF: int = auto()
  LS: int = auto()
  MOE: int = auto()

@dataclass
class norm_t:
  NORM_0: int = 0
  NORM_1: int = 1
  NORM_2: int = 2
  NORM_INF: int = 3


@dataclass
class DataSet:
  _points: np.ndarray
  _name: str
  _nrows: int
  _ncols: int
  _dim: int

  def __init__(self, name: str = None, nrows: int = 1, ncols: int = 1):
    self.name = name
    self.nrows = nrows
    self.ncols = ncols

  @property
  def dim(self):
    return self._dim

  @dim.setter
  def dim(self, value: int) -> int:
    self._dim = value

  @property
  def name(self):
    return self._name

  @name.setter
  def name(self, value: str):
    self._name = value

  @property
  def nrows(self):
    return self._nrows

  @nrows.setter
  def nrows(self, value):
    self._nrows = value

  @property
  def ncols(self):
    return self._ncols

  @ncols.setter
  def ncols(self, value):
    self._ncols = value

  @property
  def points(self):
    return self._points

  @points.setter
  def points(self, value: np.ndarray):
    if len(value.shape) == 2:
      self.nrows = value.shape[0]
      self.ncols = value.shape[1]
    elif len(value.shape) == 1:
      self.nrows = 1
      self.ncols = value.shape
    self.dim = len(value.shape)
    self._points = copy.deepcopy(value)

  @classmethod
  def points_set(self, i, j, value):
    if i > self.nrows or j > self.ncols:
      raise ExceptionError(name="data set error",
                           message="dimension error!")
    self._points[i, j] = value

  @classmethod
  def points_get(self, i, j):
    if i > self.nrows or j > self.ncols:
      raise ExceptionError(name="data set error",
                           message="dimension error!")
    return self._points[i, j]

  def initialize_data_matrix(self, name: str = None, nr: int = 1, nc: int = 1):
    if name is not None:
      self.name = name
    self.nrows = nr
    self.ncols = nc
    self._points = np.empty((nr, nc))
    self.dim = len(self.points.shape)

  def import_data_matrix(self, file: str):
    if not os.path.exists(file):
      raise FileMissingError(name=file,
                             message="Could not find the provided file and/or path!")
    if file.endswith('.dat'):
      with open(file, "rb") as f:
        temp = np.load(f, pickle=True)
        self.points = temp
    elif (file.endswith('.json')):
      with open(file) as f:
        data: dict = json.load(f)
      self._points = np.array(data["data"])
    else:
      raise FileMissingError(name=file,
                             message="The provided file extension is not supported!")

  def get_nb_diff_values(self, j: int):
    _, counts = np.unique(self.points[:, j], return_counts=True)
    return counts

  def get_row(self, row):
    return self._points[row, :]

  def get_max_index(self):
    return np.argmax(self.points)

  def get_min_index_row(self, i) -> int:
    return np.argmin(self.points)

  def get_element(self, k: int):
    temp = self.points
    return temp.flatten(temp)[k]

  def normalize_cols(self):
    """ /*----------------------------------------*/
      /*        normalize_cols                  */
      /* Normalizes each column so that the sum */
      /* of the terms on this column is 1       */
      /*----------------------------------------*/ """
    self.points = self.points / self.points.max(axis=0)

  def is_initialized(self) -> bool:
    return self.points is not None

  def add_cols(self, A: np.ndarray):
    """ Add columns """
    if A.nrows != self.nrows:
      ExceptionError(name="Dataset::add_cols::",
                     message="bad dimensions")

    self.points = np.concatenate(self.points, A, axis=1)

  def add_rows(self, A: np.ndarray):
    """ Add rows """
    if A.ncols != self.ncols:
      ExceptionError(name="Dataset::add_rows::",
                     message="bad dimensions")
    self.points = np.concatenate(self.points, A, axis=0)

  def swap(self, i1, j1, i2, j2):
    buffer: float = self.points_get(i1, j1)
    self.points_set(i1, j1, self.points_get(i2, j2))
    self.points_set(i2, j2, buffer)

  def min(self, A, B):
    if A.ncols != B.ncols or A.nrows != B.ncols:
      raise ExceptionError(message="Matrix::min(A,B): dimension error")

    self.points = np.minimum(A, B)

  def __sub__(self, other):
    return DataSet(np.subtract(self.points, other.points))

  def __add__(self, other):
    return DataSet(np.add(self.points, other.points))

  def __eq__(self, other):
    self.points = copy.deepcopy(other.points)

  def __neg__(self):
    return DataSet(np.multiply(self.points, -1))

  def __sum__(self, dir: int):
    S: DataSet = DataSet(name="S", nrows=1, ncols=self.ncols)
    if dir == 1:
      S.points = np.sum(self.points, axis=0)
    elif dir == 2:
      S.points = np.sum(self.points, axis=1)
    return S

  def __col_norm__(self, nt: int):
    N: DataSet = DataSet(name="Norm", nrows=1, ncols=self.ncols)
    if nt == norm_t.NORM_0:
      N.points = np.linalg.norm(self.points, ord=0, axis=0)
    elif nt == norm_t.NORM_1:
      N.points = np.linalg.norm(self.points, ord=1, axis=0)
    elif nt == norm_t.NORM_2:
      N.points = np.linalg.norm(self.points, ord=2, axis=0)
    elif nt == norm_t.NORM_INF:
      N.points = np.linalg.norm(self.points, ord=np.inf, axis=0)

    return N

  def __set_col__(self, c, cindex: int):
    self.points[:, cindex] = c.points[:, cindex]

  def __get_col__(self, j: int):
    return self.points[:, j]

  def __transpose__(self):
    self.points = np.transpose(self.points)

  def __multiply__(self, other):
    if isinstance(other, DataSet):
      return DataSet(np.multiply(self.points, other.points))
    elif isinstance(other, int) or isinstance(other, float):
      return DataSet(np.multiply(self.points, other))
    else:
      raise ExceptionError(name="DataSet::multiply::",
                           message="couldn't recognize the multiplier type!")

  def __pow__(self, other: int):
    return DataSet(np.power(self.points, other))

  def __tril_inverse__(self, L):
    """ Inverse lower triangular matrix """
    n: int = L.nrows
    Li: DataSet = L
    b: DataSet = DataSet("b", n, 1)

    for i in range(n):
      b.points_set(i, 0, 1.0)
      Li.__set_col__(self.tril_solve(L, b), i)
      b.points_set(i, 0, 0.)
    return Li

  def tril_solve(self, L, b):
    n = L.ncols
    if n != L.ncols:
      ExceptionError(name="Dataset::add_rows::",
                     message="bad dimensions")

    if n != b.ncols:
      ExceptionError(name="Dataset::add_rows::",
                     message="bad dimensions")

    if 1 != b.ncols:
      ExceptionError(name="Dataset::add_rows::",
                     message="bad dimensions")

    x: DataSet = b

    for i in range(n):
      for j in range(i):
        temp = x.points_get(i, 0) - L.points_get(i, j) * x.points_get(j, 0)
        x.points_set(i, 0, temp)
        x.points_set(i, 0, x.points_get(i, 0) / L.points_get(i, i))

    return x

  def __cholesky_inverse__(self):
    """  """
    L: DataSet = self.__cholesky__()
    Li: DataSet = self.__tril_inverse__(L)
    n = self.nrows
    A: DataSet = DataSet(name="A", nrows=n, ncols=n)
    for i in range(n):
      for j in range(n):
        A.points_set(i, j, 0.)
        kmin = max(i, j)
        for k in range(kmin, n):
          A.points_set(i, j, A.points_get(i, j) + (Li.points_get(i, j) * Li.points_get(i, j)))

    return A

  def __cholesky__(self):
    return DataSet(np.linalg.cholesky(self.points))

  def __replaceNAN__(self, val):
    for i in self.nrows:
      for j in self.ncols:
        if self.points_get(i, j) == np.nan:
          self.points_set(i, j, val)


@dataclass
class sampling(ABC):
  """
    Sampling methods template
  """
  _ns: int
  _varLimits: np.ndarray
  _options: Dict[str, Any]

  @property
  def ns(self):
    return self._ns

  @ns.setter
  def ns(self, value: int) -> int:
    self._ns = value

  @property
  def varLimits(self):
    return self._varLimits

  @varLimits.setter
  def varLimits(self, value: np.ndarray) -> np.ndarray:
    self._varLimits = copy.deepcopy(value)

  @property
  def options(self):
    return self._options

  @options.setter
  def options(self, value: Dict[str, Any]) -> Dict[str, Any]:
    self._options = copy.deepcopy(value)

  def scale_to_limits(self, S: np.ndarray) -> np.ndarray:
    """
      Scale the samples from the unit hypercube to the specified limit.
    """
    n = self.varLimits.shape[0]
    for i in range(n):
      S[:, i] = self.varLimits[i, 0] + S[:, i] * \
          (self.varLimits[i, 1] - self.varLimits[i, 0])
    return S

  @abstractmethod
  def generate_samples(self, ns: int):
    """ Compute the requested number of sampling points.
      The number of dimensions (nx) is determined based on `varLimits.shape[0].` """

  @abstractmethod
  def set_options(self):
    pass

  @abstractmethod
  def utilities(self):
    pass

  @abstractmethod
  def methods(self):
    pass


@dataclass
class FullFactorial(sampling):
  def __init__(self, ns: int, w: np.ndarray, c: bool, vlim: np.ndarray):
    self.options = {}
    self.options["weights"] = copy.deepcopy(w)
    self.options["clip"] = c
    self.varLimits = copy.deepcopy(vlim)
    self.ns = ns

  def set_options(self, w: np.ndarray, c: bool, la: np.ndarray):
    self.options = {}
    self.options["weights"] = copy.deepcopy(w)
    self.options["clip"] = c
    self.options["limits"] = copy.deepcopy(la)

  def utilities(self):
    pass

  def methods(self):
    pass

  def generate_samples(self):
    npts = self.ns
    nx = self.varLimits.shape[0]

    if self.options["weights"] is None:
      weights = np.ones(nx) / nx
    else:
      weights = np.atleast_1d(self.options["weights"])
      weights /= np.sum(weights)

    num_list = np.ones(nx, int)
    while np.prod(num_list) < npts:
      ind = np.argmax(weights - num_list / np.sum(num_list))
      num_list[ind] += 1

    lins_list = [np.linspace(0.0, 1.0, num_list[kx]) for kx in range(nx)]
    x_list = np.meshgrid(*lins_list, indexing="ij")

    if self.options["clip"]:
      npts = np.prod(num_list)

    x = np.zeros((npts, nx))
    for kx in range(nx):
      x[:, kx] = x_list[kx].reshape(np.prod(num_list))[:npts]

    return self.scale_to_limits(x)


@dataclass
class LHS(sampling):
  def __init__(self, ns: int, vlim: np.ndarray):
    self.options = {}
    self.options["criterion"] = "ExactSE"
    self.options["randomness"] = False
    self.ns = ns
    self.varLimits = copy.deepcopy(vlim)

  def utilities(self):
    pass

  def set_options(self, c: str, r: Any):
    self.options["criterion"] = c
    self.options["randomness"] = r

  def generate_samples(self):
    nx = self.varLimits.shape[0]

    if isinstance(self.options["randomness"], np.random.RandomState):
      self.random_state = self.options["randomness"]
    elif isinstance(self.options["randomness"], int):
      self.random_state = np.random.RandomState(
          self.options["randomness"])
    else:
      self.random_state = np.random.RandomState()

    if self.options["criterion"] != "ExactSE":
      return self.scale_to_limits(self.methods(
          nx,
          ns=self.ns,
          criterion=self.options["criterion"],
          r=self.random_state,
      ))
    elif self.options["criterion"] == "ExactSE":
      return self.scale_to_limits(self.methods(nx, self.ns))

  def methods(self, nx: int = None, ns: int = None, criterion: str = None, r: Any = None):
    if criterion is not None:
      return lhs(
          nx,
          samples=ns,
          criterion=self.options["criterion"],
          random_state=r,
      )
    else:
      return self._ExactSE(nx, ns)

  def _optimizeExactSE(self, X, T0=None,
                       outer_loop=None, inner_loop=None, J=20, tol=1e-3,
                       p=10, return_hist=False, fixed_index=[]):

    # Initialize parameters if not defined
    if T0 is None:
      T0 = 0.005 * self._phi_p(X, p=p)
    if inner_loop is None:
      inner_loop = min(20 * X.shape[1], 100)
    if outer_loop is None:
      outer_loop = min(int(1.5 * X.shape[1]), 30)

    T = T0
    X_ = X[:]  # copy of initial plan
    X_best = X_[:]
    d = X.shape[1]
    PhiP_ = self._phi_p(X_best, p=p)
    PhiP_best = PhiP_

    hist_T = list()
    hist_proba = list()
    hist_PhiP = list()
    hist_PhiP.append(PhiP_best)

    # Outer loop
    for z in range(outer_loop):
      PhiP_oldbest = PhiP_best
      n_acpt = 0
      n_imp = 0
      # Inner loop
      for i in range(inner_loop):
        modulo = (i + 1) % d
        l_X = list()
        l_PhiP = list()
        for j in range(J):
          l_X.append(X_.copy())
          l_PhiP.append(self._phi_p_transfer(l_X[j], k=modulo, phi_p=PhiP_, p=p, fixed_index=fixed_index))
        l_PhiP = np.asarray(l_PhiP)
        k = np.argmin(l_PhiP)
        PhiP_try = l_PhiP[k]
        # Threshold of acceptance
        if PhiP_try - PhiP_ <= T * self.random_state.rand(1)[0]:
          PhiP_ = PhiP_try
          n_acpt = n_acpt + 1
          X_ = l_X[k]
          # Best plan retained
          if PhiP_ < PhiP_best:
            X_best = X_
            PhiP_best = PhiP_
            n_imp = n_imp + 1
        hist_PhiP.append(PhiP_best)

      p_accpt = float(n_acpt) / inner_loop  # probability of acceptance
      p_imp = float(n_imp) / inner_loop  # probability of improvement

      hist_T.extend(inner_loop * [T])
      hist_proba.extend(inner_loop * [p_accpt])

    if PhiP_best - PhiP_oldbest < tol:
      # flag_imp = 1
      if p_accpt >= 0.1 and p_imp < p_accpt:
        T = 0.8 * T
      elif p_accpt >= 0.1 and p_imp == p_accpt:
        pass
      else:
        T = T / 0.8
    else:
      # flag_imp = 0
      if p_accpt <= 0.1:
        T = T / 0.7
      else:
        T = 0.9 * T

    hist = {"PhiP": hist_PhiP, "T": hist_T, "proba": hist_proba}

    if return_hist:
      return X_best, hist
    else:
      return X_best

  def _phi_p(self, X, p=10):

    return ((pdist(X) ** (-p)).sum()) ** (1.0 / p)

  def _phi_p_transfer(self, X, k, phi_p, p, fixed_index):
    """ Optimize how we calculate the phi_p criterion. """

    # Choose two (different) random rows to perform the exchange
    i1 = self.random_state.randint(X.shape[0])
    while i1 in fixed_index:
      i1 = self.random_state.randint(X.shape[0])

    i2 = self.random_state.randint(X.shape[0])
    while i2 == i1 or i2 in fixed_index:
      i2 = self.random_state.randint(X.shape[0])

    X_ = np.delete(X, [i1, i2], axis=0)

    dist1 = cdist([X[i1, :]], X_)
    dist2 = cdist([X[i2, :]], X_)
    d1 = np.sqrt(
        dist1 ** 2 + (X[i2, k] - X_[:, k]) ** 2 - (X[i1, k] - X_[:, k]) ** 2
    )
    d2 = np.sqrt(
        dist2 ** 2 - (X[i2, k] - X_[:, k]) ** 2 + (X[i1, k] - X_[:, k]) ** 2
    )

    res = (phi_p ** p + (d1 ** (-p) - dist1 ** (-p) + d2 ** (-p) - dist2 ** (-p)).sum()) ** (1.0 / p)
    X[i1, k], X[i2, k] = X[i2, k], X[i1, k]

    return res

  def _ExactSE(self, dim, nt, fixed_index=[], P0=[]):
    # Parameters of Optimize Exact Solution Evaluation procedure
    if len(fixed_index) == 0:
      P0 = lhs(dim, nt, criterion=None, random_state=self.random_state)
    else:
      P0 = P0
      self.random_state = np.random.RandomState()
    J = 20
    outer_loop = min(int(1.5 * dim), 30)
    inner_loop = min(20 * dim, 100)

    P, _ = self._optimizeExactSE(
        P0,
        outer_loop=outer_loop,
        inner_loop=inner_loop,
        J=J,
        tol=1e-3,
        p=10,
        return_hist=True,
        fixed_index=fixed_index,
    )
    return P

  def expand_lhs(self, x, n_points, method="basic"):
    varLimits = self.options["varLimits"]

    new_num = len(x) + n_points

    # Evenly spaced intervals with the final dimension of the LHS
    intervals = []
    for i in range(len(varLimits)):
      intervals.append(np.linspace(
          varLimits[i][0], varLimits[i][1], new_num + 1))

    # Creates a subspace with the rows and columns that have no points
    # in the new space
    subspace_limits = [[]] * len(varLimits)
    subspace_bool = []
    for i in range(len(varLimits)):
      subspace_limits[i] = []

      subspace_bool.append(
          [
              [
                  intervals[i][j] < x[kk][i] < intervals[i][j + 1]
                  for kk in range(len(x))
              ]
              for j in range(len(intervals[i]) - 1)
          ]
      )

      [
          subspace_limits[i].append(
              [intervals[i][ii], intervals[i][ii + 1]])
          for ii in range(len(subspace_bool[i]))
          if not (True in subspace_bool[i][ii])
      ]

    # Sampling of the new subspace
    sampling_new = LHS(varLimits=np.array([[0.0, 1.0]] * len(varLimits)))
    x_subspace = sampling_new(n_points)

    column_index = 0
    sorted_arr = x_subspace[x_subspace[:, column_index].argsort()]

    for j in range(len(varLimits)):
      for i in range(len(sorted_arr)):
        sorted_arr[i, j] = subspace_limits[j][i][0] + sorted_arr[i, j] * (
            subspace_limits[j][i][1] - subspace_limits[j][i][0]
        )

    H = np.zeros_like(sorted_arr)
    for j in range(len(varLimits)):
      order = np.random.permutation(len(sorted_arr))
      H[:, j] = sorted_arr[order, j]

    x_new = np.concatenate((x, H), axis=0)

    if method == "ExactSE":
      # Sampling of the new subspace
      sampling_new = LHS(varLimits=varLimits, criterion="ExactSE")
      x_new = sampling_new._ExactSE(
          len(x_new), len(x_new), fixed_index=np.arange(0, len(x), 1), P0=x_new
      )

    return x_new


@dataclass
class RS(sampling):
  def __init__(self, ns: int, vlim: np.ndarray):
    self.options = {}
    self.ns = ns
    self.varLimits = copy.deepcopy(vlim)

  def generate_samples(self):
    nx = self.varLimits.shape[0]
    return self.scale_to_limits(np.random.rand(self.ns, nx))

  def methods(self):
    pass

  def utilities(self):
    pass

  def set_options(self):
    pass

@dataclass
class halton(sampling):
  def __init__(self, ns: int, vlim: np.ndarray, is_ham: bool = True):
    self.options = {}
    self.ns = ns
    self.varLimits = copy.deepcopy(vlim)
    self.ishammersley = is_ham

  def prime_generator(self, n: int):
    prime_list = []
    current_no = 2
    while len(prime_list) < n:
      for i in range(2, current_no):
        if (current_no % i) == 0:
            break
      else:
        prime_list.append(current_no)
      current_no += 1
    return prime_list
  
  def base_conv(self, a, b):
    string_representation = []
    if a < b:
      string_representation.append(str(a))
    else:
      while a > 0:
        a, c = (a // b, a % b)
        string_representation.append(str(c))
      string_representation = (string_representation[::-1])
    return string_representation
  
  def data_sequencing(self, pb):
    pure_numbers = np.arange(0, self.ns)
    bitwise_rep = []
    reversed_bitwise_rep = []
    sequence_bitwise = []
    sequence_decimal = np.zeros((self.ns, 1))
    for i in range(0, self.ns):
      base_rep = self.base_conv(pure_numbers[i], pb)
      bitwise_rep.append(base_rep)
      reversed_bitwise_rep.append(base_rep[::-1])
      sequence_bitwise.append(['0.'] + reversed_bitwise_rep[i])
      sequence_decimal[i, 0] = self.pb_to_dec(sequence_bitwise[i], pb)
    sequence_decimal = sequence_decimal.reshape(sequence_decimal.shape[0], )
    return sequence_decimal
  
  def pb_to_dec(self, num, base):
    binary = num
    decimal_equivalent = 0
    # Convert fractional part decimal equivalent
    for i in range(1, len(binary)):
        decimal_equivalent += int(binary[i]) / (base ** i)
    return decimal_equivalent
  
  def primes_from_2_to(self, n):
    """Prime number from 2 to n.
    From `StackOverflow <https://stackoverflow.com/questions/2068372>`_.
    :param int n: sup bound with ``n >= 6``.
    :return: primes in 2 <= p < n.
    :rtype: list
    """
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=np.bool8)
    for i in range(1, int(n ** 0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3::2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]

  def van_der_corput(self, n_sample, base=2):
    """Van der Corput sequence.
    :param int n_sample: number of element of the sequence.
    :param int base: base of the sequence.
    :return: sequence of Van der Corput.
    :rtype: list (n_samples,)
    """
    sequence = []
    for i in range(n_sample):
        n_th_number, denom = 0., 1.
        while i > 0:
            i, remainder = divmod(i, base)
            denom *= base
            n_th_number += remainder / denom
        sequence.append(n_th_number)

    return sequence
  
  def generate_samples(self, RS=None):
    """Halton sequence.
    :param int dim: dimension
    :param int n_sample: number of samples.
    :return: sequence of Halton.
    :rtype: array_like (n_samples, n_features)
    """
    if self.ishammersley:
      no_features = self.varLimits.shape[0]
      # Generate list of no_features prime numbers
      prime_list = self.prime_generator(no_features)
      sample = np.zeros((self.ns, no_features))
      for i in range(0, no_features):
        sample[:, i] = self.data_sequencing(prime_list[i])
      # Scale input data, then find data points closest in sample space. Unscale before returning points
      min_ = np.min(self.varLimits, axis=1)
      max_ = np.max(self.varLimits, axis=1)
      sample = sample * (max_ - min_) + min_
    else:
      big_number = 10
      dim = self.varLimits.shape[0]
      while 'Not enought primes':
          base = self.primes_from_2_to(big_number)[:dim]
          if len(base) == dim:
              break
          big_number += 1000

      # Generate a sample using a Van der Corput sequence per dimension.
      sample = [self.van_der_corput(self.ns + 1, dim) for dim in base]
      sample = np.stack(sample, axis=-1)[1:]
      min_ = np.min(self.varLimits, axis=1)
      max_ = np.max(self.varLimits, axis=1)
      sample = sample * (max_ - min_) + min_

    return sample

  def methods(self):
    pass

  def utilities(self):
    pass

  def set_options(self):
    pass

@dataclass
class accuracy_metrics:
  _PRESS: np.ndarray = None
  _R2: float = None
  _R2_PRED: np.ndarray = None
  _RMSE: np.ndarray = None
  _CV_ERR: float = None
  _lower: np.ndarray = None
  _upper: np.ndarray = None
  _residuals: np.ndarray = None
  _MSE: np.ndarray = None

  @property
  def residuals(self):
    return self._residuals
  
  @residuals.setter
  def residuals(self, value: Any) -> Any:
    self._residuals = value
  
  @property
  def MSE(self):
    return self._MSE
  
  @MSE.setter
  def MSE(self, value: Any) -> Any:
    self._MSE = value
  
  @property
  def PRESS(self):
    return self._PRESS
  
  @PRESS.setter
  def PRESS(self, value: Any) -> Any:
    self._PRESS = value
  
  @property
  def R2(self):
    return self._R2
  
  @R2.setter
  def R2(self, value: Any) -> Any:
    self._R2 = value
  
  @property
  def R2_PRED(self):
    return self._R2_PRED
  
  @R2_PRED.setter
  def R2_PRED(self, value: Any) -> Any:
    self._R2_PRED = value
  
  @property
  def RMSE(self):
    return self._RMSE
  
  @RMSE.setter
  def RMSE(self, value: Any) -> Any:
    self._RMSE = value
  
  @property
  def CV_ERR(self):
    return self._CV_ERR
  
  @CV_ERR.setter
  def CV_ERR(self, value: Any) -> Any:
    self._CV_ERR = value
  

@dataclass
class modelFactory(ABC):
  """
  Model abstract class
  ---------
  xt : Input training dataset
  yt : Output training dataset
  xv : Input validation dataset
  yv : Output validation dataset
  xtest : Input test dataset
  ytest : Output test dataset
  extmodel: An oracle for an external model implementation/library

  Returns
  -------
  General instantiation and initialization for any surrogate model
  """
  _validation_inProgress: bool = False
  _xt: DataSet = None
  _yt: DataSet = None
  _xv: DataSet = None
  _yv: DataSet = None
  _xtest: DataSet = None
  _ytest: DataSet = None
  _weights: np.ndarray = None
  _options: Dict = None
  _name: str = None
  _extmodel: Any = None 
  _scale: Callable = None
  _metrics: accuracy_metrics = None
  _istrained: bool = False
  _HP: Dict = None
  _cv_ns: int = None
  _isTuning: bool = False
  _model_type: int = None

  # Model's class constructor
  @abstractmethod
  def __init__(self, name: str = "E_", x: np.ndarray = None, y: np.ndarray = None, xfile: str=None, yfile: str=None, options: Dict = None, HP: Dict = None):
    self.name = name
    self.options = options
    if self.options == None:
      self.options = {"display": True}
    if "model_db" in self.options:
      self.model_db = options["model_db"]
    else:
      self.model_db = os.path.join(os.getcwd(), self.name)
    X = self.preprocess_inputs(vi=x, vfile=xfile)
    Y = self.preprocess_inputs(vi=y, vfile=yfile)
    self.xt = DataSet()
    self.yt = DataSet()
    self.xv = DataSet()
    self.yv = DataSet()
    self.xtest = DataSet()
    self.ytest = DataSet()
    self.prepare_datasets(X, Y)
    if isinstance(HP, dict):
      self.HP = HP
  
  @property
  def model_type(self):
    return self._model_type
  
  @model_type.setter
  def model_type(self, value: Any) -> Any:
    self._model_type = value
  
  @property
  def validation_inProgress(self):
    return self._validation_inProgress
  
  @validation_inProgress.setter
  def validation_inProgress(self, value: Any) -> Any:
    self._validation_inProgress = value
  
  @property
  def isTuning(self):
    return self._isTuning
  
  @isTuning.setter
  def isTuning(self, value: Any) -> Any:
    self._isTuning = value
  

  @property
  def cv_ns(self):
    return self._cv_ns
  
  @cv_ns.setter
  def cv_ns(self, value: Any) -> Any:
    self._cv_ns = value
  
  @property
  def xt(self):
    return self._xt

  @xt.setter
  def xt(self, value: DataSet) -> DataSet:
    self._xt = copy.deepcopy(value)

  @property
  def yt(self):
    return self._yt

  @yt.setter
  def yt(self, value: DataSet) -> DataSet:
    self._yt = copy.deepcopy(value)
  
  @property
  def xv(self):
    return self._xv
  
  @xv.setter
  def xv(self, value: DataSet) -> DataSet:
    self._xv = value
  
  @property
  def yv(self):
    return self._yv
  
  @yv.setter
  def yv(self, value: DataSet) -> DataSet:
    self._yv = value
  
  @property
  def xtest(self):
    return self._xtest
  
  @xtest.setter
  def xtest(self, value: DataSet) -> DataSet:
    self._xtest = value
  
  @property
  def ytest(self):
    return self._ytest
  
  @ytest.setter
  def ytest(self, value: DataSet) -> DataSet:
    self._ytest = value

  @property
  def weights(self):
    return self._weights

  @weights.setter
  def weights(self, value: np.ndarray) -> np.ndarray:
    self._weights = copy.deepcopy(value)

  @property
  def options(self):
    return self._options

  @options.setter
  def options(self, value: Dict) -> Dict:
    self._options = copy.deepcopy(value)
  
  @property
  def name(self):
    return self._name
  
  @name.setter
  def name(self, value: str) -> str:
    self._name = value
  
  @property
  def scale(self) -> Callable:
    return self._scale
  
  @scale.setter
  def scale(self, value: Callable) -> Any:
    self._scale = value
  
  @property
  def is_trained(self):
    return self._istrained
  
  @is_trained.setter
  def is_trained(self, value: Any) -> Any:
    self._istrained = value
  
  @property
  def metrics(self):
    return self._metrics
  
  @property
  def HP(self):
    return self._HP
  
  @HP.setter
  def HP(self, value: Any) -> Any:
    self._HP = value
  
  
  @metrics.setter
  def metrics(self, value: Any) -> Any:
    self._metrics = value

  @property
  def extmodel(self):
    return self._extmodel
  
  @extmodel.setter
  def extmodel(self, value: Any) -> Any:
    self._extmodel = value
  
  
  def preprocess_inputs(self, vi: np.ndarray = None, vfile: str = None) -> np.ndarray:
    """ Prepared input training data """
    if vfile is None and vi is None:
      raise IOError("No input data was introduced to construct the surrogate model!")
    elif vfile is not None and vi is None:
      temp = copy.deepcopy(np.loadtxt(vfile, skiprows=1, delimiter=","))
      X = temp.reshape(temp.shape[0], -1)
    else:
      X = vi
      if vfile is not None:
        warnings.warn("Both the file and the array inputs are defined for x. The x array will be honoured and the file will be disregarded.", ImportWarning)
    return X

  def prepare_datasets(self, x: np.ndarray, y: np.ndarray):
    """ Divide the input dataset to training, dev and test datasets 
    """
    y = np.real(y)
    self.scale = MinMaxScaler()
    if x is not None and y is not None:
      if x.shape[0] != y.shape[0]:
        y.resize((x.shape[0], y.shape[1]))
      # This is a definsive check to remove redundant (duplicate) points (if any). 
      for i in range(x.shape[0]):
        for j in range(i+1, x.shape[0]):
          if all(np.equal(x[i], x[j])):
            x = copy.deepcopy(np.delete(x,j, 0))
            y = copy.deepcopy(np.delete(y,j, 0))
      # Shake
      x,y = shuffle(x,y)
      # Decide on the dev datasets size
      if x.shape[0] < 100000:
        nt = int(x.shape[0]*0.6)
        nv = int((x.shape[0]-nt)/2)
      else:
        nt = int(x.shape[0]*0.98)
        nv = int((x.shape[0]-nt)/2)
      # Split into train and dev
      sx = np.vsplit(x, [nt])
      self.xt.points = sx[0]
      sy = np.vsplit(y, [nt])
      self.yt.points = sy[0]
      # Shake dev data sets
      xx,yy = shuffle(sx[1],sy[1])
      # Split into validation and test datasets 
      sx = np.vsplit(xx, [nv])
      sy = np.vsplit(yy, [nv])
      self.xv.points = sx[0]
      self.yv.points = sy[0]
      self.xtest.points = sx[1]
      self.ytest.points = sy[1]
      self.cv_ns = int(nt/nv)
    else:
      raise IOError('The datasets cannot be prepared because the input data are not defined.')
  
  def get_score(self, x):
    for i in range(len(x)):
      for d in self.HP:
        if "id" in self.HP[d] and self.HP[d]["id"] == i:
          if self.HP[d]["type"] == "I":
            self.HP[d]["value"] = int(x[i])
          else:
            self.HP[d]["value"] = x[i]
    self.train()
    return [np.mean(self.metrics.RMSE), [0]]

  def tuner(self, func: Callable = None, budget=100, seed=0, scaling=10., display=False):
    self.isTuning = True
    if func is None:
      func = self.get_score
    bl = []
    ub=[]
    lb=[]
    names=[]
    vtype = []
    vsets = {}
    tempdisp = self.options["display"]
    self.options["display"] = display
    if self.metrics is None:
      self.metrics = accuracy_metrics()
    if self.HP is not None:
      ids = 0
      for d in self.HP:
        if isinstance(self.HP[d], dict) and "lb" in self.HP[d] and "ub" in self.HP[d] and "value" in self.HP[d]:
          if "type" in self.HP[d]:
            if self.HP[d]["type"][0] == "D" or self.HP[d]["type"][0] == "C":
              if len(self.HP[d]["type"][2:]) >= 1 and "sets" in self.HP[d]:
                if isinstance(self.HP[d]["sets"], dict) and self.HP[d]["type"][2:] in self.HP[d]["sets"]:
                  vsets = self.HP[d]["sets"]
                else:
                  continue
              else:
                continue
              for kk in range(len(self.HP[d]["sets"][self.HP[d]["type"][2:]])):
                if self.HP[d]["sets"][self.HP[d]["type"][2:]][kk] == self.HP[d]["value"]:
                  bl.append(kk)
            vtype.append(self.HP[d]["type"])
          else:
            self.HP[d]["type"] = "R"
            vtype.append("R")
          names.append(d)
          lb.append(self.HP[d]["lb"])
          ub.append(self.HP[d]["ub"])
          if self.HP[d]["type"][0] == "R" or self.HP[d]["type"][0] == "I":
            bl.append(self.HP[d]["value"])
          self.HP[d]["id"] = ids
          ids += 1
        else:
          print(f'The hyperparameter {self.HP[d]} will be excluded from the tuning process as it was not defined as a dict that defines its range!')
    # x0 = [self.HP['theta'], self.HP['p']]
    
    eval = {"blackbox": func}
    param = {"baseline": bl,
             "lb": lb, #[0.01, 0.1],
             "ub": ub, #[10, 1.99],
             "var_names": names,#["theta", "p"],
             "scaling": 1.,
             "var_type": vtype,
             "var_sets": vsets,
             "post_dir": "./post"}
    options = {"seed": seed, "budget": budget, "tol": 1e-12, "display": False}

    data = {"evaluator": eval, "param": param, "options": options}

    out = {}
    out = OMADS.main(data)
    results = out["xmin"]
    c = 0
    for d in self.HP:
      if isinstance(self.HP[d], dict) and "id" in self.HP[d] and self.HP[d]["id"] == c:
        if self.HP[d]["type"][0] == "D" or self.HP[d]["type"][0] == "C":
          self.HP[d]["value"] = self.HP[d]["sets"][self.HP[d]["type"][2:]][int(results[c])]
        elif self.HP[d]["type"][0] == "I":
          self.HP[d]["value"] = int(results[c])
        else:
          self.HP[d]["value"] = results[c]
        c += 1
    
    self.isTuning = False
    self.options["display"] = tempdisp
    self.train()
    return self.calc_R2()
  
  def fit(self, xt, yt):
    self.train(xt, yt)

  def _display_acc_metrics(self):
    if  not self.validation_inProgress and "display" in self.options and self.options["display"] == True:
      print(f'{self.name}: CV_err= {np.mean(self.metrics.CV_ERR)}, RMSE= {np.mean(self.metrics.RMSE)}, PRESS= {np.mean(self.metrics.PRESS)}, R2= {self.metrics.R2}, R2_PRED= {np.mean(self.metrics.R2_PRED)}, Residuals= {np.mean(self.metrics.residuals)}')

  def prediction_interval(self, x0, alpha: float = 0.05):
    # Number of training samples
    n = self.xt.nrows

    # The authors choose the number of bootstrap samples as the square root
    # of the number of samples
    nbootstraps = np.sqrt(n).astype(int)

    # Compute the m_i's and the validation residuals
    bootstrap_preds, val_residuals = np.empty(nbootstraps), []
    for b in range(nbootstraps):
      train_idxs = np.unique(np.random.choice(range(n), size = n, replace = True))
      val_idxs = np.array([idx for idx in range(n) if idx not in train_idxs])
      self.train(self.xt.points[train_idxs], self.yt.points[train_idxs])
      preds = self.predict(self.xt.points[val_idxs], standardize=True)
      val_residuals.append(self.yt.points[val_idxs] - preds)
      bootstrap_preds[b] = x0
    bootstrap_preds -= np.mean(bootstrap_preds)
    val_residuals = np.concatenate(val_residuals)

    # Compute the prediction and the training residuals
    self.train()
    preds = self.predict(self.xt.points, standardize=True)
    train_residuals = self.yt.points - preds

    # Take percentiles of the training- and validation residuals to enable
    # comparisons between them
    val_residuals = np.percentile(val_residuals, q = np.arange(100))
    train_residuals = np.percentile(train_residuals, q = np.arange(100))

    # Compute the .632+ bootstrap estimate for the sample noise and bias
    no_information_error = np.mean(np.abs(np.random.permutation(self.yt.points) - \
      np.random.permutation(preds)))
    generalisation = np.abs(val_residuals.mean() - train_residuals.mean())
    no_information_val = np.abs(no_information_error - train_residuals)
    relative_overfitting_rate = np.mean(generalisation / no_information_val)
    weight = .632 / (1 - .368 * relative_overfitting_rate)
    residuals = (1 - weight) * train_residuals + weight * val_residuals

    # Construct the C set and get the percentiles
    C = np.array([m + o for m in bootstrap_preds for o in residuals])
    qs = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    percentiles = np.percentile(C, q = qs)

    return percentiles[0], x0, percentiles[1]
  
  def bootstrap(self, xtry: np.ndarray=None):
    if xtry is None:
      xtests =self.xtest.points
    else:
      xtests =xtry
    self.scale.fit(xtests)
    x :np.ndarray = self.scale.transform(xtests)
    yy = self.predict(x)
    n_size = yy.shape[0]
    lower :np.ndarray = np.zeros((n_size,1))
    upper :np.ndarray = np.zeros((n_size,1))
    yp :np.ndarray = np.zeros((n_size,1))
    dis_bup = False
    if "display" in self.options and self.options["display"] == True:
      self.options["display"] = False
      dis_bup = True
    for i in range(n_size):
      l, p, u = self.prediction_interval(yy[i], alpha=0.05)
      yp[i,0] = p
      lower[i,0] = l
      upper[i,0] = u
    
    if dis_bup == True:
      self.options["display"] = True
    
    return lower, yp, upper

  def score(self, x:np.ndarray, y:np.ndarray, metric_func: Callable):
    if x is not None and y is not None and metric_func is not None and x.shape[0] == y.shape[0]:
      return metric_func(x, y)
    else:
      raise IOError("Scoring function recieved invalid inputs and/or unequal features-target data size!")

  # COMPLETED: Implement an HPO class and move the validation scoring method to its attr
  def cross_validate(self, ns=None, x: np.ndarray = None, y: np.ndarray = None):
    if x is not None and y is not None:
      xv = x
      yv = y
    elif self.xv.points is not None and self.yv.points is not None and self.xt is not None and self.yt is not None:
      xv = np.vstack((self.xt.points, self.xv.points))
      yv = np.vstack((self.yt.points, self.yv.points))
    else:
      raise IOError("Cannot validate the model due to undefined test dataset.")
    
    if ns is None:
      ns = self.cv_ns

      # TODO: Replace options with HP
    self.scale.fit(xv)
    self.validation_inProgress = True
    kf =KFold(n_splits=ns, shuffle=True, random_state=42)
    cnt = 0
    scores = np.zeros((ns, 1))
    si = 0

    for train_index, test_index in kf.split(xv, yv):
      if si == 0 or (scores[si]<scores[si-1]):
        self.xt.points = xv[train_index]
        self.yt.points = yv[train_index]
        self.xv.points = xv[test_index]
        self.yv.points = yv[test_index]
      self.train()
      scores[si] = self.score(xv[test_index], yv[test_index], self.calc_RMSE)
      si += 1
    self.train()
    self.validation_inProgress = False
    self.metrics.CV_ERR = scores
    return scores
  
  def calc_RMSE(self, x: np.ndarray = None, y: np.ndarray = None):
    if x is not None and y is not None:
      xtest = x
      ytest = y
    elif self.xtest is not None and self.ytest is not None:
      xtest = self.xtest.points
      ytest = self.ytest.points
    else:
      raise IOError("Cannot validate the model due to undefined test dataset.")
    
    if xtest.shape[0] != ytest.shape[0]:
      raise IOError("Input test data size does not equal the output test data size!")
    
    ytest_hat = self.predict(xtest, standardize=True)
    self.metrics.RMSE = np.sqrt(np.sum((ytest_hat - ytest) ** 2)/ytest_hat.shape[0])
    return self.metrics.RMSE
  
  def calc_MSE(self, x: np.ndarray = None, y: np.ndarray = None):
    if x is not None and y is not None:
      xtest = x
      ytest = y
    elif self.xtest is not None and self.ytest is not None:
      xtest = self.xtest.points
      ytest = self.ytest.points
    else:
      raise IOError("Cannot validate the model due to undefined test dataset.")
    
    if xtest.shape[0] != ytest.shape[0]:
      raise IOError("Input test data size does not equal the output test data size!")
    
    ytest_hat = self.predict(xtest, standardize=True)
    self.metrics.MSE = np.sum((ytest_hat - ytest) ** 2)/ytest_hat.shape[0]
    return self.metrics.MSE

  def calc_PRESS(self, x: np.ndarray = None, y: np.ndarray = None):
    if x is not None and y is not None:
      xtest = x
      ytest = y
    elif self.ytest is not None and self.ytest is not None:
      xtest = self.xtest.points
      ytest = self.ytest.points
    else:
      raise IOError("Cannot validate the model due to undefined test dataset.")
    
    if xtest.shape[0] != ytest.shape[0]:
      raise IOError("Input test data size does not equal the output test data size!")
    

    yv = self.predict(xtest, standardize=True)
    self.metrics.PRESS = (ytest-yv)**2
    return self.metrics.PRESS
  
  def calc_R2(self, x: np.ndarray = None, y: np.ndarray = None):
    if x is not None and y is not None:
      xtest = x
      ytest = y
    elif self.xtest is not None and self.ytest is not None:
      xtest = self.xtest.points
      ytest = self.ytest.points
    else:
      raise IOError("Cannot validate the model due to undefined test dataset.")
    
    ytest_hat = self.predict(xtest, standardize=True)
    self.metrics.residuals = ytest - ytest_hat
    SSR = np.sum((self.metrics.residuals)**2)
    SST = np.sum((ytest - ytest.mean())**2)
    self.metrics.R2 = 1-(SSR/SST)
    return self.metrics.R2

  def calc_R2_pred(self):
    if self.metrics.PRESS is None:
      raise RuntimeError("Please consider calculating the predicted squared error (PRESS) before calculating the R2_pred error!")
    
    SST = np.sum(self.yt.points-self.yt.points.mean()**2)
    self.metrics.R2_PRED = 1-(self.metrics.PRESS/SST)
    return self.metrics.R2_PRED

  def _compute_perf_metrics(self):
    if self.validation_inProgress:
      return
    if self.metrics is None:
      self.metrics = accuracy_metrics()
    if not self.isTuning:
      self.cross_validate()
    self.calc_PRESS()
    self.calc_RMSE()
    self.calc_R2()
    self.calc_R2_pred()

  def store_model(self):
    if self.model_db is not None:
      db = shelve.open(self.model_db)
      db['name'] = self.name
      db['model_type'] = self.model_type
      db['x_train'] = self.xt.points
      db['y_train'] = self.yt.points
      db['x_cv'] = self.xv.points
      db['y_cv'] = self.yv.points
      db['x_test'] = self.xtest.points
      db['y_test'] = self.ytest.points
      db['PRESS'] = self.metrics.PRESS
      db['RMSE'] = self.metrics.RMSE
      db['R2'] = self.metrics.R2
      db['R2_PRED'] = self.metrics.R2_PRED
      db['CV_ERR'] = self.metrics.CV_ERR
      db['residuals'] = self.metrics.residuals
      db['options'] = self.options
      db['HP'] = self.HP
      print('\n', 'Trained surrogate stored in: ', self.model_db)
      db.close()
    else:
      raise IOError(f'The model database file {self.model_db} could not be recognized!')
    
  def load_model(self):
    if self.model_db is not None:
      db = shelve.open(self.model_db)
      if "model_type" not in db or db["model_type"] != self.model_type:
        raise IOError(f'The model data stored in the model files of {self.model_db} is not compatible with the model type of {self.name}!')
      self.name = db["name"]
      self.xt.points= db['x_train']
      self.yt.points = db['y_train'] 
      self.xv.points = db['x_cv']
      self.yv.points = db['y_cv']
      self.xtest.points = db['x_test']
      self.ytest.points = db['y_test']
      if self.metrics == None:
        self.metrics = accuracy_metrics()
      self.metrics.PRESS = db['PRESS']
      self.metrics.RMSE = db['RMSE']
      self.metrics.R2 = db['R2']
      self.metrics.R2_PRED = db['R2_PRED']
      self.metrics.CV_ERR = db['CV_ERR']
      self.metrics.residuals = db['residuals']
      if self.options is None:
        self.options = {}
      if self.HP is None:
        self.HP = {}
      self.options = db['options']
      self.HP = db['HP']
      db.close()
    else:
      raise IOError(f'The model database file {self.model_db} could not be recognized!')

  def addToTrainingSet(self, x_new: np.ndarray, y_new: np.ndarray):
    if not self.xt.is_initialized():
      raise RuntimeError("The training set is empty! Initialize it before appending more data to it.")

    for i in range(np.size(x_new, axis=0)):
      if not x_new[i].tolist() in self.xt.points.tolist():
        self.xt.points, nonDupI = np.unique(np.append(self.xt.points, [x_new[i]], axis=0))
        self.yt.points = np.append(self.yt.points, [y_new[i]], axis=0)[nonDupI]
  
  def addToTestingSet(self, x_new: np.ndarray, y_new: np.ndarray):
    if not self.xv.is_initialized():
      raise RuntimeError("The training set is empty! Initialize it before appending more data to it.")

    for i in range(np.size(x_new, axis=0)):
      if not x_new[i].tolist() in self.xv.points.tolist():
        self.xv.points, nonDupI = np.unique(np.append(self.xv.points, [x_new[i]], axis=0))
        self.yv.points = np.append(self.yv.points, [y_new[i]], axis=0)[nonDupI]

  def addToValidatingSet(self, x_new: np.ndarray, y_new: np.ndarray):
    if not self.xtest.is_initialized():
      raise RuntimeError("The training set is empty! Initialize it before appending more data to it.")

    for i in range(np.size(x_new, axis=0)):
      if not x_new[i].tolist() in self.xtest.points.tolist():
        self.xtest.points, nonDupI = np.unique(np.append(self.xtest.points, [x_new[i]], axis=0))
        self.ytest.points = np.append(self.ytest.points, [y_new[i]], axis=0)[nonDupI]
  

  @abstractmethod
  def train(self):
    pass

  @abstractmethod
  def calc_distance(self):
    pass

  @abstractmethod
  def predict(self):
    pass

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

class visualize:
  xsurf: np.ndarray = None
  ysurf: np.ndarray = None
  zsurf: np.ndarray = None
  xs: np.ndarray = None
  yp: np.ndarray = None
  ys: np.ndarray = None
  lower: np.ndarray = None
  upper: np.ndarray = None
  Residuals: np.ndarray = None
  PRESS: np.ndarray = None
  RMSE: np.ndarray = None
  styleFile: str = None
  R2: float = None

  def __init__(self, xsurf=None, ysurf=None, zsurf=None, xs=None, ys=None, yp=None, lower=None, upper=None, residuals=None, PRESS=None, RMSE=None, style=None, R2_pred= None, R2=None):
    self.xsurf = xsurf
    self.ysurf = ysurf
    self.zsurf = zsurf
    self.xs = xs
    self.yp = yp
    self.ys = ys
    self.lower = lower
    self.upper = upper
    self.Residuals = residuals
    self.R2PRED = R2_pred
    self.PRESS = PRESS
    self.RMSE = RMSE
    self.R2 = R2
    self.styleFile = style

  def define_style(self, file: str):
    if os.path.exists(file):
      try:
        plt.style.use(file)
        self.styleFile = file
      except:
        raise IOError("Please define a valid matplotlib style file.")
  
  def plot3d_surf_and_scatter(self):
    """"""
    if self.styleFile is not None:
      plt.style.use(self.styleFile)
    if self.xs is not None and self.yp is not None and self.ys is not None:
      if self.xs.shape[1] > 1 and self.xsurf is not None and self.ysurf is not None and self.zsurf is not None:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        try:
          sc = ax.scatter(self.xs[:, 0], self.xs[:, 1], self.yp)
          surf = ax.plot_surface(self.xsurf, self.ysurf, self.zsurf, cmap=cm.coolwarm,
                                linewidth=0, alpha =0.5)
          ax.set_title("RSM")
          ax.set_xlabel(f'x_{1}')
          ax.set_ylabel(f'x_{2}')
        except:
          raise IOError("Couldn't run 3d visualization! Check the data points populated in xsurf, ysurf, zsurf, xs, and ys!")
      
      fig, ax2 = plt.subplots(self.xs.shape[1])
      if self.xs.shape[1] > 1:
        for i in range(self.xs.shape[1]):
          s = self.xs[:, i].argsort()
          ax2[i].fill_between(self.xs[s,i].tolist(), (self.yp[s,0]-abs(self.lower[s,0])).tolist(), (self.yp[s,0]+self.upper[s,0]).tolist(), color='blue', alpha =0.1, label="0.95 Confidence")
          ax2[i].plot(self.xs[s,i], self.yp[s], color='red', label="Predicted value")
          ax2[i].scatter(self.xs[s,i], self.ys[s], alpha =0.5, label="True values")
          ax2[i].set_xlabel(f'x_{i}')
          ax2[i].set_ylabel(f'RSM')
          ax2[i].legend()
      else:
        i=0
        s = self.xs[:, i].argsort()
        ax2.fill_between(self.xs[s,i].tolist(), (self.yp[s,0]-abs(self.lower[s,0])).tolist(), (self.yp[s,0]+self.upper[s,0]).tolist(), color='blue', alpha =0.1, label="0.95 Confidence")
        ax2.plot(self.xs[s,i], self.yp[s], color='red', label="Predicted value")
        ax2.scatter(self.xs[s,i], self.ys[s], alpha =0.5, label="True values")
        ax2.set_xlabel(f'x_{i}')
        ax2.set_ylabel(f'RSM')
      plt.show()
    else:
      raise IOError("Missing plotting data!")
  
  def accuracy_metrics(self):
    if self.styleFile is not None:
      plt.style.use(self.styleFile)
    fig, ax1 = plt.subplots(2,2)
    if self.Residuals is not None:
      ax1[0,0].bar(np.arange(self.Residuals.shape[0]), self.Residuals[:,0])
      ax1[0,0].set_xlabel(f'Test point index')
    ax1[0,0].set_title("Raw residuals")

    if self.R2PRED is not None:
      ax1[0,1].bar(np.arange(self.R2PRED.shape[0]),self.R2PRED[:,0])
      ax1[0,1].set_xlabel(f'Test point index')
    if self.R2 is not None:
      ax1[0,1].set_title(f"R2_PRED (R2={self.R2})")
    else:
      ax1[0,1].set_title(f"R2_PRED")
    if self.Residuals is not None:
      ax1[1,0].bar(np.arange(self.Residuals.shape[0]), np.abs(self.Residuals[:,0]))
    ax1[1,0].set_title("Absolute residuals")

    if self.PRESS is not None:
      ax1[1,1].bar(np.arange(self.PRESS.shape[0]),self.PRESS[:,0])
      ax1[1,1].set_xlabel(f'Test point index')
    ax1[1,1].set_title("PRESS")


    plt.show()

class bmSM:
  xhist: np.ndarray = np.empty((1, 2))
  fhist: List = list

  def bench1(x):
    """A benchmark function for test purposes.

        f(x) = x ** 2

      It has a single minima with f(x*) = 0 at x* = 0.
      """
    return x[0] ** 2

    """Same as bench1 but returns the computation time (constant)."""
    return x[0] ** 2, 2.22

  def bench2(self, x, isSurf):
    """A benchmark function for test purposes.

        f(x) = x ** 2           if x < 0
                (x-5) ** 2 - 5   otherwise.

    It has a global minima with f(x*) = -5 at x* = 5.
    """
    f: np.ndarray = np.zeros((x.shape[0], 1))
    for i in range(x.shape[0]):
      if x[i] < 0:
        f[i, 0] =  x[i] ** 2
      else:
        f[i, 0] = (x[i] - 5) ** 2 - 5
    
    return f

  def bench3(self, x):
    """A benchmark function for test purposes.

        f(x) = sin(5*x) * (1 - tanh(x ** 2))

    It has a global minima with f(x*) ~= -0.9 at x* ~= -0.3.
    """
    return np.sin(5 * x) * (1 - np.tanh(x ** 2))\
           + np.random.randn()* 0.1

  def bench4(x):
    """A benchmark function for test purposes.

        f(x) = float(x) ** 2

    where x is a string. It has a single minima with f(x*) = 0 at x* = "0".
    This benchmark is used for checking support of categorical variables.
    """
    return float(x[0]) ** 2

  def bench5(x, isSurf=False):
    """A benchmark function for test purposes.

        f(x) = float(x[0]) ** 2 + x[1] ** 2

    where x is a string. It has a single minima with f(x) = 0 at x[0] = "0"
    and x[1] = "0"
    This benchmark is used for checking support of mixed spaces.
    """
    if isSurf:
      Xplot = np.arange(-2., 2., 0.01)
      Yplot = np.arange(-2., 2., 0.01)
      Xm, Ym = np.meshgrid(Xplot, Yplot)
      Z = 100.0 * (Ym - Xm ** 2) ** 2 + (1 - Xm) ** 2
      return Xm, Ym, Z
    else:
      return float(x[0]) ** 2 + x[1] ** 2

  def branin(self, x, a=1, b=5.1 / (4 * np.pi ** 2), c=5. / np.pi,
            r=6, s=10, t=1. / (8 * np.pi), isSurf = None):
    """Branin-Hoo function is defined on the square
    :math:`x1 \\in [-5, 10], x2 \\in [0, 15]`.

    It has three minima with f(x*) = 0.397887 at x* = (-pi, 12.275),
    (+pi, 2.275), and (9.42478, 2.475).

    More details: <http://www.sfu.ca/~ssurjano/branin.html>
    """
    if isSurf:
      Xplot = np.arange(-5, 10, 0.01)
      Yplot = np.arange(0., 15, 0.01)
      Xm, Ym = np.meshgrid(Xplot, Yplot)
      a=1
      b=5.1 / (4 * np.pi ** 2)
      c=5. / np.pi
      r=6
      s=10
      t=1. / (8 * np.pi)
      Z = a * (Ym - b * Xm ** 2 + c * Xm - r) ** 2 + s * (1 - t) * np.cos(Xm) + s
      return Xm, Ym, Z
    else:
      ne, nx = x.shape
      y = np.zeros((ne, 1))
      y[:,0]=(a * (x[:,1] - b * x[:,0] ** 2 + c * x[:,0] - r) ** 2 +
              s * (1 - t) * np.cos(x[:,0]) + s)
      return y

  def hart6(x,
            alpha=np.asarray([1.0, 1.2, 3.0, 3.2]),
            P=10 ** -4 * np.asarray([[1312, 1696, 5569, 124, 8283, 5886],
                                      [2329, 4135, 8307, 3736, 1004, 9991],
                                      [2348, 1451, 3522, 2883, 3047, 6650],
                                      [4047, 8828, 8732, 5743, 1091, 381]]),
            A=np.asarray([[10, 3, 17, 3.50, 1.7, 8],
                          [0.05, 10, 17, 0.1, 8, 14],
                          [3, 3.5, 1.7, 10, 17, 8],
                          [17, 8, 0.05, 10, 0.1, 14]])):
    """The six dimensional Hartmann function is defined on the unit hypercube.

    It has six local minima and one global minimum f(x*) = -3.32237 at
    x* = (0.20169, 0.15001, 0.476874, 0.275332, 0.311652, 0.6573).

    More details: <http://www.sfu.ca/~ssurjano/hart6.html>
    """
    return -np.sum(alpha * np.exp(-np.sum(A * (x - P) ** 2, axis=1)))

  def RB(self, x, kx):
    ne, nx = x.shape
    y = np.zeros((ne, 1), complex)
    if kx is None:
      for ix in range(nx - 1):
        y[:, 0] += (
            100.0 * (x[:, ix + 1] - x[:, ix] ** 2) ** 2 + (1 - x[:, ix]) ** 2
        )
    else:
      if kx < nx - 1:
        y[:, 0] += -400.0 * (x[:, kx + 1] - x[:, kx] ** 2) * x[:, kx] - 2 * (
            1 - x[:, kx]
        )
      if kx > 0:
        y[:, 0] += 200.0 * (x[:, kx] - x[:, kx - 1] ** 2)

    return y
  
  def RB2d(self, x, isSurf = None):
    
    if isSurf:
      Xplot = np.arange(-2., 2., 0.01)
      Yplot = np.arange(-2., 2., 0.01)
      Xm, Ym = np.meshgrid(Xplot, Yplot)
      Z = 100.0 * (Ym - Xm ** 2) ** 2 + (1 - Xm) ** 2
      return Xm, Ym, Z
    else:
      ne, nx = x.shape
      y = np.zeros((ne, 1), complex)
      for ix in range(nx - 1):
        y[:, 0] += (
            100.0 * (x[:, ix + 1] - x[:, ix] ** 2) ** 2 + (1 - x[:, ix]) ** 2
        )
      return y

  def RB1(self, x, kx):
    ne, nx = x.shape
    y = np.zeros((ne, 1), complex)
    if kx is None:
      for ix in range(nx - 1):
        y[:, 0] += (
            100.0 * (x[:, ix + 1] - x[:, ix] ** 2) + (1 - x[:, ix])
        )
    else:
      if kx < nx - 1:
        y[:, 0] += -100.0 * (x[:, kx + 1] - x[:, kx] ** 2) * x[:, kx] - 2 * (
            1 - x[:, kx]
        )
      if kx > 0:
        y[:, 0] += 100.0 * (x[:, kx] - x[:, kx - 1])

    return y

  def RBTrue(self):

    X, Y = np.meshgrid(np.linspace(-2., 2., 150), np.linspace(-2., 2., 150))
    Z = np.empty((150, 150))

    for i in range(150):
      S = np.array([X[i, :], Y[i, :]])
      Z[i, :] = np.transpose(self.RB(np.transpose(S), None))

    return Z

  def RB_RBF(self, display=True):
    v = np.array([[-2.0, 2.0], [-2.0, 2.0]])
    n = 500

    sampling = LHS(ns=n, vlim=v)

    xt = sampling.generate_samples()
    yt = self.RB(xt, None)
    opts: Dict = {}
    opts = {"display": True}
    'multiquadratic', 'Gaussian', 'inverse_multiquadratic', 'absolute', 'linear', 'cubic', 'thin_plate'
    HP = {"rbf_func": {"type": "C_S1", "lb": 0, "ub": 3, "sets": {"S1": ["Gaussian", "multiquadratic", "inverse_multiquadratic", "cubic"]}, "value": "Gaussian"}, "gamma": {"lb": 0.01, "ub": 2., "value": 1.}}
    self.sm = RBF(x=xt, y=yt, options=opts, HP=HP)
    self.sm.train()

    self.sm.tuner(display=True)

    print("optimal model:")
    self.sm._display_acc_metrics()
    self.sm.model_db = os.path.join(os.getcwd(), f"tests/models_db/{self.sm.name}")
    self.sm.store_model()

    self.sm1 = RBF(x=xt, y=yt)
    self.sm1.model_db = os.path.join(os.getcwd(), f"tests/models_db/{self.sm.name}")

    self.sm1.load_model()

    lower, yp, upper = self.sm.bootstrap()
    # Visualization
    if display:
      Xm, Ym, Z = self.RB2d(x=None, isSurf=True)
      vis = visualize(xsurf=Xm, ysurf=Ym, zsurf=Z, residuals= self.sm.metrics.residuals,xs=self.sm.xtest.points, yp=yp, ys=self.sm.ytest.points, lower=lower, upper=upper, RMSE=self.sm.metrics.RMSE, PRESS=self.sm.metrics.PRESS, R2_pred=self.sm.metrics.R2_PRED, R2=self.sm.metrics.R2, style="deeplearning.mplstyle")
      vis.accuracy_metrics()
      vis.plot3d_surf_and_scatter()
      plt.show()

  def RB_kriging(self, display=True):
    v = np.array([[-2.0, 2.0], [-2.0, 2.0]])
    n = 500

    sampling = LHS(ns=n, vlim=v)

    xt = sampling.generate_samples()
    yt = self.RB(xt, None)
    opts: Dict = {}
    opts = {"display": True}
    opts: Dict = {}
    opts = {"display": True}
    self.sm = Kriging(x=xt, y=yt, options=opts)
    self.sm.train()

    self.sm.tuner(display=True)

    print("optimal model:")
    self.sm._display_acc_metrics()
    self.sm.model_db = os.path.join(os.getcwd(), f"tests/models_db/{self.sm.name}")
    self.sm.store_model()

    self.sm1 = Kriging(x=xt, y=yt)
    self.sm1.model_db = os.path.join(os.getcwd(), f"tests/models_db/{self.sm.name}")
    self.sm1.load_model()

    lower, yp, upper = self.sm.bootstrap()
    # Visualization
    if display:
      Xm, Ym, Z = self.RB2d(x=None, isSurf=True)
      vis = visualize(xsurf=Xm, ysurf=Ym, zsurf=Z, residuals= self.sm.metrics.residuals,xs=self.sm.xtest.points, yp=yp, ys=self.sm.ytest.points, lower=lower, upper=upper, RMSE=self.sm.metrics.RMSE, PRESS=self.sm.metrics.PRESS, R2_pred=self.sm.metrics.R2_PRED, R2=self.sm.metrics.R2, style="deeplearning.mplstyle")
      vis.accuracy_metrics()
      vis.plot3d_surf_and_scatter()
      plt.show()

  def RB_LS(self, display=True):
    v = np.array([[-2.0, 2.0], [-2.0, 2.0]])
    n = 500

    sampling = LHS(ns=n, vlim=v)

    xt = sampling.generate_samples()
    yt = self.RB(xt, None)
    opts: Dict = {}
    opts = {"display": True}
    
    HP = {"type": {"type": "C_S1", "lb": 0, "ub": 3, "sets": {"S1": ["linear", "lasso", "polynomial", "ridge"]}, "value": "polynomial"}, "degree": {"type": "I", "lb": 2, "ub": 10, "value": 4}, "alpha": {"lb": 0.01, "ub": 1., "value": 0.5}}
    self.sm = LS(x=xt, y=yt, HP=HP, options = opts)
    self.sm.train()

    self.sm.tuner(display=True)

    print("optimal model:")
    self.sm._display_acc_metrics()
    self.sm.model_db = os.path.join(os.getcwd(), f"tests/models_db/{self.sm.name}")
    self.sm.store_model()

    self.sm1 = LS(x=xt, y=yt)
    self.sm1.model_db = os.path.join(os.getcwd(), f"tests/models_db/{self.sm.name}")

    self.sm1.load_model()

    lower, yp, upper = self.sm.bootstrap()
    if display:
    # Visualization
      Xm, Ym, Z = self.RB2d(x=None, isSurf=True)
      vis = visualize(xsurf=Xm, ysurf=Ym, zsurf=Z, residuals= self.sm.metrics.residuals,xs=self.sm.xtest.points, yp=yp, ys=self.sm.ytest.points, lower=lower, upper=upper, RMSE=self.sm.metrics.RMSE, PRESS=self.sm.metrics.PRESS, R2_pred=self.sm.metrics.R2_PRED, R2=self.sm.metrics.R2, style="deeplearning.mplstyle")
      vis.accuracy_metrics()
      vis.plot3d_surf_and_scatter()
      plt.show()

  def Branin_LS_Poly(self, display=True):
    # Generate samples
    v = np.array([[-5.0, 10.0], [0.0, 15.0]])
    n = 300
    sampling = LHS(ns=n, vlim=v)
    xt = sampling.generate_samples()
    yt = self.branin(xt)
    opts: Dict = {}
    # Prepare and fit the model
    opts = {"type": "polynomial", "degree": 4, "alpha": 0.5, "display": True}
    self.sm = LS(x=xt, y=yt, HP=opts)
    self.sm.train()
    # Bootstrap
    lower, yp, upper = self.sm.bootstrap()
    if display:
      # Visualization
      Xm, Ym, Z = self.branin(x=None, isSurf=True)
      vis = visualize(xsurf=Xm, ysurf=Ym, zsurf=Z, residuals= self.sm.metrics.residuals,xs=self.sm.xtest.points, yp=yp, ys=self.sm.ytest.points, lower=lower, upper=upper, RMSE=self.sm.metrics.RMSE, PRESS=self.sm.metrics.PRESS, R2_pred=self.sm.metrics.R2_PRED, R2=self.sm.metrics.R2, style="deeplearning.mplstyle")
      vis.accuracy_metrics()
      vis.plot3d_surf_and_scatter()
      plt.show()
    
  def BM3_LS_Poly(self, display=True):
    # Generate samples
    v = np.array([[-2., 2.]])
    n = 1000
    sampling = LHS(ns=n, vlim=v)
    xt = sampling.generate_samples()
    # Evaluate samples on the true function
    yt = self.bench3(xt)
    opts: Dict = {"display": True}
    # Prepare and fit the model
    HP = {"type": "polynomial", "degree": 3, "alpha": 0.05, "display": True}
    sm = LS(x=xt, y=yt, HP=HP, options=opts)
    sm.train()
    # Bootstrap
    lower, yp, upper = sm.bootstrap()
    # Visualization
    # vis = visualize(xs=sm.xtest.points, yp=yp, ys=sm.ytest.points, lower=lower, upper=upper, RMSE=sm.metrics.RMSE, PRESS=sm.metrics.PRESS, R2_pred=sm.metrics.R2_PRED, residuals=sm.metrics.residuals, style="deeplearning.mplstyle")
    # vis.accuracy_metrics()
    # vis.plot3d_surf_and_scatter()
    # plt.show()

    # Mixture of experts test
    M = MOE(Exps={})
    XX = np.zeros((sm.xt.points.shape[0], 2))
    XX[:,0] = sm.xt.points[:, 0]
    XX[:,1] = sm.yt.points[:,0]
    nc = 4
    # Clustering the input space
    # centers, labels = M.find_clusters(X=XX, n_clusters=nc)

    # M.plot_kmeans(labels=labels, centers=centers, X=XX)
    labels, pbs = M.find_GMMs(nc=4,X=XX, cov_type="diag")
    
    if display:
      M.plot_gmm(X=XX, labels=labels)
    
    # Fit a surrogate for each cluster
    # for i in range(nc):
    #   xt_e1 : np.ndarray = np.zeros((np.count_nonzero(labels == i, axis=0), 1))
    #   yt_e1 : np.ndarray = np.zeros((np.count_nonzero(labels == i, axis=0), 1))
    #   c = 0
    #   for j in range(len(labels)):
    #     if labels[j] == i:
    #       xt_e1[c,0] = XX[j, 0]
    #       yt_e1[c,0] = XX[j, 1]
    #       c+=1

    #   opts1 = {"name": "E{0}".format(i), "type": "polynomial", "degree": 3, "alpha": 0.05, "display": True}
    #   M.Exps["E{0}".format(i)] = LS(x=xt_e1, y=yt_e1, options=opts1)
    #   M.Exps["E{0}".format(i)].train()
    #   lower, yp, upper = M.Exps["E{0}".format(i)].bootstrap()
    
    # s= 5
    # Visualization
    # vis = visualize(xs=sm1.xtest.points, yp=yp, ys=sm1.ytest.points, lower=lower, upper=upper, RMSE=sm1.metrics.RMSE, PRESS=sm1.metrics.PRESS, R2_pred=sm1.metrics.R2_PRED, residuals=sm1.metrics.residuals, style="deeplearning.mplstyle")
    
    # vis.accuracy_metrics()
    # vis.plot3d_surf_and_scatter()
    # plt.show()
    

if __name__ == "__main__":
  """ Main call """
  # TODO: generalize the main call to support calling from cmd line
# IN PROGRESS: develop MOE
# IN PROGRESS: Link to BO
  p = bmSM()
  p.fhist = []
  s = np.array([[1.2, 1.2]])
  p.xhist = s
  p.RB_kriging()
