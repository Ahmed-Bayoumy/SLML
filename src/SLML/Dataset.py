
import numpy as np
from dataclasses import dataclass
import copy
from ._common import FileMissingError, ExceptionError, norm_t
import os 
import json

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

