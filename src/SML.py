# ------------------------------------------------------------------------------------#
#  Simple Surrogate Models Library - RAF                                              #
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
#  You can find information on simple_mads at                                         #
#  https://github.com/Ahmed-Bayoumy/SML                                               #
# ------------------------------------------------------------------------------------#


from ctypes.wintypes import BOOL
from enum import Enum, auto
import os
import json
import pickle
from re import X
import sys
from tkinter.messagebox import NO
from typing import Callable, List, Dict, Any
import copy

import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pyDOE2 import lhs
import OMADS

import shelve
from scipy.spatial.distance import squareform, cdist, pdist
from scipy.optimize import minimize, rosen, rosen_der
from matplotlib import pyplot as plt
import OMADS


class FileMissingError(IOError):
	"""Custom error that is raised when the data file is missing."""

	def __init__(self, name: str, message: str) -> None:
		self.name = name
		self.message = message
		super().__init__(message)


class ExceptionError(Exception):
	"""Exception error."""

	def __init__(self, name: str, message: str) -> None:
		self.name = name
		self.message = message
		exception_type, \
			exception_object, \
			exception_traceback = sys.exc_info()
		self.filename = exception_traceback.tb_frame.f_code.co_filename
		self.line_number = exception_traceback.tb_lineno

		super().__init__(message)


class SAMPLING_METHOD(Enum):
	FULLFACTORIAL: int = auto()
	LH: int = auto()
	RS: int = auto()

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
			raise FileMissingError(
				name=file,
				message="Could not find the provided file and/or path!"
			)
		if file.endswith('.dat'):
			with open(file, "rb") as f:
				temp = np.load(f, pickle=True)
				self.points = temp
		elif (file.endswith('.json')):
			with open(file) as f:
				data: dict = json.load(f)
			self._points = np.array(data["data"])
		else:
			raise FileMissingError(
				name=file,
				message="The provided file extension is not supported!"
			)

	def get_nb_diff_values(self, j: int):
		s: List[float] = []
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
			temp = x.points_get(i, j) - L.points_get(i, j)*x.points_get(j, 0)
			x.points_set(i, 0, temp)
			for j in range(i):
				x.points_set(i, 0, x.points_get(i, 0)/L.points_get(i, i))

		return x

	def __cholesky_inverse__(self):
		L: DataSet = self.__cholesky__()
		Li: DataSet = self.__tril_inverse__(L)
		n = self.nrows



		A: DataSet = DataSet(name="A", nrows=n, ncols=n)
		for i in range(n):
			for j in range(n):
				A.points_set(i, j, 0.)
				kmin = max(i, j)
				for k in range(kmin, n):
					A.points_set(i, j, A.points_get(i, j) +
								 (Li.points_get(i, j)*Li.points_get(i, j)))

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

	def set_options(self, w: np.ndarray, c: bool, l: np.ndarray):
		self.options = {}
		self.options["weights"] = copy.deepcopy(w)
		self.options["clip"] = c
		self.options["limits"] = copy.deepcopy(l)

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
					l_PhiP.append(
						self._phi_p_transfer(
							l_X[j], k=modulo, phi_p=PhiP_, p=p, fixed_index=fixed_index
						)
					)

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
		"""
		Optimize how we calculate the phi_p criterion. 
		"""

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
			dist1 ** 2 + (X[i2, k] - X_[:, k]) ** 2 -
			(X[i1, k] - X_[:, k]) ** 2
		)
		d2 = np.sqrt(
			dist2 ** 2 - (X[i2, k] - X_[:, k]) ** 2 +
			(X[i1, k] - X_[:, k]) ** 2
		)

		res = (
			phi_p ** p + (d1 ** (-p) - dist1 ** (-p) +
						  d2 ** (-p) - dist2 ** (-p)).sum()
		) ** (1.0 / p)
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
class bmSamplingMethods:

	def test_random(self):
		v = np.array([[0.0, 4.0], [0.0, 3.0]])
		sampling = RS(ns=50, vlim=v)
		x = sampling.generate_samples()

		print(x.shape)

		plt.plot(x[:, 0], x[:, 1], "o")
		plt.xlabel("x")
		plt.ylabel("y")
		plt.show()

	def test_lhs(self):
		varLimits = np.array([[0.0, 4.0], [0.0, 3.0]])
		sampling = LHS(ns=50, vlim=varLimits)
		sampling.options["criterion"] = "ExactSE"

		x = sampling.generate_samples()

		print(x.shape)

		plt.plot(x[:, 0], x[:, 1], "o")
		plt.xlabel("x")
		plt.ylabel("y")
		plt.show()

	def test_full_factorial(self):
		varLimits = np.array([[0.0, 4.0], [0.0, 3.0]])
		sampling = FullFactorial(ns=50, w=np.array(
			[0.8, 0.2]), c=True, vlim=varLimits)
		x = sampling.generate_samples()
		print(x.shape)
		plt.plot(x[:, 0], x[:, 1], "o")
		plt.xlabel("x")
		plt.ylabel("y")
		plt.show()


@dataclass
class surrogateModel(ABC):
	_x: DataSet
	_y: DataSet
	_xt: DataSet
	_yt: DataSet
	_yp: DataSet
	_weights: np.ndarray
	_options: Dict

	@property
	def yp(self):
		return self._yp

	@yp.setter
	def yp(self, value: DataSet) -> DataSet:
		self._yp = copy.deepcopy(value)

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
	def weights(self):
		return self._weights

	@weights.setter
	def weights(self, value: np.ndarray) -> np.ndarray:
		self._weights = copy.deepcopy(value)

	@property
	def x(self):
		return self._x

	@x.setter
	def x(self, value: DataSet) -> DataSet:
		self._x = copy.deepcopy(value)

	@property
	def y(self):
		return self._y

	@y.setter
	def y(self, value: DataSet) -> DataSet:
		self._y = copy.deepcopy(value)

	@property
	def options(self):
		return self._options

	@options.setter
	def options(self, value: Dict) -> Dict:
		self._options = copy.deepcopy(value)

	@abstractmethod
	def train(self):
		pass

	@abstractmethod
	def build(self):
		pass

	@abstractmethod
	def calc_distance(self):
		pass

	@abstractmethod
	def predict(self):
		pass

	@abstractmethod
	def addToTrainingSet(self):
		pass


@dataclass
class RBF(surrogateModel):
	rbf_func: str
	epsilon: float

	def __init__(self, type: str, x: np.ndarray = None, y: np.ndarray = None, xfile='x_train.dat', yfile='y_train.dat', options: Dict = {}, rbf_func="Gaussian"):
		self.options = options
		self.weights = options["weights"]
		self.xt = DataSet(name="xt", nrows=1, ncols=1)
		self.yt = DataSet(name="yt", nrows=1, ncols=1)
		self.yp = DataSet(name="yp", nrows=1, ncols=1)
		self.rbf_func = rbf_func

		self.x = DataSet(name="x", nrows=1, ncols=1)
		if xfile is not None and x is None:
			temp = copy.deepcopy(np.loadtxt(xfile, skiprows=1, delimiter=","))
			self.xt.points = temp.reshape(temp.shape[0], -1)
		else:
			self.xt.points = x

		if type == "train":
			self.yt = DataSet(name="y", nrows=1, ncols=1)
			if yfile is not None and x is None:
				temp = copy.deepcopy(np.loadtxt(
					yfile, skiprows=1, delimiter=","))
				self.yt.points = temp.reshape(temp.shape[0], -1)
			else:
				self.yt.points = y

			self.epsilon = np.std(self.yt.points)

			self.train()

		else:
			self.predict()

	def _multiquadric(self, r: DataSet) -> np.ndarray:
		return np.sqrt(np.power(r.points, 2) + self.epsilon ** 2)

	# Inverse Multiquadratic
	def _inverse_multiquadric(self, r: DataSet) -> np.ndarray:
		return 1.0 / np.sqrt(np.power(r.points, 2) + self.epsilon ** 2)

	# Absolute value
	def _absolute_value(self, r: DataSet) -> np.ndarray:
		return np.abs(r.points)

	# Standard Gaussian
	def _gaussian(self, r: DataSet) -> np.ndarray:
		return np.exp(-(self.epsilon * r.points) ** 2)

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
		return rbf_dict[self.rbf_func](r)

	def _evaluate_radial_distance(self, a, b=None):
		if b is not None:
			return cdist(a, b, 'minkowski', p=2)
		else:
			return squareform(pdist(a, 'minkowski', p=2))

	def build(self):
		pass

	def calc_distance(self):
		pass

	def train(self):
		r: DataSet = DataSet(name="r", nrows=1, ncols=1)
		r.points = self._evaluate_radial_distance(self.xt.points)
		N = self._evaluate_kernel(r)
		self.weights = np.linalg.solve(N, self.yt.points)

	def predict(self):
		r: DataSet = DataSet(name="r", nrows=1, ncols=1)
		r.points = self._evaluate_radial_distance(
			self.xt.points, self.x.points)
		N = self._evaluate_kernel(r)
		self.yp.points = np.matmul(N.T, self.weights)
	
	def addToTrainingSet(self, x_new: np.ndarray, y_new: np.ndarray):
		if not self.xt.is_initialized():
			raise RuntimeError("The training set is empty! Initialize it before appending more data to it.")
		
		for i in range(np.size(x_new, axis=0)):
			if not x_new[i].tolist() in self.xt.points.tolist():
				self.xt.points = np.append(self.xt.points, [x_new[i]], axis=0)
				self.yt.points = np.append(self.yt.points, [y_new[i]], axis=0)




# Class to create or use a Kriging surrogate model
@dataclass
class Kriging(surrogateModel):
	scf_func: str
	epsilon: float
	r_inv: DataSet

	def build(self):
		pass

	# Kriging class constructor
	def __init__(self, type: str, x: np.ndarray = None, y: np.ndarray = None, xfile='x_train.dat', yfile='y_train.dat', options = {}, model_db='model.db', scf_func='standard'):
		self.options = options
		self.scf_func = scf_func
		self.xt = DataSet(name="xt", nrows=1, ncols=1)
		self.yt = DataSet(name="yt", nrows=1, ncols=1)
		self.yp = DataSet(name="yp", nrows=1, ncols=1)
		self.x = DataSet(name="x", nrows=1, ncols=1)
		if xfile is not None and x is None:
			self.xt.points = np.loadtxt(xfile, skiprows=1, delimiter=",")
			self.xt.points = self.x.points.reshape(self.x.points.shape[0], -1)
			self.scf_func = scf_func
			self.model_db = model_db
		else:
			self.xt.points = x

		if type == 'train':
			self.yt = DataSet(name="y", nrows=1, ncols=1)
			if yfile is not None and x is None:
				y = np.loadtxt(yfile, skiprows=1, delimiter=",")
				self.yt.points = y.reshape(y.points.shape[0], -1)
			else:
				p = os.path.abspath(os.path.join(os.getcwd(), "model.db"))
				self.model_db = p
				self.yt.points = y


			# Set initial values for theta and p
			self.theta = 0.5  # 0 < theta
			self.p = 1.0  # 0 < p < 2

			self.train()  # Run the model training function

			# Store model parameters in a Python shelve database
			db = shelve.open(self.model_db)
			db['x_train'] = self.xt.points
			db['y_train'] = self.yt.points
			db['beta'] = self.beta
			db['r_inv'] = self.r_inv
			db['theta'] = self.theta
			db['p'] = self.p
			print('\nSurrogate Data:')
			print('SCF Function: ', self.scf_func)
			print('Optimized Theta: ', db['theta'])
			print('Optimized P: ', db['p'])
			print('R Inverse: ', '\n', db['r_inv'])
			print('Beta: ', '\n', db['beta'])
			print('\n', 'Trained surrogate stored in: ', self.model_db)
			db.close()

		else:
			if model_db is not None:
				model_data = shelve.open(model_db)
				self.xt.points = model_data['x_train']
				self.yt.points = model_data['y_train']
				self.beta = model_data['beta']
				self.r_inv = model_data['r_inv']
				self.theta = model_data['theta']
				self.p = model_data['p']
				model_data.close()

				print('\nUsing', self.model_db, 'to predict values...')
			self.predict()  # Run the model prediction functions

			
	# Function to compute beta for the ordinary Kriging algorithm
	def _compute_b(self) -> np.ndarray:
		# Create a matrix of ones for calculating beta
		o = np.ones((self.xt.points.shape[0], 1))
		beta = np.linalg.multi_dot(
			[o.T, self.r_inv.points, o, o.T, self.r_inv.points, self.yt.points])
		return beta

	# Function to compute the specified SCF
	def _scf_compute(self, dist):
		if self.scf_func == 'standard':
			r = np.exp(-1 * self.theta * dist ** self.p)
			return r
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
		self.theta = x[0] 
		self.p = x[1]  
		self.r_inv: DataSet = DataSet(name="r_inv", nrows=1, ncols=1)
		self.r_inv = self._compute_r_inv(self.xt.points)
		self.beta = self._compute_b()  
		n = self.xt.points.shape[0]  
		y_b: np.ndarray = self.yt.points - \
			np.matmul(np.ones((self.yt.points.shape[0], 1)), self.beta)
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
	
	def train(self):
		x0 = [self.theta, self.p]
		fun : Callable = self._maximum_likelihood_estimator
		eval = {"blackbox": fun}
		param = {"baseline": x0,
								"lb": [0.01, 0.1],
								"ub": [10, 1.99],
								"var_names": ["theta", "p"],
								"scaling": 10.0,
								"post_dir": "./post"}
		options = {"seed": 0, "budget": 100000, "tol": 1e-12, "display": True}

		data = {"evaluator": eval, "param": param, "options":options}

		out = {}
		out = OMADS.main(data)
		results = out["xmin"]
		self.theta = results[0]
		self.p = results[1]
		self.r_inv = self._compute_r_inv(self.xt.points)
		self.beta = self._compute_b()
	def predict(self):
		r = self._scf_compute(self.calc_distance(self.xt.points, self.x.points)).T
		y_b = self.yt.points - \
			np.matmul(np.ones((self.yt.points.shape[0], 1)), self.beta)
		self.yp.points = self.beta + np.linalg.multi_dot([r, self.r_inv.points, y_b])
	
	def addToTrainingSet(self, x_new: np.ndarray, y_new: np.ndarray):
		if not self.xt.is_initialized():
			raise RuntimeError("The training set is empty! Initialize it before appending more data to it.")
		
		for i in range(np.size(x_new, axis=0)):
			if not x_new[i].tolist() in self.xt.points.tolist():
				self.xt.points = np.append(self.xt.points, [x_new[i]], axis=0)
				self.yt.points = np.append(self.yt.points, [y_new[i]], axis=0)

@dataclass
class bmSM:
	xhist: np.ndarray = np.empty((1,2))
	fhist: List = list
	def RB(self, x, kx):
		ne, nx = x.shape
		y = np.zeros((ne, 1), complex)
		if kx is None:
			for ix in range(nx - 1):
				y[:, 0] += (
					100.0 * (x[:, ix + 1] - x[:, ix] ** 2) ** 2 +
					(1 - x[:, ix]) ** 2
				)
		else:
			if kx < nx - 1:
				y[:, 0] += -400.0 * (x[:, kx + 1] - x[:, kx] ** 2) * x[:, kx] - 2 * (
					1 - x[:, kx]
				)
			if kx > 0:
				y[:, 0] += 200.0 * (x[:, kx] - x[:, kx - 1] ** 2)

		return y
	
	def RB1(self, x, kx):
		ne, nx = x.shape
		y = np.zeros((ne, 1), complex)
		if kx is None:
			for ix in range(nx - 1):
				y[:, 0] += (
					100.0 * (x[:, ix + 1] - x[:, ix] ** 2) +
					(1 - x[:, ix])
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

		plt.contour(X, Y, Z, 250)
		# self.xhist = np.array([[0.85,0.8]])
		self.RB_min_true()
		plt.plot(self.xhist[:,0], self.xhist[:,1], '-k')
		plt.plot(1., 1., 'D')
		plt.plot(self.xhist[0,0], self.xhist[0,1],'o')
		plt.plot(self.xhist[-1,0], self.xhist[-1,1],'s')
		plt.xlabel("x")
		plt.ylabel("y")
		plt.show()


	def RB_RBF(self):
		v = np.array([[-2.0, 2.0], [-2.0, 2.0]])
		n = 50
		# Generate initial sample points
		sampling = FullFactorial(ns=n, vlim=v, w=None, c=False)
		# Create the training set xt, yt
		xt = sampling.generate_samples()
		yt = self.RB(xt, None)
		opts: Dict = {}
		opts["weights"] = np.zeros((1,1))
		# Build the surrogate model
		self.sm = RBF(type="train", x=xt, y=yt, options=opts, rbf_func="cubic")
		w1 = self.sm.weights

		# Generate points to predict the approximated responses on
		sampling = LHS(ns=n, vlim=v)
		sampling.options["criterion"] = "ExactSE"
		sampling.options["randomness"] = 451236
		xp = sampling.generate_samples()
		self.sm.x.points = xp
		# Predict responses using the trained surrogate 
		self.sm.predict()
		yp = copy.deepcopy(self.sm.yp)
		# The following commented lines used for testing samples injection and surrogate retraining
		# xp_new: DataSet = DataSet(name="extend", nrows=1, ncols=1)
		# yp_new: DataSet = DataSet(name="extend", nrows=1, ncols=1)
		# np.random.seed = 1000323
		# sampling2 = LHS(ns=n, vlim=v)
		# # Create the training set xt, yt
		# xt2 = sampling2.generate_samples()
		# yt2 = self.RB1(xt, None)
		# self.sm.addToTrainingSet(xt2, yt2)
		# self.sm.train()
		# self.sm.yp.points = np.zeros((1,1))
		# w2 = self.sm.weights
		# self.sm.predict()
		# yp2 = copy.deepcopy(self.sm.yp)
		X, Y = np.meshgrid(np.sort(xp[:, 0]), np.sort(xp[:, 1]))
		Z = np.ndarray((n, n))

		for i in range(n):
			S = np.array([np.transpose(X[i, :]), np.transpose(Y[i, :])])
			self.sm.x.points = S.T
			self.sm.predict()
			Z[i, :] = self.sm.yp.points.T

		plt.contour(X, Y, Z, 250)
		self.RB_min_rbf()
		plt.plot(self.xhist[:,0], self.xhist[:,1], '-k')
		plt.plot(1., 1., 'D')
		plt.plot(self.xhist[0,0], self.xhist[0,1],'o')
		plt.plot(self.xhist[-1,0], self.xhist[-1,1],'s')
		plt.xlabel("x")
		plt.ylabel("y")
		plt.show()

	def RB_RBF1(self):
		v = np.array([[-2.0, 2.0], [-2.0, 2.0]])
		n = 50

		sampling = FullFactorial(ns=n, vlim=v, w=None, c=False)

		xt = sampling.generate_samples()
		yt = self.RB(xt, None)

		opts: Dict = {}
		opts["weights"] = np.empty((1, 1))

		self.sm = RBF(type="train", x=xt, y=yt, options=opts, rbf_func="linear")


		sampling = LHS(ns=10, vlim=v)
		sampling.options["criterion"] = "ExactSE"

		xp = sampling.generate_samples()
		self.sm.x.points = xp
		# Predict responses using the trained surrogate 
		self.sm.predict()
		yp = copy.deepcopy(self.sm.yp)
		X, Y = np.meshgrid(np.sort(xt[:, 0]), np.sort(xt[:, 1]))
		Z = np.ndarray((n, n))

		for i in range(n):
			S = np.array([np.transpose(X[i, :]), np.transpose(Y[i, :])])
			self.sm.x.points = S.T
			self.sm.predict()
			Z[i, :] = self.sm.yp.points.T

		plt.contour(X, Y, Z, 250)
		self.RB_min_rbf()
		plt.plot(self.xhist[:,0], self.xhist[:,1], '-k')
		plt.plot(1., 1., 'D')
		plt.plot(self.xhist[0,0], self.xhist[0,1],'o')
		plt.plot(self.xhist[-1,0], self.xhist[-1,1],'s')
		plt.xlabel("x")
		plt.ylabel("y")
		plt.show()


	def callbackF(self, Xi):

		Xi[0] = np.minimum(Xi[0], 2)
		Xi[1] = np.minimum(Xi[1], 2)
		Xi[0] = np.maximum(Xi[0], -2)
		Xi[1] = np.maximum(Xi[1], -2)
		self.xhist = np.append(self.xhist, [Xi], axis=0)
		self.fhist.append(rosen(Xi))
		self.nsqp += 1
	
	def RB_min_true(self):
		self.nsqp = 0
		x = minimize(rosen, self.xhist, callback=self.callbackF, method="SLSQP", jac=rosen_der, tol=1e-6, bounds=np.array([[0.8,1.2], [0.8,1.2]]))

	
	def rosen_krig(self, x: np.ndarray):
		X = np.append([x], np.array([[0.8, 0.8]]), axis=0)
		self.sm.x.points = X
		self.sm.predict() 
		return (self.sm.yp.points[0]).real

	def rosen_rbf(self, x: np.ndarray):
		X = np.append([x], np.array([[0.8, 0.8]]), axis=0)
		self.sm.x.points = X
		self.sm.predict() 
		return (self.sm.yp.points[0]).real	

		
	def RB_min_krig(self):
		self.nsqp = 0
		x = minimize(self.rosen_krig, self.xhist, callback=self.callbackF, method="SLSQP", tol=1e-6, bounds=np.array([[.5,1.5], [.5,1.5]]))
	
	def RB_min_rbf(self):
		self.nsqp = 0
		x = minimize(self.rosen_rbf, self.xhist, callback=self.callbackF, method="SLSQP", tol=1e-6, bounds=np.array([[.5,1.5], [.5,1.5]]))
	

	def RB_kriging(self):
		v = np.array([[-2.0, 2.0], [-2.0, 2.0]])
		n = 50
		sampling = FullFactorial(ns=n, vlim=v, w=None, c=False)
		xt = sampling.generate_samples()
		yt = self.RB(xt, None)
		opts: Dict = {}
		self.sm = Kriging(type="train", x=xt, y=yt, options=opts)
		sampling = LHS(ns=n, vlim=v)
		sampling.options["criterion"] = "ExactSE"
		xp = copy.deepcopy(sampling.generate_samples())
		# self.sm.xt.points = self.sm.x.points
		self.sm.x.points = copy.deepcopy(xp)
		self.sm.predict()
		yp = copy.deepcopy(self.sm.yp)

		# The following commented lines used for testing samples injection and surrogate retraining
		# # np.random.seed(1000323)
		# sampling2 = LHS(ns=2, vlim=v)
		# sampling.options["criterion"] = "ExactSE"
		# # Create the training set xt, yt
		# xt2 = sampling2.generate_samples()
		# yt2 = self.RB1(xt2, None)
		# self.sm.addToTrainingSet(xt2, yt2)
		# self.sm.train()
		# self.sm.yp.points = np.zeros((1,1))
		# self.sm.predict()
		# yp2 = copy.deepcopy(self.sm.yp)
		# print(yp.points-yp2.points)

		X, Y = np.meshgrid(np.sort(xp[:, 0]), np.sort(xp[:, 1]))
		Z = np.ndarray((n, n))

		for i in range(n):
			S = np.array([np.transpose(X[i, :]), np.transpose(Y[i, :])])
			self.sm.x.points = S.T
			self.sm.predict()
			Z[i, :] = self.sm.yp.points.T

		plt.contour(X, Y, Z, 250)
		self.RB_min_krig()
		plt.plot(self.xhist[:,0], self.xhist[:,1], '-k')
		plt.plot(self.xhist[0,0], self.xhist[0,1],'o')
		plt.plot(1., 1., 'D')
		plt.plot(self.xhist[-1,0], self.xhist[-1,1],'s')
		plt.xlabel("x")
		plt.ylabel("y")
		plt.show()

	def RB_kriging1(self):
		v = np.array([[-2.0, 2.0], [-2.0, 2.0]])
		n = 20
		sampling = FullFactorial(ns=n, vlim=v, w=None, c=False)

		xt = sampling.generate_samples()
		yt = self.RB(xt, None)
		
		opts: Dict = {}

		self.sm = Kriging(type="train", x=xt, y=yt, options=opts)


		sampling = LHS(ns=n, vlim=v)
		sampling.options["criterion"] = "ExactSE"

		xp = sampling.generate_samples()
		self.sm.x.points = copy.deepcopy(xp)
		self.sm.predict()
		yp = self.sm.yp
		X, Y = np.meshgrid(np.sort(xp[:, 0]), np.sort(xp[:, 1]))
		Z = np.ndarray((n, n))

		for i in range(n):
			S = np.array([np.transpose(X[i, :]), np.transpose(Y[i, :])])
			self.sm.x.points = S.T
			self.sm.predict()
			Z[i, :] = self.sm.yp.points.T

		plt.contour(X, Y, Z, 250)
		self.RB_min_krig()
		plt.plot(self.xhist[:,0], self.xhist[:,1], '-k')
		plt.plot(self.xhist[0,0], self.xhist[0,1],'o')
		plt.plot(1., 1., 'D')
		plt.plot(self.xhist[-1,0], self.xhist[-1,1],'s')
		plt.xlabel("x")
		plt.ylabel("y")
		plt.show()

if __name__ == "__main__":

	p = bmSM()
	p.fhist = []
	s = np.array([[1.2,1.2]])
	p.xhist = s
	p.RBTrue()
	p.xhist = s
	p.RB_RBF()
	p.xhist = s
	p.RB_RBF1()
	p.xhist = s
	p.RB_kriging()
	p.xhist = s
	p.RB_kriging1()
