# ------------------------------------------------------------------------------------#
#  Surrogate Model Library (SML)                                                      #
#  Copyright (C) 2018-2020  Ahmed Bayoumy - McGill University, Montreal               #
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
#  You can find information on SML at                                                 #
#  https://github.com/Ahmed-Bayoumy/SML                                               #
# ------------------------------------------------------------------------------------#

import os
import json
import pickle
import sys
from typing import List, Optional, Dict, Any
import copy

import numpy as np
import pydantic
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import OMADS


@dataclass
class scaling:
    method: int
    SCALING_NONE: int = 0
    SCALING_MEANSTD: int = 1
    SCALING_BOUNDS: int = 2


@dataclass
class constants:
    EPSILON: float = 1E-9
    boolean_rounding: int = 2
    APPROX_CDF: bool = True


@dataclass
class model_t:
    LINEAR: int = 0
    TGP: int = 1
    DYNATREE: int = 2
    PRS: int = 3
    PRS_EDGE: int = 4
    PRS_CAT: int = 5
    KS: int = 6
    CN: int = 7
    KRIGING: int = 8
    SVN: int = 9
    RBF: int = 10
    LOWESS: int = 11
    ENSEMBLE: int = 12


@dataclass
class param_status_t:
    STATUS_FIXED: int = 0
    STATUS_OPTIM: int = 1
    STATUS_MODEL_DEFINED: int = 2


@dataclass
class kernel_t:
    # Gaussian
    KERNEL_D1: int = 0
    # Inverse Quadratic
    KERNEL_D2: int = 1
    # Inverse multiquadratic
    KERNEL_D3: int = 2
    # Bi-quadratic
    KERNEL_D4: int = 3
    # Tri-quadratic
    KERNEL_D5: int = 4
    # Exp-SQRT
    KERNEL_D6: int = 5
    # EPANECHNIKOV
    KERNEL_D7: int = 6
    # Multiquadratic
    KERNEL_I0: int = 7
    # Poly1 (Spline 1)
    KERNEL_I1: int = 8
    # Poly2 (Spline 2)
    KERNEL_I2: int = 9
    # Poly3 (Spline 3)
    KERNEL_I3: int = 10
    # Poly4 (Spline 4)
    KERNEL_I4: int = 11


@dataclass
class distance_t:
    DISTANCE_NORM2: int = 0
    DISTANCE_NORM1: int = 1
    DISTANCE_NORMINF: int = 2
    DISTANCE_NORM2_IS0: int = 3
    DISTANCE_NORM2_CAT: int = 4
    NB_DISTANCE_TYPES: int = 5


@dataclass
class weight_t:
    WEIGHT_SELECT: int = 0  # Take the model with the best metrics.
    WEIGHT_OPTIM: int = 1  # Optimize the metric
    WEIGHT_WTA1: int = 2  # Goel, Ensemble of surrogates 2007
    WEIGHT_WTA3: int = 3  # Goel, Ensemble of surrogates 2007
    WEIGHT_EXTERN: int = 4  # Belief vector is set externally by the user.


@dataclass
class metric_t:
    METRIC_EMAX: int = 0  # Max absolute error
    METRIC_EMAXCV: int = 1  # Max absolute error on cross-validation value
    METRIC_RMSE: int = 2  # Root mean square error
    METRIC_ARMSE: int = 3  # Agregate Root mean square error
    METRIC_RMSECV: int = 4  # Leave-one-out cross-validation
    METRIC_ARMSECV: int = 5  # Agregate Leave-one-out cross-validation
    METRIC_OE: int = 6  # Order error on the training points
    METRIC_OECV: int = 7  # Order error on the cross-validation output
    METRIC_AOE: int = 8  # Agregate Order error
    METRIC_AOECV: int = 9  # Agregate Order error on the cross-validation output
    METRIC_EFIOE: int = 10  # Order error on the cross-validation output
    METRIC_EFIOECV: int = 11  # Agregate Order error on the cross-validation output
    METRIC_LINV: int = 12  # Inverse of the likelihood

    def metric_uses_cv(self, mt: int):
        if mt in [self.METRIC_EMAXCV, self.METRIC_RMSECV, self.METRIC_OECV, self.METRIC_ARMSECV, self.METRIC_AOECV,
                  self.METRIC_EFIOECV]:
            return True
        elif mt in [self.METRIC_EMAX, self.METRIC_RMSE, self.METRIC_OE, self.METRIC_LINV, self.METRIC_ARMSE,
                    self.METRIC_AOE, self.METRIC_EFIOE]:
            return False
        else:
            raise ExceptionError(name="metrics", message="Undefined metric")

    def metric_type_to_norm_type(self, mt: int):
        if mt in [self.METRIC_EMAX, self.METRIC_EMAXCV]:
            return norm_t.NORM_INF
        elif mt in [self.METRIC_RMSE, self.METRIC_RMSECV, self.METRIC_ARMSE, self.METRIC_ARMSECV]:
            return norm_t.NORM_2
        else:
            raise ExceptionError(name="metric_type_to_norm", message="This metric does not have an associated norm")



@dataclass
class bbo_t:
    BBO_OBJ: int = 0
    BBO_CON: int = 1
    BBO_DUM: int = 2


@dataclass
class norm_t:
    NORM_0: int = 0
    NORM_1: int = 1
    NORM_2: int = 2
    NORM_INF: int = 3



@dataclass
class param_domain_t:
    PARAM_DOMAIN_CONTINUOUS: int = 0
    PARAM_DOMAIN_INTEGER: int = 1
    PARAM_DOMAIN_BOOL: int = 2
    PARAM_DOMAIN_CAT: int = 3
    PARAM_DOMAIN_MISC: int = 4


@dataclass
class DEF_Settings:
    _norm_t: norm_t
    _scaling_t: scaling
    _bbo_t: bbo_t
    _param_domain_t: param_domain_t
    _param_status_t: param_status_t
    _EPSILON: float = constants.EPSILON
    _APPROX_CDF: bool = True
    _scaling_method: int = scaling.SCALING_MEANSTD
    # 0: no boolean scaling
    # 1: threshold = (Z_UB+Z_LB)/2;
    # 2: threshold = mean(Z)
    _boolean_rounding: int = 2


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


@dataclass
class surrogate_utils(ABC):
    """ Surrogate utilities """
    a: int = 0

    @abstractmethod
    def isdef(self, mt: int, j: Optional[int] = None) -> bool:
        """ Check if the value is defined """

    def normcdf(self, x:float,  mu: Optional[float]=None, sigma: Optional[float]=None):
        """ /*----------------------------------------*/
            /*  Compute CUMULATIVE Density Function   */
            /*  (Centered & Normalized Gaussian law)  */
            /*----------------------------------------*/ """
        if mu is not None and sigma is not None:
            if sigma < constants.EPSILON:
                raise Exception("Surrogate_Utils::normpdf: sigma is <0")
            if constants.APPROX_CDF:
                sigma = max(sigma, constants.EPSILON)
            # compute CDF
            if sigma < constants.EPSILON:
                if x > mu:
                    return 1.0
                else:
                    return 0.
            else:
                # Normal case
                x = (x - mu) / sigma

        if abs(x) < constants.EPSILON:
            Phi = 0.5
        else:
            t = 1.0 / (1.0 + 0.2316419 * abs(x))
            t2 = t * t
            v = np.exp(-x * x / 2.0) * t * (0.319381530 - 0.356563782 * t +
                                            1.781477937 * t2 - 1.821255978 * t * t2 + 1.330274429 * t2 * t2) / 2.506628274631
            if x < 0.0:
                Phi = v
            else:
                Phi = 1.0 - v
        return Phi

    def normei (self, fh: float, sh: float, f_min: float):
        if sh <-constants.EPSILON:
            raise Exception("Surrogate_Utils::normei: sigma is <0")
        if constants.APPROX_CDF:
            sh = max(sh, constants.EPSILON)
        if sh<constants.EPSILON:
            if (fh<f_min):
                return f_min-fh
            else:
                return 0
        else:
            d: float = (f_min-fh)/sh
            return (f_min-fh)*self.normcdf(d) + sh*self.normcdf(d)



class DataSet(pydantic.BaseModel):
    _points: List[List[float]]
    _name: str
    _nrows: int
    _ncols: int

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
    def points(self, value):
        self._points = copy.deepcopy(value)

    @classmethod
    def points_set(self, i, j, value):
        self._points[i][j] = value

    @classmethod
    def points_get(self, i, j):
        return self._points[i][j]

    def initialize_data_matrix(self, name: str, nr: int, nc: int):
        if name:
            self.name = name
        self.nrows = nr
        self.ncols = nc
        for _ in range(nr):
            self._points.append([0] * nc)

    def import_data_matrix(self, file: str):
        if not os.path.exists(file):
            raise FileMissingError(
                name=file,
                message="Could not find the provided file and/or path!"
            )
        if file.endswith('.dat'):
            with open(file, "rb") as f:
                _points = copy.deepcopy(pickle.load(f))
        elif (file.endswith('.json')):
            with open(file) as f:
                data: dict = json.load(f)
            self._points = copy.deepcopy(data["data"])
        else:
            raise FileMissingError(
                name=file,
                message="The provided file extension is not supported!"
            )

    def get_nb_diff_values(self, j: int):
        s: List[float] = []
        for i in range(self.nrows):
            s.append(self.points[i][j])
        return len(s)

    def get_row(self, row):
        return self._points[row]

    def get_max_index(self):
        ki = 0
        vmax = 0.
        for i in range(self.nrows):
            for j in range(self.ncols):
                if self.points_get(i, j) > vmax:
                    vmax = self.points_get(i, j)
                    k = ki
                ki += 1
        return k

    def get_min_index_row(self, i)->int:
        vmin = 0.
        for j in range(self.ncols):
            if self.points_get(i, j) < vmin:
                vmin = self.points_get(i, j)
                jmin = j
        return j

    def get_element(self, k: int):
        ki = 0
        for i in range(self.nrows):
            for j in range(self.ncols):
                if k == ki:
                    return self.points_get(i, j)
                ki += 1
        ExceptionError(name="Dataset::get_element::", message="The element index exceeds that data set size.")

    def normalize_cols(self):
        """ /*----------------------------------------*/
            /*        normalize_cols                  */
            /* Normalizes each column so that the sum */
            /* of the terms on this column is 1       */
            /*----------------------------------------*/ """
        for j in range(self.ncols):
            d = 0
            for i in range(self.nrows):
                d += self.points[i][j]
            if d == 0:
                for i in range(self.nrows):
                    self.points[i][j] = 1 / self.nrows
            else:
                for i in range(self.nrows):
                    self.points[i][j] /= d

    def is_initialized(self) -> bool:
        return self.nrows > 0 or self.ncols > 0

    def add_cols(self, A):
        """ Add columns """
        if A.nrows != self.nrows:
            ExceptionError(name="Dataset::add_cols::", message="bad dimensions")

        new_nbCols: int = self.ncols + A.ncols

        for i in range(self.nrows):
            x: List[float] = [] * new_nbCols
            for j in range(self.ncols):
                x[j] = self.points_get(i, j)
            for j in range(new_nbCols):
                x[j] = A.points_get(i, j - self.ncols)
            self.points[i] = x
            self.ncols = new_nbCols

    def add_rows(self, A):
        """ Add rows """
        if A.ncols != self.ncols:
            ExceptionError(name="Dataset::add_rows::", message="bad dimensions")
        self.points.append(A)

    def swap(self, i1, j1, i2, j2):
        buffer: float = self.points_get(i1, j1)
        self.points_set(i1, j1, self.points_get(i2, j2))
        self.points_set(i2, j2, buffer)

    def min(self, A, B):
        if A.ncols != B.ncols or A.nrows != B.ncols:
            raise ExceptionError(message="Matrix::min(A,B): dimension error")

        self.nrows = A.nrows
        self.ncols = A.ncols
        for i in range(self.nrows):
            for j in range(self.ncols):
                self.points_set(i,j, min(A.points_get(i,j), B.points_get(i,j)))


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
        for i in range(self.nrows): self.points[i][cindex] = c.points_get(i, cindex)

    def __get_col__(self, j: int):
        c: DataSet = DataSet(name=self.name, nrows=self.nrows, ncols=1)
        for i in range(self.nrows): c.points[i][0] = self.points[i][j]
        return c










@dataclass
class Kernel:
    K_type: kernel_t
    NB_KERNEL_TYPES: int
    NB_DECREASING_KERNEL_TYPES: int

    def kernel_is_decreasing(self, kt: int) -> bool:
        kt_D = [self.K_type.KERNEL_D1, self.K_type.KERNEL_D2, self.K_type.KERNEL_D3,
                self.K_type.KERNEL_D4, self.K_type.KERNEL_D5, self.K_type.KERNEL_D6,
                self.K_type.KERNEL_D7]
        kt_I = [self.K_type.KERNEL_I0, self.K_type.KERNEL_I1,
                self.K_type.KERNEL_I2, self.K_type.KERNEL_I3, self.K_type.KERNEL_I4]
        for f in kt_D:
            if (f == kt):
                return True
        for f in kt_I:
            if (f == kt):
                return False
        ExceptionError(name="Kernel_t:kernel_is_decreasing", message="Undefined kernel type")
        return False

    def kernel_has_parameter(self, kt: int) -> bool:
        kt_1 = [self.K_type.KERNEL_D1, self.K_type.KERNEL_D2, self.K_type.KERNEL_D3,
                self.K_type.KERNEL_D4, self.K_type.KERNEL_D5, self.K_type.KERNEL_D6,
                self.K_type.KERNEL_D7, self.K_type.KERNEL_I0]
        kt_2 = [self.K_type.KERNEL_I1,
                self.K_type.KERNEL_I2, self.K_type.KERNEL_I3, self.K_type.KERNEL_I4]
        for f in kt_1:
            if (f == kt):
                return True
        for f in kt_2:
            if (f == kt):
                return False
        ExceptionError(name="Kernel_t:kernel_is_decreasing", message="Undefined kernel type")
        return False

    def kernel_dmin(self, kt) -> int:
        kt_1 = [self.K_type.KERNEL_D1, self.K_type.KERNEL_D2, self.K_type.KERNEL_D3,
                self.K_type.KERNEL_D4, self.K_type.KERNEL_D5, self.K_type.KERNEL_D6,
                self.K_type.KERNEL_D7]
        kt_2 = [self.K_type.KERNEL_I0, self.K_type.KERNEL_I1]
        kt_3 = [self.K_type.KERNEL_I2, self.K_type.KERNEL_I3, self.K_type.KERNEL_I4]
        for f in kt_1:
            if (f == kt):
                return -1
        for f in kt_2:
            if (f == kt):
                return 0
        for f in kt_3:
            if (f == kt):
                return 1
        ExceptionError(name="Kernel_t:kernel_is_decreasing", message="Undefined kernel type")
        return -2

    def compute(self, kt: int, ks: float, r: float) -> float:
        """ Compute the value of the kernel """
        if kt == self.K_type.KERNEL_D1:
            return np.exp(-np.pi * ks * ks * r * r)
        elif kt == self.K_type.KERNEL_D2:
            return 1. / (1. + np.pi * np.pi * ks * ks * r * r)
        elif kt == self.K_type.KERNEL_D3:
            return 1. / np.sqrt(1. + 52.015 * ks * ks * r * r)
        elif kt == self.K_type.KERNEL_D4:
            ksr: float = abs(ks * r) * 16. / 15.
            if ksr <= 1:
                d: float = 1 - ksr * ksr
                return d * d
            return 0.
        elif kt == self.K_type.KERNEL_D5:
            ksr: float = abs(ks * r) * 162. / 140.
            if ksr <= 1:
                d: float = 1 - ksr * ksr * ksr
                return d * d * d
            return 0.
        elif kt == self.K_type.KERNEL_D6:
            return np.exp(-np.sqrt(4 * ks * r))
        elif kt == self.K_type.KERNEL_D7:
            ksr: float = abs(ks * r)
            if ksr <= 3. / 4.: return 1. - (16. / 9.) * ksr * ksr
            return 0.
        elif kt == self.K_type.KERNEL_I0:
            return np.sqrt(1. + ks * ks * r * r)
        elif kt == self.K_type.KERNEL_I1:
            # Polyharmonic spline
            return r
        elif kt == self.K_type.KERNEL_I2:
            # (Thin Plate Splin)
            if r == 0.: return 0.
            return np.log(r) * r * r
        elif kt == self.K_type.KERNEL_I3:
            return r * r * r
        elif kt == self.K_type.KERNEL_I4:
            if r == 0.: return 0.
            r2: float = r * r
            return r2 * r2 * np.log(r)
        else:
            ExceptionError(name="Kernel_t:Compute", message="Undefined kernel type")
        return 0.

    def compute_2d(self, kt: int, ks: float, R: DataSet) -> DataSet:
        """ Compute the value of the kernel """
        nbRows: int = R.nrows
        nbCols: int = R.ncols
        for i in range(nbRows):
            for j in range(nbCols):
                R.points_set(kt, ks, R.points_get(i, j))
        return R


class TrainingDataSet(pydantic.BaseModel, surrogate_utils):
    """ Represents the data model"""
    # Number of data points in X and Z
    _p: int
    # Dimension
    _n: int
    # number of response outputs
    _m: int
    # Check if the data has been processed and ready to be used
    _ready: bool
    # Function type
    _bbo: List[int]
    # Objective index
    _bbo_is_def: bool
    _j_obj: int
    # Optimal responses
    _f_min: float
    _fs_min: float
    # Index of the point where f_min is reached
    _i_min: int
    # Data points
    # p x n
    _X: DataSet
    # p x m
    _Z: DataSet

    # Scaled matrices
    # p x n
    _Xs = DataSet
    # p x m
    _Zs = DataSet

    # Distance Matrix
    # p x p
    _Ds: DataSet

    # Nb of varying data
    ## Nb varying input
    _nvar: int = -1
    ## Nb of varying output
    _mvar: int = -1
    # Nb of different points
    _pvar: int = -1


    # Data
    _X_lb: List[float]
    _X_ub: List[float]
    _X_scaling_a: List[float]
    _X_scaling_b: List[float]
    _X_mean: List[float]
    _X_std: List[float]
    _X_nbdiff: List[float]
    _X_nbdiff1: int
    _X_nbdiff2: int
    _Z_lb: List[float]
    _Z_ub: List[float]
    _Z_replace: List[float]
    _Z_scaling_a: List[float]
    _Z_scaling_b: List[float]
    _Z_mean: List[float]
    _Z_std: List[float]
    _Zs_mean: List[float]
    _Z_nbdiff: List[float]

    # Mean distance between points
    _Ds_mean: float

    @property
    def pvar(self):
        return self._pvar

    @pvar.setter
    def pvar(self, value):
        self._pvar = value

    @property
    def nvar(self):
        return self._nvar

    @nvar.setter
    def nvar(self, value):
        self._nvar = value

    def get_input_dim(self):
        return self._n

    def get_f_min(self):
        self.check_ready()
        return self._f_min

    def get_output_dim(self):
        return self._m

    def get_nb_points(self):
        return self._p

    @property
    def ready(self)->bool:
        return self._ready

    @ready.setter
    def ready(self, value):
        self._ready = value

    def check_ready(self):
        if self.ready is  False:
            raise ExceptionError(name="Error", message="TrainingSet::check_ready(): TrainingSet not ready. "
                                         "Use method TrainingSet::build()")
        else:
            return self.ready
    @property
    def Xs(self):
        return self._Xs
    
    @Xs.setter
    def Xs(self, value):
        self._Xs = value
    
    @property
    def Zs(self):
        return self._Zs
    
    @Zs.setter
    def Zs(self, value):
        self._Zs = value
        
    @property
    def Ds(self):
        return self._Ds
    
    @Ds.setter
    def Ds(self, value):
        self._Ds = value

    def get_matrix_Xs(self)->DataSet:
        self.check_ready()
        return self.Xs

    def get_matrix_Zs(self)->DataSet:
        self.check_ready()
        return self.Zs

    def get_matrix_Ds(self)->DataSet:
        self.check_ready()
        return self.Ds

    # Data preparation
    @staticmethod
    def compute_nbdiff(MAT: DataSet) -> int:
        nj = MAT.ncols
        njvar = 0
        nbdiff = []
        for j in range(nj):
            nbdiff.append(MAT.get_nb_diff_values(j))
            if nbdiff[j] > 1: njvar += 1

        return njvar

    def isdef(self, value) -> bool:
        """ Not NAN nor INF """
        if np.isnan(value): return False
        if np.isinf(value): return False
        if np.fabs(value) >= np.inf: return False
        if np.fabs(value) >= 1e16: return False
        return True

    def compute_bounds(self):
        # Bound of X
        for j in range(self._n):
            self._X_lb[j] = np.inf
            self._X_ub[j] = -np.inf
            for i in range(self._p):
                v = self._X.points[i][j]
                self._X_lb[j] = min(v, self._X_lb[j])
                self._X_ub[j] = max(v, self._X_ub[j])

        # Bound of Z
        for j in range(self._n):
            self._Z_lb[j] = np.inf
            self._Z_ub[j] = -np.inf
            for i in range(self._p):
                v = self._Z.points[i][j]
                self._Z_lb[j] = min(v, self._Z_lb[j])
                self._Z_ub[j] = max(v, self._Z_ub[j])

            # Compute replacement value for undef Z
            # If there are no correct bounds defined yet
            if ((self._Z_lb[j]) or (self._Z_ub[j])):
                self._Z_replace[j] = 1.0
            else:
                self._Z_replace[j] = max(self._Z_ub[j], 0.) + 0.1 * max(self._Z_ub[j] - self._Z_lb[j], 1.)

    def compute_scaling(self):
        """ Compute scaling parameters """
        # Neutral values
        for j in range(self._n):
            self._X_scaling_a[j] = 1
            self._X_scaling_b[j] = 0
        for j in range(self._m):
            self._Z_scaling_a[j] = 1
            self._Z_scaling_b[j] = 0

        if scaling.method == scaling.SCALING_NONE:
            return
        elif scaling.method == scaling.SCALING_MEANSTD:
            # Compute mean and std over columns of X and Z
            self.compute_mean_std()
            # Compute scaling constants
            for j in range(self._n):
                if self._X_nbdiff[j] > 1: self._X_scaling_a[j] = 1 / self._X_std[j]
                self._X_scaling_b[j] = -self._X_mean[j] * self._X_scaling_a[j]
            for j in range(self._m):
                if self._Z_nbdiff[j] > 1: self._Z_scaling_a[j] = 1 / self._Z_std[j]
                self._Z_scaling_b[j] = -self._Z_mean[j] * self._Z_scaling_a[j]
            return
        elif scaling.method == scaling.SCALING_BOUNDS:
            # Compute scaling constants
            for j in range(self._n):
                if self._X_nbdiff[j] > 1: self._X_scaling_a[j] = 1 / self._X_std[j]
                self._X_scaling_b[j] = -self._X_mean[j] * self._X_scaling_a[j]
            for j in range(self._m):
                if self._Z_nbdiff[j] > 1: self._Z_scaling_a[j] = 1 / self._Z_std[j]
                self._Z_scaling_b[j] = -self._Z_mean[j] * self._Z_scaling_a[j]
            return

    def compute_scaled_matrices(self):
        """ Compute scale matrices _Xs and _Zs """
        # Compute _Xs
        for j in range(self._n):
            for i in range(self._p):
                v = self._X.points[i][j] * self._X_scaling_a[j] + self._X_scaling_b[j]
                self._Xs.points_set(i, j, v)

        # Compute _Zs and Mean_Zs
        for j in range(self._n):
            mu = 0
            for i in range(self._p):
                v = self._Z.points[i][j]
                if not self.isdef(v): v = self._Z_replace
                v *= self._Z_scaling_a[j] + self._Z_scaling_b[j]
                mu += v
                self._Zs.points_set(i, j, v)
            self._Zs_mean[j] = mu / self._p

    def compute_Ds(self):
        """ compute distance matrix; the columns of a matrix """
        self._pvar = copy.deepcopy(self._p)
        self._DS_mean = 0.0
        unique: bool
        for i1 in range(self._p - 1):
            self._Ds.points_set(i1, i1, 0.)
            unique = copy.deepcopy(True)
            for i2 in range(self._p):
                d = 0
                for j in range(self._n):
                    di1i2 = self._Xs.points_get(i1, j) - self._Xs.points_get(i2, j)
                    d += di1i2 * di1i2
                d = np.sqrt(d)
                self._Ds.points_set(i1, i2, d)
                self._Ds.points_set(i2, i1, d)
                # Compute the mean distance between the points
                self._Ds_mean += d
                # If d==0, then the point i2 is not unique.
                if np.fabs(d) < constants.EPSILON:
                    unique = copy.deepcopy(False)
                # If there are some points equal to the point of index i2,
                # then reduce the number of different points.
            if not unique: self._pvar -= 1
        self._DS_mean /= self._pvar * (self._pvar - 1) / 2

    def Z_scale(self, z, j: int = None):
        if j is None:
            for j in range(self._m):
                z[j] = self._Z_scaling_a[j] * z[j] + self._Z_scaling_b[j]
        else:
            return self._Z_scaling_a[j] * z + self._Z_scaling_b[j]

    def compute_f_min(self):
        """ compute fs_min (scaled value of f_min) """
        # Go through all points
        for i in range(self._p):
            # Get the unscaled objective
            f = self._Z.points[i][self._j_obj]
            # If objective is good
            if f < self._f_min:
                feasible = True
                for j in range(self._m):
                    if self._bbo[j] == bbo_t.BBO_CON:
                        if self._Z.points[i][j] > 0.: feasible = False
                if feasible:
                    self._f_min = f
                    self._i_min = i
        self._fs_min = self.Z_scale(self._f_min, self._j_obj)

    def check_singular_data(self):
        """ compute the mean and std over the columns of a matrix """
        # Check that all the _X data are defined
        e: bool = False
        # Check that all the _X data are defined
        for j in range(self._n):
            for i in range(self._p):
                if not self.isdef(self._X.points[i][j]):
                    print("_X(" + str(i) + ", " + str(j) + ") = " + str(self._X.points[i][j]))
                    e = True
        # Check that, for each output index, SOME data are defined
        isdef_Zj: bool  # True if at least one value is defined for output j.
        # Loop on the output indexes
        for j in range(self._m):
            # no def value so far
            isdef_Zj = False
            for i in range(self._p):
                if self.isdef(self._Z.points[i][j]):
                    isdef_Zj = True
            # if there is more than 10 points and no correct value was found, return an error.
            if self._p > 10 and not isdef_Zj:
                print("_Z(:," + str(j) + ") has no defined value")
                e = True
        if e: raise ExceptionError(
            name="TrainingSet::check_singular_data():",
            message="Incorrect data!"
        )

    def compute_mean_std(self):
        """ compute the mean and std over the columns of a matrix """
        # Loop on the inputs
        for j in range(self._n):
            # Loop on lines for MEAN computation
            mu = 0
            for i in range(self._p):
                mu += self._X.points[i][j]
            mu /= self._p
            self._X_mean[j] = mu
            # Loop on lines for VAR computation
            var = 0
            for i in range(self._p):
                v = self._X.points[i][j]
                var += (v - mu) * (v - mu)
            var /= (self._p - 1)
            self._X_std[j] = np.sqrt(var)

        # Loop on the outputs
        for j in range(self._m):
            # Loop on lines for MEAN computation
            mu = 0
            for i in range(self._p):
                v = self._Z.points[i][j]
                if not self.isdef(v): v = self._Z_replace[j]
                mu += v
            mu /= self._p
            self._Z_mean[j] = mu
            # Loop on lines for VAR computation
            var = 0
            for i in range(self._p):
                v = self._Z.points[i][j]
                if not self.isdef(v): v = self._Z_replace[j]
                var += (v - mu) * (v - mu)
            var /= (self._p - 1)
            self._Z_std[j] = np.sqrt(var)

    def build(self):
        # Check the dimensions
        if self._X.nrows != self._Z.nrows:
            raise ExceptionError(
                name="TrainingSet::build():",
                message="Dimension error"
            )
        # Check number of points
        if self._p < 1:
            raise ExceptionError(
                name="TrainingSet::build():",
                message="Empty training set"
            )
        if not self._ready:
            # Compute the number of varying input and output
            self._nvar, self._X_nbdiff = self.compute_nbdiff(self._X)
            self._mvar, self._Z_nbdiff = self.compute_nbdiff(self._Z)
            self._X_nbdiff1 = 0
            self._X_nbdiff2 = 0
            for j in range(self._n):
                if self._X_nbdiff[j] > 1: self._X_nbdiff1 += 1
                if self._X_nbdiff[j] > 2: self._X_nbdiff2 += 1

            # Check singular data (inf and void)
            self.check_singular_data()
            # Compute bounds over columns of X and Z
            self.compute_bounds()
            # Compute scaling values
            self.compute_scaling()
            # Compute scaled matrices
            self.compute_scaled_matrices()
            # Build matrix of distances between each pair of points
            self.compute_Ds()
            # Compute fs_min
            self.compute_f_min()
            # The training set is now ready!
            self._ready = True

        # _bbo is considered as defined. It can not be modified anymore.
        self._bbo_is_def = True

    def get_distances_norm1(self, A: DataSet, B: DataSet):
        n: int = A.ncols
        if B.ncols != n:
            raise ExceptionError(name="DataSet::get_distances_norm2::", message="dimension error")
        pa: int = A.nrows
        pb: int = B.nrows
        D: DataSet = DataSet(name="D", nrows=pa, ncols=pb)
        v: float
        ia: int
        ib: int
        j: int
        for ia in range(pa):
            for ib in range(pb):
                # Distance between the point ia of the cache and the point ib of the matrix XXs
                v = 0
                for j in range(n):
                    v += abs(A.points_get(ia, j) - B.points_get(ib, j))
                D.points[ia][ib] = v
        return D

    def get_distances_norm2(self, A: DataSet, B: DataSet):
        n: int = A.ncols
        if B.ncols != n:
            ExceptionError(name="DataSet::get_distances_norm2::", message="dimension error")

        pa: int = A.nrows
        pb: int = B.nrows
        D: DataSet = DataSet(name="D", nrows=pa, ncols=pb)
        v: float
        ia: int
        ib: int
        j: int

        for ia in range(pa):
            for ib in range(pb):
                # Distance between the point ia of the cache and the point ib of the matrix XXs
                v = 0
                for j in range(n):
                    d = A.points_get(ia, j) - B.points_get(ib, j)
                    v += d * d
                D.points[ia][ib] = np.sqrt(v)
        return D

    def get_distances_norminf(self, A: DataSet, B: DataSet):
        n: int = A.ncols
        if B.ncols != n:
            ExceptionError(name="DataSet::get_distances_norm2::", message="dimension error")

        pa: int = A.nrows
        pb: int = B.nrows
        D: DataSet = DataSet(name="D", nrows=pa, ncols=pb)
        v: float
        ia: int
        ib: int
        j: int

        for ia in range(pa):
            for ib in range(pb):
                # Distance between the point ia of the cache and the point ib of the matrix XXs
                v = 0
                for j in range(n):
                    v = max(v, abs(A.points_get(ia, j) - B.points_get(ib, j)))
                D.points[ia][ib] = v
        return D

    def X_scale_xindex(self, x: float, var_index: int):
        return self._X_scaling_a[var_index]*x + self._X_scaling_b[var_index]

    def X_scale_x(self, x: List[float]):
        for j in range(self._n):
            x[j] = self._X_scaling_a[j]*x[j] + self._X_scaling_b[j]

    def X_unscale_x(self, y: List[float]):
        for j in range(self._n):
            y[j] = (y[j]-self._X_scaling_b[j])/self._X_scaling_a[j]

    def X_unscale_xindex(self, y: float, var_index: int):
        return (y-self._X_scaling_b[var_index])/self._X_scaling_a[var_index]


    def Z_scale_z(self, z: List[float]):
        for j in range(self._m):
            z[j] = self._Z_scaling_a[j]*z[j] + self._Z_scaling_b[j]

    def Z_scale_zindex(self, z: float, var_index: int):
        return self._Z_scaling_a[var_index]*z + self._Z_scaling_b[var_index]

    def Z_unscale_w(self, w: List[float]):
        for j in range(self._n):
            w[j] = self.Z_unscale_windex(w[j], j)

    def Z_unscale_windex(self, w: float, j: int):
        if constants.boolean_rounding and self._Z_nbdiff[j] == 2:
            Zs_middle: float
            if constants.boolean_rounding == 1:
                Zs_middle = self.Z_scale((self._Z_ub[j]+self._Z_lb[j])/2., j)
            elif constants.boolean_rounding == 2:
                Zs_middle = self._Zs_mean[j]
            return Zs_middle
        else:
            return (w - self._Z_scaling_b[j]) / self._Z_scaling_a[j]


    def ZE_unscale(self, w: float, j: int):
        """ /*------------------------------------------*/
            /*    ZE unscale: w->z: z = (w)/a           */
            /* Used to unscale errors, std and EI       */
            /*------------------------------------------*/ """
        return w/self._Z_scaling_a[j]

    def X_scale_DataSet(self, X: DataSet):
        p: int = X.nrows
        n: int = X.ncols
        if n != self._n:
            raise ExceptionError(name="TrainingSet::TrainingSet():", message="Diemension error")
        v: float
        for i in range(p):
            for j in range(n):
                v = X.points_get(i,j)
                v = self.X_scale_xindex(x=v, var_index=j)
                X.points_set(i,j,v)
        return X

    def Z_unscale_DataSet(self, Z: DataSet):
        p = Z.nrows
        m = Z.ncols
        if m != self._m:
            raise ExceptionError(name="TrainingSet::TrainingSet():", message="Diemension error")

        v: float
        for i in range(p):
            for j in range(m):
                v = Z.points_get(i, j)
                v = self.Z_unscale_windex(w=v, j=j)
                Z.points_set(i, j, v)
        return Z

    def ZE_unscale_DataSet(self, ZE:DataSet):
        p = ZE.nrows
        m = ZE.ncols
        if m != self._m:
            raise ExceptionError(name="TrainingSet::TrainingSet():", message="Diemension error")

        v: float
        for i in range(p):
            for j in range(m):
                v = ZE.points_get(i, j)
                v = self.Z_unscale_windex(w=v, j=j)
                ZE.points_set(i, j, v)
        return ZE

    def get_d1_over_d2(self, XXs: DataSet):
        if XXs.nrows:
            raise ExceptionError(name="TrainingSet::get_d1_over_d2:", message=" XXs must have only one line.")
        d1: float = np.inf
        d2: float = np.inf
        d: float
        dxj: float
        i: int
        i1: int
        j: int
        i1 = 0 # -> Index of the closest point
        # If only 1 point, it is not possible to compute d2,
        # so we use a dummy value.
        if self._p==1:
            return 1.0
        # Parcours des points
        d =0.
        for i in range(self._p):
            # calculate d
            d = 0.
            for j in range(self._n):
                dxj = XXs.points_get(0, j) - self._Xs.points_get(i, j)
                d += dxj*dxj
            if d == 0:
                return 0.
            if d < d1:
                d2=d1
                d1=d
                i1=i
            elif d < d2 and self._Ds.points_get(i, i1)>0:
                d2 = d
        return np.sqrt(d1/d2)

    def get_d1(self, XXs:DataSet):
        if XXs.nrows:
            raise ExceptionError(name="TrainingSet::get_d1_over_d2:", message=" XXs must have only one line.")
        d1: float = np.inf
        d: float
        dxj: float
        i: int
        j: int
        for i in range(self._p):
            # calculate d
            d = 0.
            for j in range(self._n):
                dxj = XXs.points_get(0, j) - self._Xs.points_get(i, j)
                d += dxj*dxj
            if d == 0:
                return 0.
            if d < d1:
                d1=d

        return np.sqrt(d1)

    def get_exclusion_area_penalty(self, XXs: DataSet, tc: float):
        pxx: int = XXs.nrows
        r12: float
        p: float

        # //double logtc = log(tc);
        #
        #   // tc = 0 => no penalty
        #   // tc > 0 => infinite penalty for points of the cache
        #   // Small value of tc (close to 0) => penalty is null nearly everywhere
        #   // Large value of tc (close to 1) => penalty is non null nearly everywhere

        P: DataSet = DataSet()
        P.name = "P"
        P.nrows = pxx
        P.ncols = 1
        for i in range(pxx):
            temp: DataSet = DataSet(name=XXs.name, nrows = 1, ncols = XXs.ncols)
            temp.points = XXs.points[i]
            r12 = self.get_d1_over_d2(temp)
            if r12 < tc:
                p = 1e9 - r12
            else:
                p = 0.
            P.points_set(i,0,p)
        return P

    def get_distance_to_closest(self, XXs:DataSet):
        pxx: int = XXs.nrows
        d: float
        P: DataSet = DataSet()
        P.name = "P"
        P.nrows = pxx
        P.ncols = 1
        for i in range(pxx):
            temp: DataSet = DataSet(name=XXs.name, nrows = 1, ncols = XXs.ncols)
            temp.points = XXs.points[i]
            d = self.get_d1(temp)
            P.points_set(i,0,d)
        return P

    def select_greedy(self, X: DataSet, imin: int, pS: int, lambda0: float, dt: int):
        # /*--------------------------------------*/
        # /*       select points                  */
        # /*--------------------------------------*/
        p = X.nrows
        n = X.ncols

        if pS<3 or pS>=p:
            raise ExceptionError(name="TrainingSet::TrainingSet(): ", message="wrong value of pS")

        S: List[int] = []
        inew: int
        xnew: DataSet = DataSet(name="xnew", nrows = 1, ncols = n)
        x: DataSet = DataSet(name="x", nrows = 1, ncols = n)

        xnew.points = X.get_row(imin)
        dB: DataSet = self.get_distances(X, xnew, dt)
        dB.name = "dB"
        S.append(imin)
        inew = dB.get_max_index()
        xnew.points = X.get_row(inew)
        dS: DataSet = self.get_distances(X, xnew, dt)
        S.append(inew)

        #   // As B is in S, we can take the min of both distances
        dS: DataSet = DataSet().min(dS, dB)
        Lambda: float = 0.
        if lambda0 != 0:
            for i in range(p):
                if dB.get_element(i):
                    Lambda = max(Lambda, dS.get_element(i)/dB.get_element(i))
            Lambda *= lambda0

        while len(S) < pS:
            temp: DataSet = DataSet()
            temp.points = np.subtract(dS.points, np.multiply(Lambda, dB.points))
            inew = temp.get_max_index()

            if dS.get_element(inew)==0:
                Lambda *= 0.99
                if Lambda<1e-6:
                    break
            else:
                S.append(inew)
                xnew.points = X.get_row(inew)
                dS = DataSet().min(dS, self.get_distances(X, xnew, dt))
                dS.name = "dS"
        return dS


    def DISTANCE_NORM2_IS0(self, A: DataSet, B: DataSet):
        n: int = A.ncols
        if B.ncols != n:
            ExceptionError(name="DataSet::get_distances_norm2::", message="dimension error")

        pa: int = A.nrows
        pb: int = B.nrows
        D: DataSet = DataSet(name="D", nrows=pa, ncols=pb)
        v: float
        ia: int
        ib: int
        j: int
        x0: List[float] = [0]*n

        for j in range(n):
            # TODO: scaling ...
            x0[j] = self.X_scale_xindex(0., j)

        for ia in range(pa):
            for ib in range(pb):
                # Distance between the point ia of the cache and the point ib of the matrix XXs
                v = 0
                for j in range(n):
                    v = max(v, abs(A.points_get(ia, j) - B.points_get(ib, j)))
                D.points[ia][ib] = v
        return D

    def get_distances(self, A: DataSet, B: DataSet, dt: int) -> DataSet:
        if dt == distance_t.DISTANCE_NORM1:
            return self.get_distances_norm1(A, B)
        elif dt == distance_t.DISTANCE_NORM2:
            return self.get_distances_norm2(A, B)
        elif dt == distance_t.DISTANCE_NORMINF:
            return self.get_distances_norminf(A, B)
        elif dt == distance_t.DISTANCE_NORM2_IS0:
            return self.DISTANCE_NORM2_IS0(A, B)

    def get_bbo(self, j:int):
        self.check_ready()
        return self._bbo[j]



@dataclass
class Surrogate_Parameters(Kernel):
    NB_KERNEL_TYPES = 11
    NB_DECREASING_KERNEL_TYPES = 6
    _type: int
    _degree: int
    _degree_status: int
    _kernel_type_status: int
    _kernel_coef_status: int
    _ridge_status: int
    _weight: DataSet
    _weight_type: weight_t
    _covariance_coef: DataSet
    _preset: str
    _output: str
    _nb_parameter_optimization: int
    _list_AP = ["DEGREE", "RIDGE", "KERNEL_TYPE", "KERNEL_COEF",
                "DISTANCE_TYPE", "WEIGHT_TYPE", "TYPE", "OUTPUT", "METRIC_TYPE", "PRESET", "BUDGET"]
    _budget: int = 100
    _metric_type: int = metric_t.METRIC_AOECV
    _covariance_coef_status: int = param_status_t.STATUS_FIXED
    _weight_status: int = param_status_t.STATUS_MODEL_DEFINED
    _distance_type: int = distance_t.DISTANCE_NORM2
    _distance_type_status: int = param_status_t.STATUS_FIXED
    _ridge: float = 0.001
    _kernel_coef: float = 1.
    _kernel_type: int = kernel_t.KERNEL_D1

    def set_defaults(self):
        """ Set defaults """
        if self.type in (model_t.LINEAR, model_t.TGP, model_t.SVN):
            ExceptionError(name="Surrogate_Parameters::set_defaults::", message="Not implemented yet!")
        elif self.type is model_t.KRIGING:
            self.distance_type = distance_t.DISTANCE_NORM2
            self.distance_type_status = param_status_t.STATUS_FIXED
            self.ridge = 1E-16
            self.ridge_status = param_status_t.STATUS_OPTIM
            self.covariance_coef.name = "COVARIANCE_COEF"
            self.covariance_coef.nrows = 1
            self.covariance_coef.ncols = 2
            self.covariance_coef.points_set(0, 0, 2)
            self.covariance_coef.points_set(0, 1, 2)
            self.covariance_coef_status = param_status_t.STATUS_OPTIM
            return
        elif self.type in (model_t.PRS, model_t.PRS_EDGE, model_t.PRS_CAT):
            self.degree = 2
            self.degree_status = param_status_t.STATUS_FIXED
            self.ridge = 0.001
            self.ridge_status = param_status_t.STATUS_FIXED
            return
        elif self.type is model_t.KS:
            self.kernel_type = kernel_t.KERNEL_D1
            self.kernel_type_status = param_status_t.STATUS_FIXED
            self.kernel_coef = 5
            self.kernel_coef_status = param_status_t.STATUS_OPTIM
            self.distance_type = distance_t.DISTANCE_NORM2
            self.distance_type_status = param_status_t.STATUS_FIXED
            return
        elif self.type is model_t.RBF:
            self.kernel_type = kernel_t.KERNEL_I2
            self.kernel_type_status = param_status_t.STATUS_FIXED
            self.kernel_coef = 1
            self.kernel_coef_status = param_status_t.STATUS_OPTIM
            self.distance_type = distance_t.DISTANCE_NORM2
            self.distance_type_status = param_status_t.STATUS_FIXED
            self.ridge = 0.001
            self.ridge_status = param_status_t.STATUS_FIXED
            self.preset = "I"
        elif self.type is model_t.LOWESS:
            self.kernel_type = kernel_t.KERNEL_D1
            self.kernel_type_status = param_status_t.STATUS_FIXED
            self.kernel_coef = 1.
            self.kernel_coef_status = param_status_t.STATUS_OPTIM
            self.distance_type = distance_t.DISTANCE_NORM2
            self.distance_type_status = param_status_t.STATUS_FIXED
            self.degree = 2
            self.degree_status = param_status_t.STATUS_FIXED
            self.ridge = 0.001
            self.ridge_status = param_status_t.STATUS_FIXED
            self.preset = "DGN"
            return
        elif self.type is model_t.ENSEMBLE:
            self.weight_type = weight_t.WEIGHT_SELECT
            self.weight_status = param_status_t.STATUS_MODEL_DEFINED
            self.preset = "DEFAULT"
            return
        elif self.type is model_t.CN:
            return
        else:
            ExceptionError(name="SURROGATE_PARAMETERS::Set_defaults::type", message="Undefined type")
        # Default output
        self.output = "NULL"

    def get_x(self) -> DataSet:
        """ Creates a vector that contains the numerical values of all the parameters that must be optmized. """
        X: DataSet = DataSet()
        X.name = "X"
        X.nrows = 1
        X.ncols = self.nb_parameter_optimization
        k: int = 0
        if self.degree_status is param_status_t.STATUS_OPTIM:
            k += 1
            X.points_set(0, k, float(self.degree))
        if self.ridge_status is param_status_t.STATUS_OPTIM:
            k += 1
            X.points_set(0, k, float(self.ridge))
        if self.kernel_coef_status is param_status_t.STATUS_OPTIM:
            k += 1
            X.points_set(0, k, float(self.kernel_coef))
        if self.kernel_type_status is param_status_t.STATUS_OPTIM:
            k += 1
            X.points_set(0, k, float(self.kernel_type))
        if self.distance_type_status is param_status_t.STATUS_OPTIM:
            k += 1
            X.points_set(0, k, float(self.distance_type))
        if self.covariance_coef_status is param_status_t.STATUS_OPTIM:
            for j in range(self.covariance_coef.ncols):
                k += 1
                X.points_set(0, k, self.covariance_coef.points_get(0, j))
        if self.weight_status is param_status_t.STATUS_OPTIM:
            for i in range(self.weight.nrows):
                for j in range(self.weight.ncols):
                    k += 1
                    X.points_set(0, k, self.weight.points_get(i, j))
        if k != self.nb_parameter_optimization:
            print("k = " + str(k))
            print("_nb_parameter_optimization = " + str(self.nb_parameter_optimization))
            ExceptionError(name="Surrogate_Paramaeters:get_x::", message="Inconcistency in the value of k.")
        return X

    def set_x(self, X: DataSet):
        """ Set the parameters from an external value of x """
        k: int = 0
        if self.degree_status is param_status_t.STATUS_OPTIM:
            k += 1
            self.degree = int(X.get_element(k))
        if self.ridge_status is param_status_t.STATUS_OPTIM:
            k += 1
            self.ridge = int(X.get_element(k))
        if self.kernel_coef_status is param_status_t.STATUS_OPTIM:
            k += 1
            self.kernel_coef = int(X.get_element(k))
        if self.kernel_type_status is param_status_t.STATUS_OPTIM:
            k += 1
            self.kernel_type = int(X.get_element(k))
        if self.distance_type_status is param_status_t.STATUS_OPTIM:
            k += 1
            self.distance_type = int(X.get_element(k))
        if self.covariance_coef_status is param_status_t.STATUS_OPTIM:
            for j in range(self.covariance_coef.ncols):
                k += 1
                self.covariance_coef.points_set(0, j, X.get_element(k))
        if self.weight_status is param_status_t.STATUS_OPTIM:
            for i in range(self.weight.nrows):
                for j in range(self.weight.ncols):
                    k += 1
                    self.weight.points_set(i, j, X.get_element(k))
            self.weight.normalize_cols()

        if k != self.nb_parameter_optimization:
            print("k = " + str(k))
            print("_nb_parameter_optimization = " + str(self.nb_parameter_optimization))
            ExceptionError(name="Surrogate_Paramaeters:get_x::", message="Inconcistency in the value of k.")

    def get_x_bounds(self, LB: DataSet, UB: DataSet, domain: List[int], logscale: List[bool]):
        """ /*------------------------------------------------------------------------------------*/
            /*  Parameter domains and definitions                                                 */
            /* Defines the bounds, domain and log-scale for each parameter that must be optimized */
            /*------------------------------------------------------------------------------------*/ """
        if not LB.is_initialized() or not UB.is_initialized() or domain is None or logscale is None:
            ExceptionError(name="SP::get_x_bounds::", message="Uninitialized bounds")

        N = self.nb_parameter_optimization
        for i in range(N):
            logscale.append(False)
        k = 0
        # DEGREE
        if self.degree_status is param_status_t.STATUS_OPTIM:
            LB.points_set(0, k, 0)
            if self.type is model_t.LOWESS:
                UB.points_set(0, k, 2)
                domain[k] = param_domain_t.PARAM_DOMAIN_INTEGER
            else:
                UB.points_set(0, k, 6)
                domain[k] = param_domain_t.PARAM_DOMAIN_INTEGER
            k += 1
        # RIDGE
        if self.ridge_status is param_status_t.STATUS_OPTIM:
            LB.points_set(0, k, 1e-16)
            UB.points_set(0, k, 1e-1)
            domain[k] = param_domain_t.PARAM_DOMAIN_CONTINUOUS
            logscale[k] = True
            k += 1
        # KERNEL_COEF
        if self.kernel_coef_status is param_status_t.STATUS_OPTIM:
            LB.points_set(0, k, 1e-2)
            UB.points_set(0, k, 100)
            domain[k] = param_domain_t.PARAM_DOMAIN_CONTINUOUS
            logscale[k] = True
            k += 1
        # KERNEL_TYPE
        if self.kernel_type_status is param_status_t.STATUS_OPTIM:
            LB.points_set(0, k, 0)
            if self.type is model_t.RBF:
                UB.points_set(0, k, float(self.NB_KERNEL_TYPES - 1))
                domain[k] = param_domain_t.PARAM_DOMAIN_CAT
                k += 1
        # COVARIANCE COEF
        if self.covariance_coef_status is param_status_t.STATUS_OPTIM:
            v: int = int(self.covariance_coef.ncols / 2)
            for j in range(v):
                # Exponent parameter
                LB.points_set(0, k, 0.5)
                UB.points_set(0, k, 3.0)
                domain[k] = param_domain_t.PARAM_DOMAIN_CONTINUOUS
                logscale[k] = False
                k += 1
                # Factor parameter
                LB.points_set(0, k, 1e-3)
                UB.points_set(0, k, 1)
                domain[k] = param_domain_t.PARAM_DOMAIN_CONTINUOUS
                logscale[k] = True
                k += 1
        # WEIGHT
        if self.weight_status is param_status_t.STATUS_OPTIM:
            for i in range(self.weight.nrows):
                for j in range(self.weight.ncols):
                    LB.points_set(0, k, 0)
                    UB.points_set(0, k, 1)
                    domain[k] = param_domain_t.PARAM_DOMAIN_CONTINUOUS
                    logscale[k] = False
                    k += 1
        if k != N:
            ExceptionError(name="Surrogate_Paramaeters:get_x_bounds::", message="Inconcistency in the value of k.")
        error: bool = False
        for j in range(N):
            # Check bounds order
            if LB.get_element(j) >= UB.get_element(j):
                error = True
                print("Variable " + str(j) + " LB (= " + str(LB.get_element(j)) + ") >= UB (= "
                      + str(UB.get_element(j)) + ")")
            # Check that only continuous variables are using a log scale
            if logscale[j] and domain[j] != param_domain_t.PARAM_DOMAIN_CONTINUOUS:
                error = True
                print("Variable " + str(j) + " Uses logscale and is not continuous.")
            # Check that variables with log scale have bounds of the same sign.
            if logscale[j]:
                if LB.get_element(j) * UB.get_element(j) <= 0:
                    print("The bounds are not appropriate for logscale optimization.")
            # Check domain types
            if domain[j] is param_domain_t.PARAM_DOMAIN_CONTINUOUS:
                continue
            if domain[j] in (param_domain_t.PARAM_DOMAIN_INTEGER, param_domain_t.PARAM_DOMAIN_CAT):
                if float(round(LB.get_element(j))) != LB.get_element(j):
                    error = True
                    print("Variable " + str(j) + ": LB is not an integer")
                if float(round(UB.get_element(j))) != UB.get_element(j):
                    error = True
                    print("Variable " + str(j) + ": UB is not an integer")
            if domain[j] is param_domain_t.PARAM_DOMAIN_BOOL:
                if LB.get_element(j) == 1:
                    error = True
                    print("Variable " + str(j) + " LB is not 0")
                if UB.get_element(j) == 0:
                    error = True
                    print("Variable " + str(j) + " UB is not 1")
            if domain[j] is param_domain_t.PARAM_DOMAIN_MISC:
                error = True
                print("Variable " + str(j) + " is MISC")

        if error:
            ExceptionError(name="SP::get_x_bounds::", message="Error in definition of LB, UB or domain!")
        return LB, UB, domain, logscale

    def check_x(self) -> bool:
        """ Check parameter X """
        X: DataSet = self.get_x()
        error: bool = False
        # Check dimension of X
        if X.nrows != 1:
            error = True
            print("Number of rows is not 1")
        N: int = self.nb_parameter_optimization
        if X.ncols != N:
            error = True
            print("Number of cols is not consistent with _nb_parameter_optimization")

        # Get bound info
        LB: DataSet = DataSet(name="LB", nrows=1, ncols=N)
        UB: DataSet = DataSet(name="UB", nrows=1, ncols=N)
        d: List[int] = [] * N
        ls: List[bool] = [] * N
        domain, logscale = self.get_x_bounds(LB, UB, d, ls)
        for j in range(self.nb_parameter_optimization):
            # Check bounds
            if X.get_element(j) < LB.get_element(j):
                error = True
                print("X at " + str(j) + " < LB")
            if X.get_element(j) > UB.get_element(j):
                error = True
                print("X at " + str(j) + " > UB")
            # Check types
            if domain[j] in (param_domain_t.PARAM_DOMAIN_CONTINUOUS, param_domain_t.PARAM_DOMAIN_INTEGER,
                             param_domain_t.PARAM_DOMAIN_CAT):
                if float(round(X.get_element(j) != X.get_element(j))):
                    error = True
                    print("Variable " + str(j) + ": LB is not an integer")

            if domain[j] is param_domain_t.PARAM_DOMAIN_BOOL:
                if X.get_element(j) == 1:
                    error = True
                    print("Variable " + str(j) + " is not BOOLEAN")

            if domain[j] is param_domain_t.PARAM_DOMAIN_MISC:
                error = True
                print("Variable " + str(j) + " is MISC")

        # Check dimension of _covariance_coef
        if self.covariance_coef.nrows > 1:
            error = True
            print(" Covariance_coef should have only one row.")
        if error:
            ExceptionError(name="SP::check::", message="Invalid X!")

        return not error

    def get_x_penalty(self) -> float:

        pen: float = 0.
        if self.degree_status is param_status_t.STATUS_OPTIM:
            pen += self.degree
        if self.ridge_status is param_status_t.STATUS_OPTIM:
            pen += np.log(self.ridge)
        if self.kernel_coef_status is param_status_t.STATUS_OPTIM:
            pen += np.log(self.kernel_coef)
        if self.distance_type_status is param_status_t.STATUS_OPTIM:
            if self.distance_type is distance_t.DISTANCE_NORM2:
                pen += 0
            if self.distance_type in (distance_t.DISTANCE_NORM1, distance_t.DISTANCE_NORMINF):
                pen += 1
            if self.distance_type in (distance_t.DISTANCE_NORM2_IS0, distance_t.DISTANCE_NORM2_CAT):
                pen += 10
        if self.covariance_coef_status is param_status_t.STATUS_OPTIM:
            v: int = int(self.covariance_coef.ncols / 2)
            ip: int = 0
            for i in range(v):
                ip += 1
                # Exponent (the larger, the smoother)
                pen -= self.covariance_coef.get_element(ip)
                # Factor (the smaller, the smoother)
                ip += 1
                pen += np.log(self.covariance_coef.get_element(ip))
        if self.weight_status == param_status_t.STATUS_OPTIM:
            wij: float
            for i in range(self.weight.nrows):
                for j in range(self.weight.ncols):
                    wij = self.weight.points_get(i, j)
                    pen += wij * wij
        if np.isinf(pen): pen = np.inf
        if np.isnan(pen): pen = np.inf

        return pen

    def update_covariance_coef(self, v: int):
        """ /*-------------------------------------------------*/
            /*  update the dimension of _covariance_parameter  */
            /*  for Kriging models                             */
            /*-------------------------------------------------*/
            // The matrix containing the covariance coefficients, for Kriging models,
            // is initialized with 2 components (factor and exponent).
            // In sgtelib 2.0.1, the factor and exponent are the same for all input variables.
            // However, it would be possible to use a different value of factor and exponent
            // for each variable (which is the case in most Kriging implementation).
            // The problem is that the Surrogate_Parameters class is built before the model itself,
            // and the instance Surrogate_Parameters is not able to retrieve the dimension of
            // the input space from either the training set or the model. The following method allows
            // the model to tell to the Surrogate_Parameters what must be the dimension of
            // the set of covariance coefficients.
            // This function is not used for now, but might be used in future versions of sgtelib. """
        v0: int = int(self.covariance_coef.ncols / 2)
        if v < v0: ExceptionError(name="SP::update_covariance_coef::", message="v < v0")
        if v0 == v: return

        # Filling values
        factor_mean: float = 0.
        exponent_mean: float = 0.
        # Compute the mean value for exponent and factor.
        k: int = 0
        for i in range(v0):
            k += 1
            exponent_mean += self.covariance_coef.get_element(k)
            k += 1
            factor_mean += self.covariance_coef.get_element(k)
        exponent_mean /= v0
        factor_mean /= v0
        # Create additional columns
        Add: DataSet = DataSet(name="Add", nrows=1, ncols=2)
        Add.points_set(0, 0, exponent_mean)
        Add.points_set(0, 1, factor_mean)

        for i in range(v - v0):
            self.covariance_coef.add_cols(Add)

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        self._type = value

    @property
    def degree(self):
        return self._degree

    @degree.setter
    def degree(self, value):
        self._degree = value

    @property
    def degree_status(self):
        return self._degree_status

    @degree_status.setter
    def degree_status(self, value):
        self._degree_status = value

    @property
    def kernel_type(self):
        return self._kernel_type

    @kernel_type.setter
    def kernel_type(self, value):
        self._kernel_type = value

    @property
    def kernel_type_status(self):
        return self._kernel_type_status

    @kernel_type_status.setter
    def kernel_type_status(self, value):
        self._kernel_type_status = value

    @property
    def kernel_coef(self):
        return self._kernel_coef

    @kernel_coef.setter
    def kernel_coef(self, value):
        self._kernel_coef = value

    @property
    def kernel_coef_status(self):
        return self._kernel_coef_status

    @kernel_coef_status.setter
    def kernel_coef_status(self, value):
        self._kernel_coef_status = value

    @property
    def ridge(self):
        return self._ridge

    @ridge.setter
    def ridge(self, value):
        self._ridge = value

    @property
    def ridge_status(self):
        return self._ridge_status

    @ridge_status.setter
    def ridge_status(self, value):
        self._ridge_status = value

    @property
    def distance_type(self):
        return self._distance_type

    @distance_type.setter
    def distance_type(self, value):
        self._distance_type = value

    @property
    def distance_type_status(self):
        return self._distance_type_status

    @distance_type_status.setter
    def distance_type_status(self, value):
        self._distance_type_status = value

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value):
        self._weight = value

    @property
    def weight_type(self):
        return self._weight_type

    @weight_type.setter
    def weight_type(self, value):
        self._weight_type = value

    @property
    def weight_status(self):
        return self._weight_status

    @weight_status.setter
    def weight_status(self, value):
        self._weight_status = value

    @property
    def covariance_coef(self):
        return self._covariance_coef

    @covariance_coef.setter
    def covariance_coef(self, value):
        self._covariance_coef = value

    @property
    def covariance_coef_status(self):
        return self._covariance_coef_status

    @covariance_coef_status.setter
    def covariance_coef_status(self, value):
        self._covariance_coef_status = value

    @property
    def metric_type(self):
        return self._metric_type

    @metric_type.setter
    def metric_type(self, value):
        self._metric_type = value

    @property
    def preset(self):
        return self._preset

    @preset.setter
    def preset(self, value):
        self._preset = value

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, value):
        self._output = value

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, value):
        self._output = value

    @property
    def budget(self):
        return self._budget

    @budget.setter
    def budget(self, value):
        self._budget = value

    @property
    def nb_parameter_optimization(self):
        return self._nb_parameter_optimization

    @nb_parameter_optimization.setter
    def nb_parameter_optimization(self, value):
        self._nb_parameter_optimization = value

    def Surrogate_Parameters(self, mt):
        """ Use a model_type to define the model, then use default values
        associated to this specific model_type, or Use a string provided by the user.
        Read the model_type in the string, then parse the string to read all the other parameters.
        """
        if isinstance(mt, int):
            self._type = mt
            self.set_defaults()
            self.check()
        elif isinstance(mt, str):
            self._type = self.read_model_type(mt)
            self.set_defaults()
            self.read_string(mt)
            self.check()
        else:
            raise ExceptionError(
                name="Surrogate parameters: model type",
                message="The provided model " + str(mt) + " has an invalid type!"
            )

    def read_model_type(self, model_description: str) -> int:
        """ Extract the model type from a string """
        if model_description.islower():
            model_description.upper()
        return getattr(model_t, model_description)

    def is_authorized_optim(self, field: str) -> bool:
        """ Indicate if the field given in input can be optimized """

        for f in self._list_AP:
            if (f == field):
                return True
        Exception("Undefined field")
        return False

    def check(self):
        """ Verify the consistency of the set of parameters """
        E = "Surrogate_Parameters::check::"
        if self.type is model_t.SVN:
            ExceptionError(name=E + "TYPE", message="Not implemented yet!")
        elif self.type is model_t.PRS_CAT:
            if self.degree < 0:
                ExceptionError(name=E + "DEGREE", message="degree must be >= 0")
            elif self.ridge < 0:
                ExceptionError(name=E + "RIDGE", message="ridge must be >= 0")
        elif self.type is model_t.KRIGING:
            if self.kernel_is_decreasing(self.kernel_type):
                ExceptionError(name=E + "Non-decreasing_Kernel", message="kernel_type must be decreasing")
        elif self.type is model_t.KS:
            if self.kernel_coef <= 0:
                ExceptionError(name=E + "Non-decreasing_Kernel", message="kernel_coef must be > 0")
            if self.kernel_is_decreasing(self.kernel_type):
                ExceptionError(name=E + "Non-decreasing_Kernel", message="kernel_type must be decreasing")
        elif self.type is model_t.RBF:
            if self.kernel_coef <= 0:
                ExceptionError(name=E + "Non-decreasing_Kernel", message="kernel_coef must be > 0")
            if self.ridge <= 0:
                ExceptionError(name=E + "RBF::RIDGE", message="ridge must be >= 0")
            if not self.kernel_has_parameter(self.kernel_type) \
                    and (self.kernel_type_status is param_status_t.STATUS_FIXED):
                self.kernel_coef = 1
                self.kernel_coef_status = param_status_t.STATUS_FIXED
        elif self.type is model_t.LOWESS:
            if self.degree < 0 or self.degree > 2:
                ExceptionError(name=E + "LOWESS::DEGREE", message="degree for LOWESS model must be 0, 1 or 2")
            if self.ridge <= 0:
                ExceptionError(name=E + "RBF::RIDGE", message="ridge must be >= 0")
            # // The default preset for LOWESS models is DGN.
            #       // The preset defines how the weight of each data point is computed.
            #       // D : w_i = phi(distance_i), where distance_i is the distance between the prediction point
            #       //           and the data point x_i.
            #       // DEN : w_i = phi(distance_i/dq_i), where dq_i is the distance between the prediction point
            #       //             and the q^th closest data point, and dq_i is computed with empirical method.
            #       // DGN : w_i = phi(distance_i/dq_i), where dq_i is computed with gamma method.
            #       // RE : w_i = phi(rank_i), where rank_i is the rank of x_i in terms of distance
            #       //            to the prediction point, and the rank_i is computed with empirical method.
            #       // RG : w_i = phi(rank_i), where the rank is computed with gamma method.
            #       // REN : w_i = same as RE but the ranks are normalized in [0,1]
            #       // RGN : w_i = same as RG but the ranks are normalized in [0,1]
            ps = ["D", "DEN", "DGN", "RE", "RG", "REN", "RGN"]
            for f in ps:
                if (f == self.preset):
                    ExceptionError(name=E + "LOWESS::PRESET", message="preset not recognized")
            if self.kernel_is_decreasing(self.kernel_type):
                ExceptionError(name=E + "Non-decreasing_Kernel", message="kernel_type must be decreasing")
        elif self.type is not model_t.ENSEMBLE and self.type is not model_t.CN:
            ExceptionError(name=E + "Model type", message="Undefined type")

        # Count the number of parameters to optimize
        self.nb_parameter_optimization = 0
        if self.degree_status is param_status_t.STATUS_OPTIM: self.nb_parameter_optimization += 1
        if self.kernel_type_status is param_status_t.STATUS_OPTIM: self.nb_parameter_optimization += 1
        if self.kernel_coef_status is param_status_t.STATUS_OPTIM: self.nb_parameter_optimization += 1
        if self.ridge_status is param_status_t.STATUS_OPTIM: self.nb_parameter_optimization += 1
        if self.distance_type_status is param_status_t.STATUS_OPTIM: self.nb_parameter_optimization += 1
        if self.covariance_coef_status is param_status_t.STATUS_OPTIM: self.nb_parameter_optimization += \
            self.covariance_coef.ncols * self.covariance_coef.nrows
        if self.weight_status is param_status_t.STATUS_OPTIM: self.nb_parameter_optimization += \
            self.weight.nrows * self.weight.ncols

    def display(self):
        """ Display outputs """

    # class Surrogate(pydantic.BaseModel, surrogate_utils):





@dataclass
class Surrogate(metric_t(), surrogate_utils()):
    """ Surrogate model construction """
    _param: Surrogate_Parameters
    _trainingSet: TrainingDataSet = TrainingDataSet()
    _metrics: Optional[Dict[int, DataSet]] = Dict[int, DataSet]
    _out: Optional[DataSet] = DataSet()
    _selected_points: List[Any] = field(default_factory=[1, -1])
    _n: int = _trainingSet.get_input_dim()
    _m: int = _trainingSet.get_output_dim()
    _p_ts: int = 0
    _p_ts_old: int = 999999999
    _p: int = 0
    _p_old: int = 999999999
    _ready: bool = False
    _Zhs: float = None
    _Shs: float = None
    _Zvs: float = None
    _Svs: float = None
    _psize_max: float = 0.5
    _display: bool = False
    _mt_defined: bool = False

    def display(self):
        """ Display """

    # /*=========================================================*/
    # /*=========================================================*/
    # /*||                                                     ||*/
    # /*||             PREDICTION METHODS                      ||*/
    # /*||                                                     ||*/
    # /*=========================================================*/
    # /*=========================================================*/

    # TODO: prediction methods

    def init_private(self):
        return True

    def reset_metrics(self):
        del self._Zhs
        del self._Shs
        del self._Zvs
        del self._Svs
        del self._metrics

    def optimize_parameters(self):
        N: int = self._param.nb_parameter_optimization
        budget: int = N * self._param.budget
        display: bool = False
        if display:
            print("Begin parameter optimization")

        #   //-----------------------------------------
        #   // Bounds, log-scale and domain
        #   //-----------------------------------------
        #   // Lower and upper bound of the parameter
        lb: DataSet = DataSet(name="lb", nrows=1, ncols=N)
        ub: DataSet = DataSet(name="ub", nrows=1, ncols=N)
        #   // Log-scale: if true, then the parameter must be positive and
        #   // will be optimized with a log-scale. This is equivalent to optimizing
        #   // the log of the parameter, instead of optimizing the parameter itself.
        #   // This is very interesting for parameters like the ridge coefficient,
        #   // which can take anywhere between 1e-16 and 1.
        logscale: List[bool] = [False] * N
        #   // The "domain" indicates if the parameter is continuous, integer, boolean,
        #   // categorical, or "MISC".
        #   // MISC parameter should not be optimized.
        domain: List[int] = [0] * N

        #  Interrogating the parameter instance.
        lb, ub, domain, logscale = self._param.get_x_bounds(LB=lb, UB=ub, domain=domain, logscale=logscale)

        #   //-----------------------------------------
        #   // Compute scaling
        #   // The scaling is necessary to compute the magnitude of the poll directions.
        #   //-----------------------------------------
        scaling: DataSet = DataSet(name="scaling", nrows=1, ncols=N)

        for i in range(N):
            if domain[i] == param_domain_t.PARAM_DOMAIN_CONTINUOUS:
                if logscale[i]:
                    d = 1
                else:
                    d = (ub.get_element(i) - lb.get_element(i)) / 5.
                scaling.points_set(0, i, d)
                if d < constants.EPSILON: ExceptionError(name="SP:optimize parameters:", message="Bad scaling!")
            elif domain[i] == param_domain_t.PARAM_DOMAIN_CAT:
                scaling.points_set(0, i, ub.get_element(i) - lb.get_element(i))
            else:
                scaling.points_set(0, i, 1)

        # //-------------------------------------------------------
        #   // Display the information about optimized parameters
        #   //-------------------------------------------------------

        if display:
            print("Model: " + str(self._param.type) + print("lb: [ "))
            # TODO: Complete printing model info
            # if (display){
            #     std::cout << "Model: " << get_short_string() << "\n";
            #     std::cout << "lb: [ ";
            #     for (i=0 ; i<N ; i++) std::cout << lb[i] << " ";
            #     std::cout << "]\n";
            #     std::cout << "ub: [ ";
            #     for (i=0 ; i<N ; i++) std::cout << ub[i] << " ";
            #     std::cout << "]\n";
            #     std::cout << "scaling: [ ";
            #     for (i=0 ; i<N ; i++){
            #       std::cout << scaling[i];
            #       if (logscale[i]) std::cout << "(log)";
            #       std::cout << " ";
            #     }
            #     std::cout << "]\n";
            #   }
        #   //----------------------------------------
        #   // Build set of starting points
        #   //----------------------------------------
        nx0: int = int(budget / 10)
        X0: DataSet = DataSet(name="X0", nrows=nx0, ncols=N)
        use_lh: bool = True
        # Random set of starting points
        for j in range(N):
            lbj = lb.get_element(j)
            ubj = ub.get_element(j)
            d: float = 0.
            for i in range(nx0):
                if use_lh:
                    d = float(i - 1.) / float(nx0 - 2.)
                else:
                    d = np.random.uniform()
            if logscale[j]: d = lb.get_element(j) * (ubj / lbj) ** d
            X0.points_set(i, j, d)
            del d
        if use_lh:
            # Shuffle columns (except first column)
            if N > 1:
                i2: int
                for j in range(N):
                    for i in range(nx0):
                        i2 = i + np.floor(np.random.uniform() * (nx0 - i))
                        if i2 < i or i2 >= nx0:
                            ExceptionError(name="SP::optimize_parameters::", message="Error in permutation indexes!")
                        X0.swap(i, j, i2, j)
        # Add the default values.
        X0.add_rows(self._param.get_x())
        #  //---------------------------------------------
        #   // Budget, poll size, success and objectives
        #   //---------------------------------------------
        xtry: DataSet = DataSet("xtry", 1, N)
        #  // f contains the value of the error metric, returned by the model.
        #   // The smallest f, the better the model.
        #   // fmin is the smallest value of f found so
        fmin: float = np.inf
        # // p is a penalty that allows to chose between
        #   // two sets of parameters that have the same f value.
        #   // pmin is the value of the best set of parameters so far;
        #   // For a given set of parameters, the value of p is returned by the class
        #   // Surrogate_Parameters.
        #   // The penalty is particularly necessary for certain classes of error metrics
        #   // that are piece-wise constant (for example, all the order error metrics).

        eval = {"blackbox": self.eval_objective}
        param = {"baseline": X0,
                 "lb": lb,
                 "ub": ub,
                 "var_names": ["x"]*N,
                 "scaling": 10.0,
                 "post_dir": "./post"}
        options = {"seed": 0, "budget": budget, "tol": 1e-12, "display": False}

        data = {"evaluator": eval, "param": param, "options": options}
        xmin: DataSet = DataSet(name="xmin", nrows=1, ncols=N)
        out = OMADS.main(data)

        self._param.set_x(out["xmin"])
        self._param.check()
        fmin = self.eval_objective()

        # Check for Nan
        if np.isnan(xmin.points) or np.isinf(xmin.points):
            return False

        return True

    def build_private(self) -> bool:
        self._ready = True
        return True

    def one_metric_value_per_bbo(self, mt):
        if mt in [self.METRIC_EMAX,
                  self.METRIC_EMAXCV,
                  self.METRIC_RMSE,
                  self.METRIC_RMSECV,
                  self.METRIC_OE,
                  self.METRIC_OECV,
                  self.METRIC_LINV]:
            return True
        elif mt in [self.METRIC_ARMSE,
                    self.METRIC_ARMSECV,
                    self.METRIC_AOE,
                    self.METRIC_AOECV,
                    self.METRIC_EFIOE,
                    self.METRIC_EFIOECV]:
            return False
        else:
            raise ExceptionError(name="Surrogate::one_metric_value_per_bbo", message="metric")

    def is_metric_defined(self, mt: int):
        # Check if the key exists
        # TODO: check better way for that check
        if self._metrics.get(mt) is None: return False
        # Check the size of the vector
        metric_vector_size: int = len(self._metrics[mt])
        if metric_vector_size<=0: return False
        return True

    def compute_order_error(self, Zpred: DataSet):
        # // Compute the order-efficiency metric by comparing the
        # // values of - _Zs (in the trainingset)
        # //           - Zpred (input of this function)
        # // Put the results in "OE" (output of this function)
        OE: DataSet = DataSet("OE", 1, Zpred.ncols)
        nfails: int
        Zs: DataSet = self.get_matrix_Zs()

        for j in range(self._m):
            if self._trainingSet.get_bbo(j) == bbo_t.BBO_OBJ:
                nfails = 0
                for i1 in range(self._p):
                    z1 = Zs.points_get(i1,j)
                    z1h = Zpred.points_get(i1, j)
                    for i2 in range(self._p):
                        z2 = Zs.points_get(i2, j)
                        z2h = Zpred.points_get(i2, j)
                        if (z1-z2 < 0 or z1h-z2h < 0): nfails += 1
                OE.points_set(0, j, float(nfails)/float(self._p*self._p))
                # ===============================================================
            elif self._trainingSet.get_bbo(j) == bbo_t.BBO_CON:
                nfails = 0
                c0 = self._trainingSet.Z_scale(0., j)
                for i in range(self._p):
                    z1 = Zs.points_get(i, j) - c0
                    z1h = Zpred.points_get(i,j) - c0
                    if z1<0 or z1h<0: nfails +=1
                    OE.points_set(0, j, float(nfails)/float(self._p))
            elif self._trainingSet.get_bbo(j) == bbo_t.BBO_DUM:
                OE.points_set(0, j, -1)

        return OE

    def compute_fh(self, Zs: DataSet):
        m: int = Zs.ncols
        p: int = Zs.nrows
        # first column f
        # second column h
        fh: DataSet = DataSet(name="fh", nrows=p, ncols = 2)
        fh.points = [[0,0]]*p

        if m == 1:
            fh.__set_col__(Zs, 0)
            return fh
        elif m==self._m:
            for j in range(self._m):
                if self._trainingSet.get_bbo(j) == bbo_t.BBO_OBJ:
                    # Copy the objective in the first column of
                    fh.__set_col__(Zs.__get_col__(j), 0)
                elif self._trainingSet.get_bbo(j) == bbo_t.BBO_CON:
                    c0 = self._trainingSet.Z_scale(0., j)
                    for i in range(p):
                        d = Zs.points_get(i,j) - c0
                        if d>0 : fh.points[i][1] += d*d
                else:
                    ExceptionError(name="compute_fh", message="Undefined type")
        else:
            ExceptionError(name="compute_fh", message="Dimension error")
        return fh

    def compute_aggregate_order_error(self, Zpred: DataSet):
        """   // Zpred must be a matrix with _p rows, and _m or 1 columns.
              // If there is only 1 column, then this column is considered as an aggregate of the
              // objective and constraints. For example, it can be the EFI. """
        fhr: DataSet = self.compute_fh(self.get_matrix_Zs())
        fhs: DataSet = self.compute_fh(Zpred)

        e = 0
        # // i1 and i2 are the indexes of the two points that are compared.
        #   // fr1 and hr1 (resp. fr2 and hr2) are the real values of f and r for these points.
        #   // fs1 and hs1 (resp. fs2 and hs2) are the surrogate (or CV) values.
        for i1 in range(self._p):
            fr1 = fhr.points_get(i1, 0)
            hr1 = fhr.points_get(i1, 1)
            fs1 = fhs.points_get(i1, 0)
            hs1 = fhs.points_get(i1, 1)
            for i2 in range(self._p):
                fr2 = fhr.points_get(i2, 0)
                hr2 = fhr.points_get(i2, 1)
                fs2 = fhs.points_get(i2, 0)
                hs2 = fhs.points_get(i2, 1)
                # Compute the order for real (r) data and for surrogate (s) model
                inf_r: bool = hr1 < hr2 or (hr1==hr2 and fr1<fr2)
                inf_s: bool = hs1 < hs2 or (hs1 == hs2 and fs1 < fs2)
                # If they don't agree, increment e. (Note that ^ is the xor operator)
                if inf_r or inf_s: e+=1
        return float(e)/float(self._p*self._p)

    def compute_efi(self, Zs: DataSet, Ss: DataSet):
        # /*----------------------------------------------------------*/
        # /*     compute EFI from the predictive mean and std         */
        # /*----------------------------------------------------------*/
        if Zs.ncols != self._m or Ss.ncols != self._m \
                or Zs.nrows != self._p or Ss.nrows != self._p:
            raise ExceptionError(name="compute_efi", message="Dimension Error")
        fmin: float = self._trainingSet.get_f_min()
        EFI: DataSet = DataSet(name="EFI", nrows=self._p, ncols=1, points=[1.0]*self._p)
        for j in range(self._m):
            if self._trainingSet.get_bbo(j) == bbo_t.BBO_OBJ:
                for i in range(self._p):
                    # Compute Expected Improvement for point i
                    ei = self.normei(Zs.points_get(i,j), Ss.points_get(i,j), fmin)
                    # Unscale Expected Improvement
                    ei = self._trainingSet.ZE_unscale(ei, j)
                    # Multiply EFI by ei
                    EFI.points[i][0] *= ei
            elif (self._trainingSet.get_bbo(j) == bbo_t.BBO_CON):
                c0 = self._trainingSet.Z_scale(0.0, j)
                for i in range(self._p):
                    EFI.points[i][0] *= self.normcdf(x=c0, mu=Zs.points_get(i, j), sigma=Ss.points_get(i, j))
        return EFI

    def compute_metric_linv(self):
        if not self.isdef(self.METRIC_LINV):
            # init
            v: DataSet = DataSet("v", 1, self._m)
            # Compute the prediction on the training points
            Zhs: DataSet = self.get_matrix_Zhs()
            Shs: DataSet = self.get_matrix_Shs()
            # True values
            Zs: DataSet = self.get_matrix_Zs()
            # TODO : improve the behavior of linv for very small s.
            for j in range(self._m):
                if self._trainingSet.get_bbo(j) != bbo_t.BBO_DUM:
                    linv = 0
                    for i in range(self._p):
                        dz = Zhs.points_get(i,j)-Zs.points_get(i,j)
                        s = Shs.points_get(i, j)
                        s = max(s, constants.EPSILON)
                        dz = max(dz, constants.EPSILON)
                        linv += (-np.log(s) - (dz/s)**2)/2
                        # // Add this point, we have log(prod g)/p
                        linv = np.exp(-linv)
                else:
                    linv = -1
                v.points_set(0, j, linv)
            self._metrics[self.METRIC_LINV] = v

    def compute_metric(self, mt: int):
        """"""
        if self.is_metric_defined(mt=mt): return True
        m: float
        j: int
        # Choose if we use the Zhs or the Zvs matrix
        # Zvs is used if we want to use cross-validation
        Zs: DataSet = self.get_matrix_Zs()
        Zs_compare: DataSet
        Ss_compare: DataSet
        if self.metric_uses_cv(mt=mt):
            Zs_compare = self.get_matrix_Zs()
            Ss_compare = self.get_matrix_Svs()
        else:
            Zs_compare = self.get_matrix_Zhs()
            Ss_compare = self.get_matrix_Shs()


        # Size of the metric vector
        vector_size: int = self.one_metric_value_per_bbo(mt)
        # Init the metric vector
        v: DataSet = DataSet(name="v", nrows=1, ncols=vector_size)
        # Norm associated to a given metric
        if mt in [self.METRIC_EMAX, self.METRIC_EMAXCV, self.METRIC_RMSE, self.METRIC_RMSECV,
                  self.METRIC_ARMSE, self.METRIC_ARMSECV]:
            associated_norm = self.metric_type_to_norm_type(mt)
            # // Compute the norm of the difference
            v = (Zs-Zs_compare).__col_norm__(associated_norm)
            if mt == self.METRIC_ARMSE or mt == self.METRIC_ARMSECV:
                # For "Aggregate" metrics, compute the sum for all BBO
                v = v.__sum__(1)
            else:
                self._trainingSet.ZE_unscale_DataSet(v)
        elif mt == self.METRIC_OE or mt == self.METRIC_OECV:
            # Order error. See paper:
            # Order-based error for managing ensembles of surrogates in mesh adaptive direct search
            v = self.compute_order_error(Zs_compare)
        elif mt == self.METRIC_AOE or mt == self.METRIC_AOECV:
            # // Aggregate order error. See paper:
            # //Locally weighted regression models for surrogate-assisted design optimization
            v = DataSet(name=v.name, nrows=v.nrows, ncols=v.ncols,
                        points=self.compute_aggregate_order_error(Zs_compare))
        elif mt == self.METRIC_EFIOE or mt == self.METRIC_EFIOECV:
            # Aggregate Order error on Expected Feasible Improvement
            v = DataSet(name=v.name, nrows=v.nrows, ncols=v.ncols,
                        points=self.compute_aggregate_order_error(-self.compute_efi(Zs_compare, Ss_compare)))
        elif mt == self.METRIC_LINV:
            # Inverse of the likelihood
            self.compute_metric_linv()
        else:
            raise ExceptionError(name="metric computation", message="Metric not recognized.")

        for j in range(vector_size):
            m = v.points[0][j]
            if np.isnan(m): m = np.inf
            if m < constants.EPSILON: m = np.inf
            if m <= 0.: m =0.

        self._metrics[mt] = v
        return True

    def _is_mtdef(self, mt: int):
        if np.isnan(mt): return False
        if np.isinf(mt): return False
        if np.fabs(mt) >= np.inf: return False
        if np.fabs(mt) >= 1e16: return False
        return True

    def isdef(self, mt: int, j: Optional[int] = None) -> bool:
        """ Not NAN nor INF """
        if j is None:
            return self._is_mtdef(mt)
        else:
            if not self._is_mtdef(mt): return False
            if j >= self._metrics[mt].ncols or j>=self._m or j<0: return False
            return True

    def get_metric(self, mt: int, j:Optional[int] = None):
        if j is None:
            # If the model is not ready, return +INF
            if not self._ready: return np.inf
            # If the metric is defined, return it
            if mt in range(12): return mt
            # Compute the metric,
            if not self.compute_metric(mt): return np.inf
            # Return value
            if self.isdef(mt=mt): return self._metrics[mt]
            return np.inf
        else:
            # If the model is not ready, return +INF
            if not self._ready: return np.inf
            # If the metric is defined, return it
            if mt in range(12): return mt
            # Compute the metric,
            if not self.compute_metric(mt): return np.inf
            # Return value
            if self.isdef(mt=mt, j=j): return self._metrics[mt].points[j]
            # Is still not defined, return INF.
            return np.inf

    def eval_objective(self):
        self.reset_metrics()
        # Build model
        ok: bool = self.build_private()
        if not ok: return np.inf
        # Get the metric type specified in the parameter.
        mt = self._param.metric_type
        metric = 0.
        # // metric_multiple_obj indicate if the given metric "mt"
        # // is scalar (one metric for all the blackbox outputs, like AOECV)
        # // or is an array (one metric for each blackbox outputs, like RMSE)
        if self.one_metric_value_per_bbo(mt):
            # TODO: check it (hint: _metrics is a dict)
            for i in range(self._m): metric += self.get_metric(mt=mt, j=i)
        else:
            metric = self.get_metric(mt=mt, j=0)

        if np.isnan(metric): return np.inf
        if np.isinf(metric): return np.inf
        return metric

    def build(self):
        """ Build surrogate"""
        # Check the parameters of the model:
        self._param.check()
        # Before building the surrogate, the trainingset must be ready
        self._trainingSet.build()
        # Number of points in the training set.
        self._p_ts = self._trainingSet.get_nb_points()
        if self._ready and self._p_ts == self._p_ts_old:
            print("Surrogate build - SKIP Build")
            return True

        # Otherwise, the model is not ready and we need to call build_private
        self._ready = False

        # Get the number of points used in the surrogate
        if len(self._selected_points) == 1 and self._selected_points[0] == -1:
            self._p = self._p_ts
        else:
            self._p = len(self._selected_points)

        # Need at least 2 point to build a surrogate.
        if self._p < 2:
            return False

        # Delete the intermediate data and metrics (they will have to be recomputed...)
        self.reset_metrics()
        print("Surrogate build - BUILD_PRIVATE")
        ok: bool = False
        #   // First, the model has to be initialized.
        #   // This step does not involve parameter optimization.
        #   // For some types of model, the initialization step does nothing.
        #   // The initialization step is necessary, for example, for RBF models, where the "preset"
        #   // has to be considered first, and the kernel have to be selected before the parameter
        #   // optimization.
        ok = self.init_private()

        if not ok:
            return False

        # Optimize parameters
        if self._param.nb_parameter_optimization > 0:
            ok = self.optimize_parameters()
            if not ok:
                self._ready = False
                return False

        # Build private
        ok = self.build_private()

        if not ok:
            self._ready = False
            return False

        #  Memorize previous number of points
        self._p_ts_old = self._p_ts
        self._p_old = self._p

        self._ready = True
        return True

    def get_matrix_Xs(self):
        """ /*--------------------------------------*/
            /*       get_Xs                         */
            /* Returns the scaled input for all     */
            /* the selected data points selected    */
            /*--------------------------------------*/ """

        self._trainingSet.build()
        # TODO: Check if this covers the multiple indices as well
        temp: DataSet = DataSet()
        temp.points = self._trainingSet.get_matrix_Xs().get_row(self._selected_points)
        return temp

    def get_matrix_Zs(self):
        """ /*--------------------------------------*/
            /*       get_Zs                         */
            /* Returns the scaled input for all     */
            /* the selected data points selected    */
            /*--------------------------------------*/ """

        self._trainingSet.build()
        # TODO: Check if this covers the multiple indices as well
        temp : DataSet = DataSet()
        temp.points = self._trainingSet.get_matrix_Zs().get_row(self._selected_points)
        return temp

    def get_matrix_Ds(self):
        """ /*--------------------------------------*/
            /*       get_Ds                         */
            /* Returns the scaled input for all     */
            /* the selected data points selected    */
            /*--------------------------------------*/ """

        self._trainingSet.build()
        # TODO: Check if this covers the multiple indices as well
        temp: DataSet = DataSet()
        temp.points = self._trainingSet.get_matrix_Ds().get_row(self._selected_points)
        return temp

    def get_matrix_Svs(self):
        # If no specific method is defined, consider Svs = Shs.
        if self._Svs is None:
            self._Svs: DataSet = DataSet(name="Svs", nrows=self._p, ncols=self._m)
            Ds: DataSet = self._trainingSet.get_matrix_Ds()
            for i in range(self._p):
                dmin: float = np.inf
                for j in range(self._p):
                    if i != j:
                        dmin = min(dmin, Ds.points_get(i,j))
                # TODO: I am not sure about it, recheck
                self._Svs.points[i] = [dmin]* len(self._Svs.points[i])
        return self._Svs

    def get_matrix_Shs(self):
        # If no specific method is defined, consider Svs = Shs.
        if self._Shs is None:
            self._Shs: DataSet = DataSet(name="Shs", nrows=self._p, ncols=self._m)
            #TODO: implement
            self.predict_private ()
            self._Shs.replace_nan(np.inf)
            self._Shs.name = "Shs"

        return self._Shs

    def get_matrix_Zhs(self):
        if self._Zhs is None:
            self._Zhs: DataSet = DataSet(name="Zhs", nrows=self._p, ncols=self._m)
            #TODO: implement
            self.predict_private()
            self._Zhs.replace_nan(np.inf)
            self._Zhs.name = "Zhs"

        return self._Zhs

@dataclass
class PRS(Surrogate):
    training_set: TrainingDataSet = TrainingDataSet()
    param: Surrogate_Parameters = field(default_factory=Surrogate_Parameters)
    _q: int = 0
    _M: DataSet = DataSet(name="M", nrows=0, ncols=0)
    _H: DataSet = DataSet(name="H", nrows=0, ncols=0)
    _Ai: DataSet = DataSet(name="Ai", nrows=0, ncols=0)
    _alpha: DataSet = DataSet(name="alpha", nrows=0, ncols=0)

    def __post_init__(self):
        super(Surrogate, self).__init__(_trainingSet=self.training_set, _param=self.param)

    def build_private(self):
        pvar: int = self._trainingSet.pvar
        nvar: int = self._trainingSet.nvar
        # Get the number of basis functions.
        self._q = self.get_nb_PRS_monomes(nvar, self._param.degree)
        # If _q is too big or there is not enough points, then quit
        if self._q>200:
            return False
        if self._q>pvar-1 and self._param.ridge==0:
            return False
        # Compute the exponents of the basis functions
        self._M = self.get_PRS_monomes()
        # DESIGN MATRIX H
        self._H = self.compute_design_matrix(self._M, self.get_matrix_Xs())
        # Compute alpha
        if not self.compute_alpha() : return False
        self._ready = True
        return True

    def compute_design_matrix(self, Monomes:DataSet, Xs: DataSet):
        """ Compute PRS matrix """
        # Nb of points in the matrix X given in argument
        n: int = Xs.ncols
        # Nb of points in the matrix X given in argument
        p: int = Xs.nrows
        v: float
        nbMonomes = Monomes.nrows

        # Init the design matrix
        H: DataSet = DataSet(name="H", nrows=p, ncols=nbMonomes)
        # Current basis function (vector column to construct 1 basis function)
        h: DataSet = DataSet(name="h", nrows=p, ncols=1)

        #   // j is the corresponding index among all input (j in [0;n-1])
        #   // jj is the index of the input variabe amongst the varying input (jj in [0;nvar-1])
        #   // k is the index of the monome (ie: the basis function) (k in [0;q-1])
        #   // i is the index of a point (i in [0;p-1])
        #   // Loop on the monomes
        for k in range(nbMonomes):
            h.points = [1.0]*h.nrows








@dataclass
class surrogate_CN(Surrogate):
    _trainingset: TrainingDataSet = None

    def build_private(self) -> bool:
        self._ready = True
        return True

    def predict_private(self, XXs: DataSet, ZZs: DataSet):
        """ predict_private (ZZs only) """
        i: int
        imin: int
        pxx: int = XXs.nrows
        # D : distance between points of XXs and other points of the trainingset
        # TODO:
        D: DataSet = self._trainingset.get_distances(XXs, self.get_matrix_Xs(),
                                                     self._param.distance_type)
        Zs: DataSet = self.get_matrix_Zs()
        # // Loop on the points of XXs
        for i in range(pxx):
            # // imin is the index of the closest neighbor of xx in Xs
            imin = D.get_min_index_row(i)
            ZZs.points[i] = Zs.get_row(imin)
        return ZZs

    def compute_cv_values(self):
        if self._Zvs is not None and self._Svs is not None:
            return True

        # Init matrices
        if self._Zvs is None:
            self._Zvs: DataSet = DataSet(name="Zvs", nrows=self._p, ncols=self._m)

        if self._Svs is None:
            self._Svs: DataSet = DataSet(name="Svs", nrows=self._p, ncols=self._m)

        imin = 0
        D: DataSet = self._trainingset.get_distances(self.get_matrix_Xs(), self.get_matrix_Xs(), self._param.distance_type)
        Zs: DataSet = self.get_matrix_Zs()

        for i in range(self._p):
            # Find the closest point to iv (not itself)
            dmin: float = np.inf
            # Loop on the points of the trainingset
            for i2 in range(self._p):
                d = D.points_get(i, i2)
                if i != i2 and d < dmin:
                    dmin = d
                    imin = i2
            self._Zvs.points[i] = Zs.get_row(imin)
            #TODO: This is wrong I need to recheck it and find a solution
            self._Svs.points[i] = dmin

        return True

    def get_matrix_Zhs(self):
        # TODO: implement file and function check ready in Surrogate class
        # check_ready(__FILE__,__FUNCTION__,__LINE__);
        if self._Zhs is None:
            self._Zhs: DataSet = self.get_matrix_Zs()

        return self._Shs

    def get_matrix_Shs(self):
        # TODO: implement file and function check ready in Surrogate class
        # check_ready(__FILE__,__FUNCTION__,__LINE__);
        if self._Shs is None:
            self._Shs: DataSet = DataSet(name="Shs", nrowa=self._p, ncols=self._m)

        return self._Shs

    def get_matrix_Zvs(self):
        # TODO: implement file and function check ready in Surrogate class
        # check_ready(__FILE__,__FUNCTION__,__LINE__);
        self.compute_cv_values()

        return self._Zvs


    def get_matrix_Svs(self):
        # TODO: implement file and function check ready in Surrogate class
        # check_ready(__FILE__,__FUNCTION__,__LINE__);
        self.compute_cv_values()

        return self._Svs

@dataclass
class Surrogate_Factory:
    TS: TrainingDataSet
    s: str
