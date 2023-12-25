

import numpy as np
from typing import Any
from dataclasses import dataclass

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
  
