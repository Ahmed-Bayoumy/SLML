import numpy as np
import pytest
from SLML import *
from typing import List, Dict
import os
import pickle
from scipy.optimize import minimize, rosen, rosen_der
import copy


def test_quick():
  p = bmSM()
  p.fhist = []
  s = np.array([[1.2, 1.2]])
  p.xhist = s
  p.RB_RBF(display=False)
  p.RB_kriging(display=False)
  p.RB_LS(display=False)



 