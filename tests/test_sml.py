import numpy as np
import pytest
from SML import *
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
  p.RBTrue()
  p.xhist = s
  p.RB_RBF()
  p.xhist = s
  p.RB_RBF1()
  p.xhist = s
  p.RB_kriging()
  p.xhist = s
  p.RB_kriging1()



 