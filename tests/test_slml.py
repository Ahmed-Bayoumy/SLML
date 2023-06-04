import numpy as np
import pytest
from SLML import *
from typing import List, Dict
import os
import pickle
from scipy.optimize import minimize, rosen, rosen_der
import copy

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

def bench2( x, isSurf):
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

def bench3( x):
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

def branin( x, a=1, b=5.1 / (4 * np.pi ** 2), c=5. / np.pi,
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

def RB( x, kx):
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

def RB2d( x, isSurf = None):
  
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

def RB1( x, kx):
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
    Z[i, :] = np.transpose(RB(np.transpose(S), None))

  return Z

def RB_RBF( display=True):
  v = np.array([[-2.0, 2.0], [-2.0, 2.0]])
  n = 500

  sampling = LHS(ns=n, vlim=v)

  xt = sampling.generate_samples()
  yt = RB(xt, None)
  opts: Dict = {}
  opts = {"display": True}
  'multiquadratic', 'Gaussian', 'inverse_multiquadratic', 'absolute', 'linear', 'cubic', 'thin_plate'
  HP = {"rbf_func": {"type": "C_S1", "lb": 0, "ub": 3, "sets": {"S1": ["Gaussian", "multiquadratic", "inverse_multiquadratic", "cubic"]}, "value": "Gaussian"}, "gamma": {"lb": 0.01, "ub": 2., "value": 1.}}
  sm = RBF(x=xt, y=yt, options=opts, HP=HP)
  sm.train()

  sm.tuner(display=True)

  print("optimal model:")
  sm._display_acc_metrics()
  sm.model_db = os.path.join(os.getcwd(), f"tests/models_db/{sm.name}")
  sm.store_model()

  sm1 = RBF(x=xt, y=yt)
  sm1.model_db = os.path.join(os.getcwd(), f"tests/models_db/{sm.name}")

  sm1.load_model()


def RB_kriging( display=True):
  v = np.array([[-2.0, 2.0], [-2.0, 2.0]])
  n = 500

  sampling = LHS(ns=n, vlim=v)

  xt = sampling.generate_samples()
  yt = RB(xt, None)
  opts: Dict = {}
  opts = {"display": True}
  opts: Dict = {}
  opts = {"display": True}
  sm = Kriging(x=xt, y=yt, options=opts)
  sm.train()

  sm.tuner(display=True)

  print("optimal model:")
  sm._display_acc_metrics()
  sm.model_db = os.path.join(os.getcwd(), f"tests/models_db/{sm.name}")
  sm.store_model()

  sm1 = Kriging(x=xt, y=yt)
  sm1.model_db = os.path.join(os.getcwd(), f"tests/models_db/{sm.name}")
  sm1.load_model()


def RB_LS( display=True):
  v = np.array([[-2.0, 2.0], [-2.0, 2.0]])
  n = 500

  sampling = LHS(ns=n, vlim=v)

  xt = sampling.generate_samples()
  yt = RB(xt, None)
  opts: Dict = {}
  opts = {"display": True}
  
  HP = {"type": {"type": "C_S1", "lb": 0, "ub": 3, "sets": {"S1": ["linear", "lasso", "polynomial", "ridge"]}, "value": "polynomial"}, "degree": {"type": "I", "lb": 2, "ub": 10, "value": 4}, "alpha": {"lb": 0.01, "ub": 1., "value": 0.5}}
  sm = LS(x=xt, y=yt, HP=HP, options = opts)
  sm.train()

  sm.tuner(display=True)

  print("optimal model:")
  sm._display_acc_metrics()
  sm.model_db = os.path.join(os.getcwd(), f"tests/models_db/{sm.name}")
  sm.store_model()

  sm1 = LS(x=xt, y=yt)
  sm1.model_db = os.path.join(os.getcwd(), f"tests/models_db/{sm.name}")

  sm1.load_model()

def Branin_LS_Poly( display=True):
  # Generate samples
  v = np.array([[-5.0, 10.0], [0.0, 15.0]])
  n = 300
  sampling = LHS(ns=n, vlim=v)
  xt = sampling.generate_samples()
  yt = branin(xt)
  opts: Dict = {}
  # Prepare and fit the model
  opts = {"type": "polynomial", "degree": 4, "alpha": 0.5, "display": True}
  sm = LS(x=xt, y=yt, HP=opts)
  sm.train()
  # Bootstrap
  lower, yp, upper = sm.bootstrap()
  


def test_quick():
  p = bmSM()
  p.fhist = []
  s = np.array([[1.2, 1.2]])
  p.xhist = s
  RB_RBF(display=False)
  RB_kriging(display=False)
  RB_LS(display=False)



 