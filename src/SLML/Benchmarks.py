
from SLML.Models import Kriging, RBF, LS, MOE
from ._visualize import visualize, plt
import numpy as np
from typing import Dict, List
import os
from pyDOE2 import lhs

def scale_to_limits(varLimits, S: np.ndarray) -> np.ndarray:
    """
      Scale the samples from the unit hypercube to the specified limit.
    """
    n = varLimits.shape[0]
    for i in range(n):
      S[:, i] = varLimits[i, 0] + S[:, i] * \
            (varLimits[i, 1] - varLimits[i, 0])
        
    return S

class bmSM:
  xhist: np.ndarray = np.empty((1, 2))
  fhist: List = list

  def bench1(self, x):
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

  def bench4(self, x):
    """A benchmark function for test purposes.

        f(x) = float(x) ** 2

    where x is a string. It has a single minima with f(x*) = 0 at x* = "0".
    This benchmark is used for checking support of categorical variables.
    """
    return float(x[0]) ** 2

  def bench5(self, x, isSurf=False):
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

  def hart6(self, x,
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

    sampling = scale_to_limits(varLimits=v, S=lhs(n=2, samples=n))

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

    sampling = scale_to_limits(varLimits=v, S=lhs(n=2, samples=n))

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

    sampling = scale_to_limits(varLimits=v, S=lhs(n=2, samples=n))

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
    sampling = scale_to_limits(varLimits=v, S=lhs(n=2, samples=n))
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
    sampling = scale_to_limits(varLimits=v, S=lhs(n=2, samples=n))
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
    
