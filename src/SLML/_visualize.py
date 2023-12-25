import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import cm


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
