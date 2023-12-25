
from .Dataset import DataSet
import numpy as np
from typing import Callable, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from ._metrics import accuracy_metrics
import os
import copy
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from OMADS import POLL
from sklearn.model_selection import KFold
import shelve


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
    out = POLL.main(data)
    results = out[0]["xmin"]
    c = 0
    for d in self.HP:
      if isinstance(self.HP[d], dict) and "id" in self.HP[d] and self.HP[d]["id"] == c:
        if self.HP[d]["type"][0] == "D" or self.HP[d]["type"][0] == "C":
          self.HP[d]["value"] = results[c]
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
