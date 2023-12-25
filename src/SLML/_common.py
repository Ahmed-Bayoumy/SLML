from enum import Enum, auto
import sys
from dataclasses import dataclass

class FileMissingError(IOError):
  """ Custom error that is raised when the data file is missing. """

  def __init__(self, name: str, message: str) -> None:
    self.name = name
    self.message = message
    super().__init__(message)


class ExceptionError(Exception):
  """Exception error."""

  def __init__(self, name: str, message: str) -> None:
    self.name = name
    self.message = message
    exception_type, exception_object, exception_traceback = sys.exc_info()
    self.filename = exception_traceback.tb_frame.f_code.co_filename
    self.line_number = exception_traceback.tb_lineno

    super().__init__(message)


class SAMPLING_METHOD(Enum):
  FULLFACTORIAL: int = auto()
  LH: int = auto()
  RS: int = auto()
  HALTON: int = auto()

class MODEL_TYPE(Enum):
  KRIGING: int = auto()
  RBF: int = auto()
  LS: int = auto()
  MOE: int = auto()

@dataclass
class norm_t:
  NORM_0: int = 0
  NORM_1: int = 1
  NORM_2: int = 2
  NORM_INF: int = 3

