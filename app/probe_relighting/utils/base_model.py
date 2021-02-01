from abc import ABCMeta, abstractmethod
from torch import nn


class BaseModel(nn.Module, metaclass=ABCMeta):

    log_it = {'data': [],
              'output': [],
              'losses': []}

    def __init__(self, opt):
        super().__init__()
        self._init(opt)

    def forward(self, data):
        return self._forward(data)

    @abstractmethod
    def _init(self, opt):
        raise NotImplementedError

    @abstractmethod
    def _forward(self, data):
        raise NotImplementedError
