from __future__ import absolute_import
import os

from . import utils
from . import models

__version__ = '0.0.1'

_data_dir = os.path.dirname(os.path.realpath(__file__)) + '/data'
if not os.path.exists(_data_dir):
    os.makedirs(_data_dir)

_models_data_dir = os.path.dirname(os.path.realpath(__file__)) + '/models/data'
if not os.path.exists(_models_data_dir):
    os.makedirs(_models_data_dir)