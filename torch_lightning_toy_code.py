# -*- coding: utf-8 -*-
"""
Created on Wed May  7 11:40:42 2025

@author: Bob
"""

import time

import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import numpy as np
import torch as th
from torch import Tensor
import torch.nn as nn
import torch.utils.data as data
from matplotlib.colors import to_rgba
from tqdm.notebook import tqdm

matplotlib_inline.backend_inline.set_matplotlib_formats("svg", "pdf")

print(f"Using torch {th.__version__}")
th.manual_seed(42)

x = Tensor(2, 3, 4)
print(x)

