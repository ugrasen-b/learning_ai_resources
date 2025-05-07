import pandas as pd
import numpy as np

class ClientRateFunction:

    def __init__(self, simcluster, parameter_file):
        self.simcluster = simcluster
        self.parameter_file = parameter_file
