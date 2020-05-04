import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pandas.testing import assert_frame_equal
from algorithms import PLSR
import unittest
import pandas as pd
import numpy as np

class TestPLSR(unittest.TestCase):
  def test_params(self):

    dataset = pd.DataFrame({'id': [1, 2], 'chemical': [2.91, 2.91], 1: [3.0, 5.0], 2: [4.0, 8.0], 3: [2.0, 2.0], 4: [3.0, 7.0]})
    dataset_val = pd.DataFrame({'id': [1, 2], 'chemical': [2.91, 2.91], 1: [3.0, 5.0], 2: [4.0, 8.0], 3: [2.0, 2.0], 4: [3.0, 7.0]})
    
    self.assertRaises(ValueError, PLSR, dataset, '5', 'test', 'test', dataset_val, scale='teste', plsr_random_state='test')