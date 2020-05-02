import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

from utils import average


class TestMakeAverage(unittest.TestCase):
  def test_structure(self):
    # Test if structure of output is correctly and if the function return the average of values
    
    df = pd.DataFrame({'id': [1, 2], 'chemical': [2.91, 2.91], 1: [3.0, 5.0], 2: [4.0, 8.0], 3: [2.0, 2.0], 4: [3.0, 7.0]})
    df_averaged = pd.DataFrame({'id': [1], 'chemical': [2.91], 1: [4.0], 2: [6.0], 3: [2.0], 4: [5.0]})
    result = average.make_average(df, 2)

    assert_frame_equal(result, df_averaged)
  
  def test_is_dataframe(self):
    # Test if the function handle with invalid params
    df = pd.DataFrame({'id': [1, 2], 'chemical': [2.91, 2.91], 1: [3.0, 5.0], 2: [4.0, 8.0], 3: [2.0, 2.0], 4: [3.0, 7.0]})
    test1 = 'test'
    test2 = 0
    test3 = True

    self.assertRaises(ValueError, average.make_average, test1, 2)
    self.assertRaises(ValueError, average.make_average, test2, 2)
    self.assertRaises(ValueError, average.make_average, test3, 2)
    self.assertRaises(ValueError, average.make_average, df, test1)
    self.assertRaises(ValueError, average.make_average, df, test2)
    self.assertRaises(ValueError, average.make_average, df, test3)


