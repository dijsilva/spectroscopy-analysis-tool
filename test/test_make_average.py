import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

from pandas.testing import assert_frame_equal
from context import average


class TestMakeAverage(unittest.TestCase):
  def test_structure(self):
    df = pd.DataFrame({'id': [1, 2], 'chemical': [2.91, 2.91], 1: [3.0, 5.0], 2: [4.0, 8.0], 3: [2.0, 2.0], 4: [3.0, 7.0]})
    df_averaged = pd.DataFrame({'id': [1], 'chemical': [2.91], 1: [4.0], 2: [6.0], 3: [2.0], 4: [5.0]})
    result = average.make_average(df, 2)

    assert_frame_equal(result, df_averaged)
  
  def test_is_dataframe(self):
    df = pd.DataFrame({'id': [1, 2], 'chemical': [2.91, 2.91], 1: [3.0, 5.0], 2: [4.0, 8.0], 3: [2.0, 2.0], 4: [3.0, 7.0]})
    df1 = 'test'
    df2 = 0
    df3 = True

    self.assertRaises(ValueError, average.make_average, df1, 2)
    self.assertRaises(ValueError, average.make_average, df2, 2)
    self.assertRaises(ValueError, average.make_average, df3, 2)
    self.assertRaises(ValueError, average.make_average, df, 'as')


