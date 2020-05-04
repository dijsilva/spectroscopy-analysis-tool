import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pandas.testing import assert_frame_equal
from transformations import snv
import unittest
import pandas as pd
import numpy as np

class TestTransformation(unittest.TestCase):
  def test_snv(self):
    df = pd.DataFrame({'id': [1, 2], 'chemical': [2.91, 2.91], 1: [3.0, 5.0], 2: [4.0, 8.0], 3: [2.0, 2.0], 4: [3.0, 7.0]})
    df_averaged = pd.DataFrame({'id': [1, 2], 'chemical': [2.91, 2.91], 1: [0, -0.188982236504614], 2: [1.22474487128318, 0.944911182523068], 3: [-1.22474487128318, -1.3228756555323], 4: [0, 0.566946709513841]})
    result = snv(df)

    assert_frame_equal(result, df_averaged)