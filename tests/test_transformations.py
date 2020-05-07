import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pandas.testing import assert_frame_equal
from transformations import snv, area_norm, plus_sg
import unittest
import pandas as pd
import numpy as np

class TestTransformation(unittest.TestCase):
    def test_snv(self):
        df = pd.DataFrame({'id': [1, 2], 'chemical': [2.91, 2.91], '1': [3.0, 5.0], '2': [4.0, 8.0], 3: [2.0, 2.0], 4: [3.0, 7.0]})
        df_result = pd.DataFrame({'id': [1, 2], 'chemical': [2.91, 2.91], '1': [0, -0.188982236504614], '2': [1.22474487128318, 0.944911182523068], 3: [-1.22474487128318, -1.3228756555323], 4: [0, 0.566946709513841]})
        result = snv(df)

        test1 = 'Test'
        test2 = 2
        test3 = True

        self.assertRaises(ValueError, snv, test1)
        self.assertRaises(ValueError, snv, test2)
        self.assertRaises(ValueError, snv, test3)
        self.assertRaises(ValueError, snv, df, test1)


        assert_frame_equal(result, df_result)
    
    
    def test_area_norm(self):
        df = pd.DataFrame({'id': [1, 2], 'chemical': [2.91, 2.91], '1': [3.0, 5.0], '2': [4.0, 8.0], 3: [2.0, 2.0], 4: [3.0, 7.0]})
        df_resutlt = pd.DataFrame({'id': [1, 2], 'chemical': [2.91, 2.91], '1': [0.25, 0.227272727272727], '2': [0.333333333333333, 0.363636363636364], 3: [0.166666666666667, 0.0909090909090909], 4: [0.25, 0.318181818181818]})
        result = area_norm(df)

        test1 = 'Test'
        test2 = 2
        test3 = True

        self.assertRaises(ValueError, snv, test1)
        self.assertRaises(ValueError, snv, test2)
        self.assertRaises(ValueError, snv, test3)
        self.assertRaises(ValueError, snv, df, test1)


        assert_frame_equal(result, df_resutlt)
    
    def test_plus_sg(self):
        
        df = pd.DataFrame({'id': [1, 2], 'chemical': [2.91, 2.91], '1': [3.0, 5.0], '2': [4.0, 8.0], 3: [2.0, 2.0], 4: [3.0, 7.0]})
        df_averaged = pd.DataFrame({'id': [1, 2], 'chemical': [2.91, 2.91], '1': [0.25, 0.227272727272727], '2': [0.333333333333333, 0.363636363636364], 3: [0.166666666666667, 0.0909090909090909], 4: [0.25, 0.318181818181818]})

        self.assertRaises(ValueError, plus_sg, 'test', 'test', 'test', 'test', 'test', 'test')