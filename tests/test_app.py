# -*- coding: utf-8 -*-
"""

I need to lear how to proper test when using Streamlit! Until then I made copy
of the functions (file functions_for_testing) in order to be able to test them.
This is way away from good testing, but hey, its better than no tests at all!
Therefore, any changed in App functions should be copied to the file 
'functions_for_testing' and then tested.

"""

import sys
sys.path.insert(0, 'D:/Rob_work/02_Project/Streamlit/App_CDFSciPy/tests')

import unittest

import numpy as np
np.random.seed(1)
from scipy import stats
import pandas as pd

from functions_for_testing import (get_multi_parameters,
                                   obtain_functional_data,
                                   which_s,
                                   how_many_params,
                                   get_n,
                                   fit_data,
                                   produce_output_for_code_download_parameters,
                                   )

class TestApp(unittest.TestCase):
    
    # For reference
    dist_a = 'alpha'
    dist_b = 'beta'
    dist_c = 'anglit'
    
    param_a = 'a'
    param_b = 'a, b'
    param_c = None
    
    
    size = 400
    sliders_params = [3.57, 0.0, 1.0]    

    # slider
    select_distribution = 'norm'

    def test_get_multi_parameters(self):
        """ Testing sliders/parameters output """      
        
        x, r, rv = get_multi_parameters(self.sliders_params)
        
        self.assertTrue(get_multi_parameters([3.57, 0.0, 1.0]), 
                        get_multi_parameters(self.sliders_params))

        self.assertEqual(get_multi_parameters([3.57, 0.0, 1.0])[0][0], 
                        x[0])



        # Checking extraction of parameters values/names
        for j, param in enumerate(self.sliders_params):
            self.assertEqual(self.sliders_params[j], param)

            dist = getattr(stats, self.dist_a)            
            self.assertEqual(dist.name, self.dist_a)
            self.assertEqual(dist.shapes, self.param_a)

            dist = getattr(stats, self.dist_b) 
            self.assertEqual(dist.name, self.dist_b)
            self.assertEqual(dist.shapes, self.param_b)
            
            dist = getattr(stats, self.dist_c) 
            self.assertEqual(dist.name, self.dist_c)
            self.assertEqual(dist.shapes, self.param_c)


    def test_obtain_functional_data(self):
        """ Testing connection functions to default parameters """
        
        names = ['alpha', 'beta', 'norm', 'gausshyper']
        # Expected default parameters on select
        params = [[3.57, 0.0, 1.0], [2.31, 0.63, 0.0, 1.0], [0.0, 1.0],
                  [13.76, 3.12, 2.51, 5.18, 0.0, 1.0] ]
        
        for i, func in enumerate(names):
            self.select_distribution = func
            self.assertEqual(params[i],
                         obtain_functional_data(self.select_distribution))
    
    def test_which_s(self):
        """ Testing sigma shading area """

        x = np.array([0,1,2,3,4,5,6,7,8,9,10])
        r = np.array([1, 2, 5]) # mean 2.66; std 1.699
        
        s = 1
        x_r, s_r = which_s(x, r, s)

        self.assertEqual(x_r.tolist(), np.array([1, 2, 3, 4]).tolist())
        self.assertAlmostEqual(round(s_r,2), 1.7)

        s = 3
        x_r, s_r = which_s(x, r, s)
        self.assertEqual(x_r.tolist(), 
                         np.array([0, 1, 2, 3, 4, 5, 6, 7]).tolist())
        self.assertAlmostEqual(round(s_r, 2), 5.1)
        
        
    def test_how_many_params(self):

        # Input
        sliders_params = [3.57, 0.0, 1.0]
        self.assertEqual(how_many_params(self.sliders_params),
                        ('scale=1.0', 'loc=0.0', ['a'], ['a=3.57']))

        # Tested individually since I have to change distributions
        #sliders_params = [13.76, 3.12, 2.51, 5.18, 0.0, 1.0]
        #select_distribution = 'gausshyper'
        #self.assertEqual(how_many_params([13.76, 3.12, 2.51, 5.18, 0.0, 1.0]),
        #               ('scale=1.0', 'loc=0.0', ['a', 'b', 'c', 'z'], 
        #                 ['a=13.76', 'b=3.12', 'c=2.51', 'z=5.18']))
        
        #sliders_params = [0.0, 1.0]
        #select_distribution = 'anglit'
        #self.assertEqual(how_many_params([0.0, 1.0]),
        #                ('scale=1.0', 'loc=0.0', '', ''))
    
    
    def test_get_n(self):
        name = ['a']
        self.assertEqual(get_n(name),
                         'a,')
        name = ['a', 'b', 'c']
        self.assertEqual(get_n(name),
                         'a, b, c,')
        name = [ ]
        self.assertEqual(get_n(name),
                         '')        
        
    def test_fit_data(self):

        # Just ignoring warnings        
        #import warnings
        #warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        # Need to update functions to change others
        chosen_distr = ['alpha']
        df = pd.read_csv('sample_da.csv',
                sep=None , engine='python', encoding='utf-8')
        df = df['Sample_4'].replace([np.inf, -np.inf], np.nan).dropna() 
        
        fit_dict_res = fit_data()
        
        # Fit gives the same results
        for distribution, result in fit_dict_res.items():
            self.assertEqual(result,
                             [3.3006473763113395, (34.77765122854245,), \
                              -16.46109282478973, 601.8404505786957])

        #chosen_distr = ['norm']
        #for distribution, result in fit_dict_res.items():
        #    self.assertEqual(result,
        #                     [3.013586155094246, (), 0.8721399999999999, \
        #                        0.47353354728044345])
        

        

if __name__ == "__main__":
    unittest.main()
