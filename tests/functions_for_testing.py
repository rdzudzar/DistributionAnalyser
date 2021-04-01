# -*- coding: utf-8 -*-
"""

Since streamlit pages are inferring with tests, I just create a copy
of the functions (file functions_for_testing) in order to be able to test them.
Hey its better to have mangled routine than no tests at all :)
Therefore, any changed in App functions should be copied to the this 
file and then tested.


"""
import numpy as np
from scipy import stats
import streamlit as st
np.random.seed(1)

import math
import scipy
import pandas as pd

from helper_functions import (stats_options,
                              creating_dictionaries,
                             )


select_distribution = 'alpha'
sliders_params = [3.57, 0.0, 1.0]

#sliders_params = [13.76, 3.12, 2.51, 5.18, 0.0, 1.0]
#select_distribution = 'gausshyper'

#sliders_params = [0.0, 1.0]
#select_distribution = 'anglit'


name_docstring_dict, name_eq_dict, name_proper_dict, \
        all_dist_params_dict, name_url_dict = creating_dictionaries()

def obtain_functional_data(select_distribution):
    """
    This function will create sliders (or input boxes) for
    each available parameter of the selected distribution. 
    Sliders will initiate with the parameter default value, as 
    obtained from the SciPy library. 
    Advanced options include sliders with smaller step interval, or
    input boxes if Users want to manually specify parameter values.
    """

    # all_dist_params_dict:    
    # See helper function for details; output example:
    # 'alpha': {'a': '3.57', 'loc': '0.00', 'scale': '1.00'},
    
    
    # "select_distribution" is defined streamlit selectbox
    if select_distribution in all_dist_params_dict.keys():
        sliders_params = []
        # Create slider for each parameter
        for i, param in enumerate(all_dist_params_dict[f'{select_distribution}']):
            parameter_value = float(all_dist_params_dict.get(f'{select_distribution}').get(param))
    
            # As the majority of the parameters are not defined for
            # values below 0; I will limit minimum value to 0.01.
            # If user know that they can go below, it's possible to 
            # enter those values manually.
            # Scale can not be 0 or less than
            min_param_value = 0.01
            step_value = 0.10
            def sliders():
                """
                Function that defines a slider. It's going to be
                initiated with the default value as defined in SciPy.
                Slider min value of 0.01; max value of 10 - are added
                arbitrary.
                """
    
                slider_i = st.slider('Default value: '+'{}'.format(param)+' = '+f'{parameter_value}',
                           min_value = min_param_value,
                           value = float("{:.2f}".format(parameter_value)),
                           max_value = 10.00,
                           step = step_value)
                
                return slider_i
            
            slider_i = sliders() 
            sliders_params.append(slider_i)
        
        return sliders_params
            
sliders_params = obtain_functional_data(select_distribution)

def get_multi_parameters(*c_params):
    """
    This function accepts multiple arguments which will be function 
    parameter values. Each function have 2-6 parameters, two being always
    the same: loc and scale.

    Parameters
    ----------
    *c_params : a list of parameters of the distribution function.

    Returns
    -------
    x : array of float64
        Generated evenly spaced numbers.
    r : array of float64
        Generated random numbers using the selected distribution.
    rv : frozen distribution

    """
    
    # Sample size
    size = 400
    # Current scipy functions have from 2 to 6 parameters (counting loc & 
    # scale) which will be in *c_params - as obtained from sliders/input box  

    # To be able to use shape parameters and loc/scale values
    # I just tell which are which, as loc/scale are always second to last and last        
    for j, param in enumerate(c_params):

        # Returns the value of the named attribute of an object
        dist = getattr(stats, select_distribution)

        # Generate evenly spaced numbers over a specified interval
        x = np.linspace(dist.ppf(0.001, *c_params[0][0:(len(*c_params)-2)],
                                 loc = c_params[0][-2], scale = c_params[0][-1]),
                        dist.ppf(0.999, *c_params[j][0:(len(*c_params)-2)],
                                 loc = c_params[0][-2], scale = c_params[0][-1]), size)
            

        # Create a frozen random variable "RV" using function parameters
        # It will be used to show the PDF
        rv = dist(*c_params[j][0:(len(*c_params)-2)], loc = c_params[0][-2],
                  scale = c_params[0][-1])

        # Generate random numbers using the selected distribution
        # These will be used for making histogram
        r = dist.rvs(*c_params[j][0:(len(*c_params)-2)], loc = c_params[0][-2],
                     scale = c_params[0][-1], size=size)
        
    return x, r, rv

x1, r1, rv1 = get_multi_parameters(sliders_params)



def which_s(x, r, s):
    """
    Compute standard deviation and the mean. 
    Original function will also:
    Shade between: mean-std and mean+std which shows sigma.
    """

    x01 = s*np.std(r)
    # Select only x values in between range
    x1 = x[ (x > (np.mean(r)-x01)) & (x < (x01+np.mean(r)))]
    # This will shade 1/2/3 sigma, limiting y on the PDF border
    
    return x1, x01

x, s = which_s(x1, r1, 1)

# Next two functions extract loc/scale/arg for generating
# python code
#sliders_params = [13.76, 3.12, 2.51, 5.18, 0.0, 1.0]
#select_distribution = 'gausshyper'
def how_many_params(sliders_params):
    """ Extract User selected parameter values """
    # For distribution containing only scale/loc
    if len(sliders_params) == 2:
        names = ""
        ps = ""
        scale = f'scale={sliders_params[-1]}'
        loc = f'loc={sliders_params[-2]}'
        
        return scale, loc, names, ps

    else:
        scale = f'scale={sliders_params[-1]}'
        loc = f'loc={sliders_params[-2]}'
        
        names = []
        ps = []
        for i, param in enumerate(sliders_params[0:-2]):
            param_name = stats_options().get(f'{select_distribution}').shapes.split(', ')
            name = f'{param_name[i]}'
            p = f'{param_name[i]}={param}'
            
            names.append(name)
            ps.append(p)
            
    return scale, loc, names, ps

# Get output
scale, loc, name, p = how_many_params(sliders_params)


# Extracted p is in the form: ['a=4.3', 'b=4.0'] so I need to remove [',']
a = str([i for i in p]).strip(" [] ").strip("'").replace("'", '').replace(", ",'\n')

# prints out: 
#a=3.57 for alpha
# for beta:
# a=2.31
# b=0.63
# for hypergauss:
#a=13.76
#b=3.12
#c=2.51
#z=5.18    


# I need to format output so I can get shape paramters without additional
# characters.
def get_n(name):
    # for distributions with only scale/loc
    if len(name) == 0:
        name = ''
    else:
        name = str([i for i in name]).strip(" [] ").strip("'").replace("'", '')+','
    return name
name = get_n(name)



# Need to update functions to change others
chosen_distr = ['alpha']
df = pd.read_csv('D:/Rob_work/02_Project/Streamlit/App_CDFSciPy/sample_data/sample_da.csv',
                 sep=None , engine='python', encoding='utf-8')
df = df['Sample_4']

def fit_data(df):
    """ 
    Modified from: https://stackoverflow.com/questions/6620471/fitting\
        -empirical-distribution-to-theoretical-ones-with-scipy-python 
    
    This function is performed with @cache - storing results in the local
    cache; read more: https://docs.streamlit.io/en/stable/caching.html
    """
    
    # If the distribution(s) are selected in the selectbox
    #if chosen_distr:
        
    # Check for nan/inf and remove them
    ## Get histogram of the data and histogram parameters
    num_bins = round(math.sqrt(len(df)))
    hist, bin_edges = np.histogram(df, num_bins, density=True)
    central_values = np.diff(bin_edges)*0.5 + bin_edges[:-1]

    results = {}
    for distribution in chosen_distr:
        
        # Go through distributions
        dist = getattr(scipy.stats, distribution)
        # Get distribution fitted parameters
        params = dist.fit(df)
        
        ## Separate parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
    
        ## Obtain PDFs
        pdf_values = [dist.pdf(c, loc=loc, scale=scale, *arg) for c in
                      central_values]

        # Calculate the RSS: residual sum of squares 
        # Also known as SSE: sum of squared estimate of errors
        # The sum of the squared differences between each observation\
        # and its group's mean; here: diff between distr. & data hist
        sse = np.sum(np.power(hist - pdf_values, 2.0))
        
        
        # Parse fit results 
        results[dist] = [sse, arg, loc, scale]

    # containes e.g. [0.5, (13,), 8, 1]
    # equivalent to  [sse, (arg,), loc, scale] as argument number varies
    results = {k: results[k] for k in sorted(results, key=results.get)}

    return results

fit_dict_res = fit_data(df)



