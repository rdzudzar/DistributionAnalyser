# Distribution Analyser
|Streamlit| |Release| |MIT licensed|

Distribution Analyser is a Web App that allows you to interactively explore 
continuous distributions from SciPy and fit distribution(s) to your data.

### Main options:

- `Explore distributions` - Interactively explore continuous distribution functions from SciPy v1.6.1.
- `Fit distributions` - Import your data and fit selected continuous distribution(s) to them.

# Explore distributions

- Select distribution
- Use sliders to change distribution parameters and see how distribution changes
- Fine tunning: 
-- **Slider** can be in steps of 0.10 / 0.01
-- Possible to enter paramter values manually

## Tweak display

- Dark/Light Mode
- Figure display **on/off** options:
-- Histogram
-- PDF
-- CDF (with option P(X<=x))
-- SF
-- Boxplot
-- Quantiles (1, 2, 3)
-- Regions of 1/2/3 σ
-- Generate table with statistics

## Export

- Generate python code with your selected distribution and it's paramteters
-- Copy code to clipboard or Download **.py** file and run it locally 

# Fit distributions

- This part of the app fits selected distribution(s) to the data
-- Import your data in a **.csv** file, or download sample data
-- Choose distribution (multiselect box), has options **All_distributions**
- Show results: 
-- Interactive Figures (limited to 15 best fits)
-- Displays results in a Table (for all distributions), which can be exported as .csv file
-- Generates Python code set up with the best fit distribution: makes figure with pdf, cdf and sf


## Use Distribution Analyser localy

Clone repository and run locally with [streamlit](https://streamlit.io/):
```
$ git clone https://github.com/rdzudzar/DistributionAnalyser.git
$ cd DistributionAnalyser
$ streamlit run main.py

```

**Requirements:**
Code is written in Python 3.8.3, below are the packages which are used in the code:

> streamlit >= 0.79.0
> matplotlib >= 3.2.2
> pandas >= 1.0.5
> numpy >= 1.18.5
> scipy >= 1.6.1
> cmasher >= 1.5.10
> bokeh >= 2.2.3
> PIL >= 7.2.0

## Structure
```
|   main.py                  # Distribution Analyser page container
|   page_introduction.py     # 1st Page in the Main app
|   page_explore.py          # 2nd Page in the Main app
|   page_fit.py              # 3rd Page in the Main app
|   helper_functions.py      # Helper functions contain pre-made properties
|   readme.md                
|   
+---images                   
|      logo_da.png           # App logo
|   	  6 other .png          # Images used on the introduction page
|
+---sample_data              
|   	  sample_da.csv         # Sample data for fitting
|
+---.streamlit               
|      config.toml           # Streamlit config file, limits file upload
|       
+---tests                    
|      test_app.py           # Tests
|   	  funcs_for_testing.py  # Copy of functions that are tested
|      __init__.py          
```

## How to use

Here is the Distribution Analyser walkthrough.

## Community guidelines

**Distribution Analyser** is an open-source and free-to-use, provided under the MIT licence.
If you like Distribution Analyser, please share it, star repo and feel free to open issues for any bugs/requests.

.. |Streamlit| image:: https://static.streamlit.io/badges/streamlit_badge_black_white.svg
   :target: https://share.streamlit.io/rdzudzar/DistributionAnalyser/main.py
   :alt: Streamlit App
   
.. |Release| image:: https://img.shields.io/github/release/manodeep/DistributionAnalyser.svg
   :target: https://github.com/rdzudzar/DistributionAnalyser/releases/latest
   :alt: Latest Release

.. |MIT licensed| image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/rdzudzar/DistributionAnalyser/raw/master/LICENSE
   :alt: MIT License
   