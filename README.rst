|logo|

Distribution Analyser 
=====================

|Streamlit| |Release| |MIT licensed| 

Distribution Analyser is a Web App that allows you to interactively explore 
continuous distributions from SciPy and fit distribution(s) to your data. 
Open the app in Streamlit and explore online, or clone repo and explore locally.

How to use
----------
Here is a link to YouTube for Distribution Analyser walkthrough.
http://www.youtube.com/watch?v=6S0b7gFY36I


Main options:
=============

- `Explore distributions` - Interactively explore continuous distribution functions from SciPy v1.6.1.
- `Fit distributions` - Import your data and fit selected continuous distribution(s) to them.

Explore distributions
---------------------

- Select distribution
- Use sliders to change distribution parameters and see how distribution changes
- Fine tunning: 
    - **Slider** can be in steps of 0.10 / 0.01
    - Possible to enter paramter values manually

Tweak display
-------------

- Dark/Light Mode
- Figure display **on/off** options:
    - Histogram
    - PDF
    - CDF (with option P(X<=x))
    - SF
    - Boxplot
    - Quantiles (1, 2, 3)
    - Regions of 1/2/3 σ
    - Generate table with statistics

Export
------ 

- Generate python code with your selected distribution and it's paramteters
    - Copy code to clipboard or Download **.py** file and run it locally 

Fit distributions
-----------------

- This part of the app fits selected distribution(s) to the data
    - Import your data in a **.csv** file, or download sample data
    - Choose distribution (multiselect box), has options **All_distributions**
- Show results: 
    - Interactive Figures (limited to 15 best fits)
    - Displays results in a Table (for all distributions), which can be exported as .csv file
    - Generates Python code set up with the best fit distribution: makes figure with pdf, cdf and sf


Use Distribution Analyser localy
--------------------------------

Clone repository and run locally with Streamlit https://streamlit.io/:
::

    $ git clone https://github.com/rdzudzar/DistributionAnalyser.git
    $ cd DistributionAnalyser
    $ streamlit run main.py


**Requirements:**
-----------------
Code is written in Python 3.8.3, below are the packages which are used in the code:

- ``streamlit >= 0.79.0``
- ``matplotlib >= 3.2.2``
- ``pandas >= 1.0.5``
- ``numpy >= 1.18.5``
- ``scipy >= 1.6.1``
- ``cmasher >= 1.5.10``
- ``bokeh >= 2.2.3``

Structure
---------

.. code-block:: raw
   
   Distribution Analyser
   
   ├── main.py                  # Distribution Analyser page container
   ├── page_introduction.py     # 1st Page in the Main app
   ├── page_explore.py          # 2nd Page in the Main app
   ├── page_fit.py              # 3rd Page in the Main app
   ├── helper_functions.py      # Helper functions contain pre-made properties
   ├── README.rst
   ├── requirements.txt         # List of used packages
   └── LICENSE
   │
   ├── images
   │   ├── logo_da.png           # App logo
   │   └── 6 other .png          # Images used on the introduction page
   ├── sample_data
   │   └── sample_da.csv         # Sample data for fitting
   ├── tests
   │   ├── test_app.py           # Tests
   │   ├── funcs_for_testing.py  # Copy of functions that are tested
   │   └── __init__.py
   ├── .streamlit
   │   └── config.toml          # Streamlit config file, limits file upload



Community guidelines
--------------------

**Distribution Analyser** is an open-source and free-to-use, provided under the MIT licence.
If you like Distribution Analyser, please share it, star repo and feel free to open issues for any bugs/requests.

.. |Streamlit| image:: https://static.streamlit.io/badges/streamlit_badge_black_white.svg
   :target: https://share.streamlit.io/rdzudzar/distributionanalyser/main/main.py
   :alt: Streamlit App
   
.. |Release| image:: https://img.shields.io/github/release/rdzudzar/DistributionAnalyser.svg
   :target: https://github.com/rdzudzar/DistributionAnalyser/releases/tag/v1.0
   :alt: Latest Release

.. |MIT licensed| image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/rdzudzar/DistributionAnalyser/blob/main/LICENSE
   :alt: MIT License
   
.. |logo| image:: https://github.com/rdzudzar/DistributionAnalyser/blob/main/images/tiny_logo_da.png
   :target: https://github.com/rdzudzar/DistributionAnalyser
   :alt: DA logo
