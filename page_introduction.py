# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 14:25:33 2021

@author: Robert https://github.com/rdzudzar
"""
import streamlit as st

def page_introduction():
    
    # Space so that 'About' box-text is lower
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    
    st.markdown("<h2 style='text-align: center;'> Welcome To </h2>", 
                unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'> Distribution Analyser</h1>", 
                unsafe_allow_html=True)
     

    st.info("""
            There are two main features: \n
            - Explore distributions 
            - Fit distributions  
            $←$ To start playing with the app, select an option on the 
            left sidebar.
            """)
    st.info("""
            - Here is a youtube link to the [Distribution Analyser walkthrough](https://www.youtube.com/watch?v=6S0b7gFY36I&t=3s).
            - App snippets and brief descriptions ⮧
            """)


    image1 = "https://raw.githubusercontent.com/rdzudzar/DistributionAnalyser/main/images/Dist1.png?token=AIAWV2ZQOGWADUFWZM3ZWBLAN3CD6"
    image2 = "https://raw.githubusercontent.com/rdzudzar/DistributionAnalyser/main/images/Dist2.png?token=AIAWV27IFN4ZLN3EAONHMVLAN3BNS"
    image3 = "https://raw.githubusercontent.com/rdzudzar/DistributionAnalyser/main/images/Dist3.png?token=AIAWV25DCGRPJRFLDPQIWN3AN3BPA"
    image4 = "https://raw.githubusercontent.com/rdzudzar/DistributionAnalyser/main/images/Fit1.png?token=AIAWV2ZVPX4HJL77ZQRTIBDAN3BQK"
    image5 = "https://raw.githubusercontent.com/rdzudzar/DistributionAnalyser/main/images/Fit2.png?token=AIAWV27QFQIAEOQSRDQVC3DAN3BRQ"
    image6 = "https://raw.githubusercontent.com/rdzudzar/DistributionAnalyser/main/images/Fit3.png?token=AIAWV265V2EQ24SLCTLEHOTAN3BSQ"


    
    def make_line():
        """ Line divider between images. """
            
        line = st.markdown('<hr style="border:1px solid gray"> </hr>',
                unsafe_allow_html=True)

        return line    


    # Images and brief explanations.
    st.error('Explore distributions')
    feature1, feature2 = st.columns([0.5,0.4])
    with feature1:
        st.image(image1, use_column_width=True)
    with feature2:
        st.warning('Select Distribution')
        st.info("""
                - Select distribution from Dropdown Menu (or type its name)
                - Change distribution parameters on sliders and see the change. 
                - Check created hyperlink to **SciPy** official documentation at the bottom of the sidebar.
                """)
    
    make_line()
    
    feature3, feature4 = st.columns([0.6,0.4])
    with feature3:        
        st.image(image2, use_column_width=True)
    with feature4:
        st.warning('Tweak Display')
        st.info("""
                - Pick *Dark/Light Theme*
                - Select **on/off** each option: Histogram, PDF, CDF, SF,
                boxplot, quantiles, or shade 1/2/3 $\sigma$.
                - Get Table with descriptive statistics.
                """)
    make_line()
    
    feature5, feature6 = st.columns([0.6,0.4])
    with feature5:
        st.image(image3, use_column_width=True)
    with feature6:
        st.warning('Export')
        st.info("""
                - Generate a Python code with selected distribution and 
                parameters
                - Save .py file or copy to clipboard to take it home.
                """)
    
    make_line()
    
    st.error('Fit distributions')
    feature7, feature8 = st.columns([0.4,0.6])
    with feature7:
        st.warning('Import')
        st.info("""
                - Import a **.csv** file with your own data (or get a sample).
                - Plot your data with or without basic statistical information.
                """)
    with feature8:
        st.image(image4, use_column_width=True)
    
    make_line()
    
    feature9, feature10 = st.columns([0.4,0.6])
    with feature9:
        st.warning('Fit')
        st.info("""
                - Multiselectbox: pick any number of distributions
                - **'All_distributions'** - select all
                - Fit distribution(s) to your data
                """)
    with feature10:
        st.image(image5, use_column_width=True)        
    
    make_line()
    
    feature10, feature11 = st.columns([0.4,0.6])
    with feature10:
        st.warning('Results & Export')
        st.info("""
                - Interactive **Figures**
                - **Table** with all fitted distribution(s) 
                - Generate **Python code** with best fit distribution 
                """)
    with feature11:
        st.image(image6, use_column_width=True)      
    
    make_line()
    
    st.info('There are 100 continuous distribution functions  \
                from **SciPy v1.6.1** available to play with.')
        
    st.markdown("""
                
                - Abriviations:
                
                    - PDF - Probability Density Function
                
                    - CDF - Cumulative Density Function
                
                    - SF - Survival Function
                
                    - P(X<=x) - Probability of obtaining a value smaller than 
                                x for selected x.
                
                    - Empirical quantiles for a data array: 
                        Q1, Q2, Q3 respectively 0.25, 0.50, 0.75
                              
                    - $\sigma$ (Standard Deviation). On plot shades: 
                        mean$\pm x\sigma$
                        
                    - SSE - Sum of squared estimate of errors
                
                """)
    
    return
