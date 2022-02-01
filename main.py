# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 11:54:36 2021

@author: Robert https://github.com/rdzudzar
"""
# Awesome Streamlit
import streamlit as st

# Add pages -- see those files for deatils within
from page_explore import page_explore
from page_fit import page_fit
from page_introduction import page_introduction

# Use random seed
import numpy as np
np.random.seed(1)


# Set the default elements on the sidebar
st.set_page_config(page_title='DistributionAnalyser')

logo, name = st.sidebar.columns(2)
with logo:
    image = 'https://raw.githubusercontent.com/rdzudzar/DistributionAnalyser/main/images/logo_da.png?token=AIAWV2ZRCFKYM42DVFTD3OLAN3CQK'
    st.image(image, use_column_width=True)
with name:
    st.markdown("<h1 style='text-align: left; color: grey;'> \
                Distribution Analyser </h1>", unsafe_allow_html=True)

st.sidebar.write(" ")


def main():
    """
    Register pages to Explore and Fit:
        page_introduction - contains page with images and brief explanations
        page_explore - contains various functions that allows exploration of
                        continuous distribution; plotting class and export
        page_fit - contains various functions that allows user to upload
                    their data as a .csv file and fit a distribution to data.
    """

    pages = {
        "Introduction": page_introduction,
        "Explore distributions": page_explore,
        "Fit distributions": page_fit,
    }

    st.sidebar.title("Main options")

    # Radio buttons to select desired option
    page = st.sidebar.radio("Select:", tuple(pages.keys()))
                                
    # Display the selected page with the session state
    pages[page]()

    # Write About
    st.sidebar.header("About")
    st.sidebar.warning(
            """
            Distribution Analyser app is created and maintained by astrophysicist
            **Dr Robert Dzudzar**, currently Executive Data Scientist at [**Tribalism**](https://www.tribalism.com.au/?utm_source=WebApp&utm_medium=DistributionAnalyser). 
            If you like this app please star its
            [**GitHub**](https://github.com/rdzudzar/DistributionAnalyser)
            repo, share it and feel free to open an issue if you find a bug 
            or if you want some additional features.
            """
    )


if __name__ == "__main__":
    main()
