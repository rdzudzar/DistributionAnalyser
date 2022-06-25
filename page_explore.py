# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 11:54:36 2021

@author: Robert https://github.com/rdzudzar
"""
# Package imports
import streamlit as st
from scipy import stats
from scipy.stats.mstats import mquantiles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import base64

# Helper function imports
# These are pre-computed so that they don't slow down the App
from helper_functions import (distr_selectbox_names,
                              stats_options,
                              creating_dictionaries,
                             )


def page_explore():
    """ 
    The first page in this app made with Streamlit is for an interactive 
    exploration of the continuous distributions that are available in SciPy.
    """
    
    name_docstring_dict, name_eq_dict, name_proper_dict, \
        all_dist_params_dict, name_url_dict = creating_dictionaries()
    
    def make_expanders(expander_name, sidebar=True):
        """ Set up expanders which contains a set of options. """
        if sidebar:         
            try:
                return st.sidebar.expander(expander_name)
            except:
                return st.sidebar.beta_expander(expander_name)
    
    st.sidebar.subheader("To explore:")
    with make_expanders("Select distribution"):
    

        # Distribution names
        display = distr_selectbox_names()
        
        # Create select box widget containing all SciPy function
        select_distribution = st.selectbox(
             'Click below (or type) to choose a distribution',
             display)
        
        st.markdown("**Parameters**")
                    
        def obtain_functional_data():
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
            
            # Advance mode will contain few more options
            advanced_mode = st.checkbox("Click to fine-tune parameters",  
                                        value=False)
                                    
            if advanced_mode:
                vary_parameters_mode = st.radio("Available options:",
                                            ('Slider stepping interval: 0.10',
                                             'Slider stepping interval: 0.01',
                                             'Manually input parameter values')
                                            )

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
                    
                    # Doing try and except which will allow slider stepping
                    # interval to be changed in the advanced mode.
                    try:
                        if vary_parameters_mode == 'Slider stepping interval: 0.10':
                            step_value = 0.10
                            slider_i = sliders()
                            sliders_params.append(slider_i)
                        
                        if vary_parameters_mode == 'Slider stepping interval: 0.01':
                            step_value = 0.01
                            slider_i = sliders()
                            sliders_params.append(slider_i)                                
                    
                        if vary_parameters_mode == 'Manually input parameter values':
                            manual = float(st.text_input('Default value: '+'{}'.format(param)+' = '+f'{parameter_value}', float("{:.2f}".format(parameter_value))))
                            sliders_params.append(manual)
                    except:
                        step_value = 0.10
                        slider_i = sliders()  
                        sliders_params.append(slider_i)
                
                # Add a note to user so that they know what to do in case 
                # they select a parameter value which is not valid.
                #st.markdown("**Notes**")
                
                #st.info(
                #        """
                #        To shift and/or scale the distribution use 
                #        the **loc** and **scale** parameters. In case of
                #        **Value Error**: you probably selected a shape
                #        parameter value for which a distribution is not defined
                #        (most often they can't be $$\leq$$0), in that case just
                #         select a different value.
                #        """
                #        )
                
                # For each selected distribution create a link to the 
                # official SciPy documentation page about that function.
                st.markdown("**SciPy official documentation:**")
                scipy_link = f'[{name_url_dict[select_distribution][1]}]({name_url_dict[select_distribution][0]})'

                #st.info(f"""
                #        Read more about: 
                #        [**{name_url_dict[select_distribution][1]}**]\
                #            ({name_url_dict[select_distribution][0]})
                #        """)

                st.info(f""" 
                        Read more about:
                        {scipy_link}
                        """)

                return sliders_params

        sliders_params = obtain_functional_data()
    
    
    # Generate title based on the selected distribution
    if select_distribution:
        st.markdown(f"<h1 style='text-align: center;'>{name_proper_dict[select_distribution]}</h1>", unsafe_allow_html=True)
    
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
            x = np.linspace(dist.ppf(0.001, *c_params[j][0:(len(*c_params)-2)],
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
    
    
    # Getting equations to display
    # Due to several different formatting of the equations, in order for them
    # to properly display in latex mode, I am using markdown for a few:
    if select_distribution in name_eq_dict.keys():
    
        if select_distribution == 'crystalball' \
            or select_distribution == 'f' \
            or select_distribution == 'genextreme' \
            or select_distribution == 'loglaplace':            
            st.markdown(f'{name_eq_dict[select_distribution]}')        
        else:        
            st.latex(f'{name_eq_dict[select_distribution]}')
    
    # Additional as I noticed that it takes long to compute levy_stable
    if select_distribution == 'levy_stable':
        st.write('*Note: it take longer to compute.')
    
    # Figure display properties expander
    with make_expanders("Tweak display"):
    
        st.markdown("**Select Figure Mode:**")
        plot_mode = st.radio("Options", ('Dark Mode', 'Light Mode'))   
            
        st.markdown("**What to show on the Figure?**")

        select_hist = st.checkbox('Histogram', value=True)
        
        # Put checkboxes for PDF and Shine in a column
        # If PDF is True (on): Shine can be True/False (on/off)
        # If PDF checkbox is False, remove Shine checkbox
        select_pdf, select_pdf_shine = st.columns(2)
        with select_pdf:
            select_pdf = st.checkbox('PDF', value=True)
            if select_pdf == False:
                select_pdf_shine = st.empty()
            else:
                with select_pdf_shine:
                    select_pdf_shine = st.checkbox('Shine', value=True)
        
        # Same functionality as for the PDF above
        select_cdf, select_cdf_shine = st.columns(2)
        with select_cdf:
            select_cdf = st.checkbox('CDF', value=False)
            if select_cdf == False:
                select_cdf_shine = st.empty()
            else:
                with select_cdf_shine:
                    select_cdf_shine = st.checkbox('Shine ', value=True)    

        # Show/Hide and explore 
        if select_cdf == False:
            select_mark_P = st.empty()
            x_cdf = st.empty()
        else:    
            select_mark_P = st.checkbox('P(X<=x)', value=False)        
            if select_mark_P:
                x_cdf = st.slider('Set x value to get: (x, P(X<=x))',
                      min_value = round(min(r1),2),
                      value = 0.5,
                      max_value = round(max(r1),2), 
                      step = 0.10)
        
        # Same functionality as for the PDF/CDF above
        select_sf, select_sf_shine = st.columns(2)
        with select_sf:
            select_sf = st.checkbox('SF', value=False)
            if select_sf == False:
                select_sf_shine = st.empty()
            else:
                with select_sf_shine:
                    select_sf_shine = st.checkbox('Shine   ', value=True) 

        # Show/hide boxplot
        select_boxplot = st.checkbox('Boxplot', value=True)
    
        # Show/hide quantile lines
        st.markdown("**Show quantile(s):**")
        left, middle, right = st.columns(3)
        with left: 
            q1 = st.checkbox('Q1', value=False)#, [0.25,0.5,0.75]
        with middle:
            q2 = st.checkbox('Q2', value=False)
        with right:
            q3 = st.checkbox('Q3', value=False)
        
        
        # Show/hide shaded sigma region(s)
        # Since widgets don't support latex yet, this is hacky way to add 
        # sigma next to each checkbox using columns
        st.markdown("**Shade region(s) of**")
        left_std, sig1, middle_std, sig2, right_std, sig3 = \
                                                st.columns([0.012, 0.044, 
                                                                0.02, 0.038,
                                                                0.02, 0.038])
        with sig1:
            st.markdown("1$\sigma$")
        with left_std: 
            # Need to leave name empty,as we want sigma
            s1 = st.checkbox('', value=False) 
        with sig2:
            st.markdown("2$\sigma$")
        with middle_std:
            # Need empty, with space so that generate key doesn't overlap
            s2 = st.checkbox(' ', value=False)
        with sig3:
            st.markdown("3$\sigma$")
        with right_std:
            s3 = st.checkbox('   ', value=False)
        
        # Show/hide a column with statistical information of the distribution
        st.markdown("**Generate descriptive statistics**")
        df_stat = st.checkbox('Yes', value=False)
    

    # Export options
    with make_expanders("Export:"):
        st.info("""
                Want **python script?** (It will contain: pdf, cdf, sf, 
                                        histogram and boxplot)
                """)
        export_code = st.button('Generate Python code')
        if export_code:
            st.write('*Code is below the Figure')
        
        #st.write('**Generated code will contain: pdf, cdf, sf, histogram and\
        #         boxplot*.')
            
    # A little of breathing room before I display 'About'
    st.sidebar.write("")    
    st.sidebar.write("")    
    st.sidebar.write("")    
    

    #######  I define a Figure class here #######   
    class Figure:
    
        """ Figure class: used to display and manipulate Figure props. """

        xlabel = 'X value'
        ylabel = 'Density'
    
        global_rc_params = {
            'legend.fontsize': 12,
            'legend.markerscale': 1.0,
            'axes.labelsize': 14,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'xtick.top': False,
            'xtick.labeltop': False,
            'ytick.right': False,
            'ytick.labelright': False,
            }
    
        lines = {'solid': '-',
                 'dashed': '--',
                 'dashes': [5, 10, 5, 10],
                 'dashes_r': (0, (5, 10))
                 }
    
        if plot_mode == 'Dark Mode':
    
            colors = {
                'pdf_line_color': '#fec44f',
                'hist_color': '#bdbdbd',
                'hist_edge_color': 'grey',
                'cdf_line_color': 'white',
                'frame_edge_color': '#525252',
                'boxplot_lines_color': 'white',
                'boxplot_face_color': 'black',
                'quant1_color': '#c7e9b4',
                'quant2_color': '#7fcdbb',
                'quant3_color': '#41b6c4',
                }
    
        if plot_mode == 'Light Mode':
    
            colors = {
                'pdf_line_color': '#08519c',
                'hist_color': '#525252',
                'hist_edge_color': 'grey',
                'cdf_line_color': 'black',
                'frame_edge_color': '#525252',
                'boxplot_lines_color': 'black',
                'boxplot_face_color': 'white',
                'quant1_color': '#b2182b',
                'quant2_color': '#35978f',
                'quant3_color': '#b35806',
                }
    
        def __init__(self, x, r, rv, xlabel, ylabel, 
                     plot_mode, global_rc_params, lines, colors):
            """ Set properties """
    
            self.x = x
            self.r = r
            self.rv = rv
            self.xlabel = xlabel
            self.ylabel = ylabel
            self.plot_mode = plot_mode
            self.global_rc_params = global_rc_params
            self.lines = lines
            self.colors = colors
    
        
        def display_mode(self):
            """ rcParameters for light and dark mode """
    
            plot_mode = self.plot_mode
    
            if plot_mode == 'Dark Mode':
                plt.style.use('dark_background')
                plt.rcParams['figure.facecolor'] = 'black'
    
            if plot_mode == 'Light Mode':
                plt.style.use('classic')
                plt.rcParams['figure.facecolor'] = 'white'
    
        def pdf_cdf_lines(self, ax):
            """ How to plot the PDF/CDF lines and setup of the "Shine" """
            
            # Make the line shine
            n_lines = 5
            diff_linewidth = 3
            alpha_value = 0.1
            
            # Plot the frozen PDF if checkbox is active
            if select_pdf:
                ax.plot(self.x, self.rv.pdf(self.x), linestyle='-', 
                           color = self.colors['pdf_line_color'], 
                           lw=1, label='PDF')
                # Set the shine-on if the checkbox is active
                if select_pdf_shine:
                    for n in range(1, n_lines):
                        ax.plot(self.x, self.rv.pdf(self.x), '-', 
                                color = self.colors['pdf_line_color'],
                                alpha=alpha_value, 
                                linewidth = (diff_linewidth*n)
                                )
    
            # Same as above, only for the CDF properties
            if select_cdf:
                ax.plot(self.x, self.rv.cdf(self.x), linestyle='-', 
                           color= self.colors['cdf_line_color'], 
                           lw=1, label='CDF')
        
                if select_cdf_shine:
                    for n in range(1, n_lines):
                        ax.plot(self.x, self.rv.cdf(self.x), '-', 
                                color = self.colors['cdf_line_color'],
                                alpha=alpha_value, 
                                linewidth = (diff_linewidth*n))  
                # Mark a point on the CDF
                if select_mark_P:
                    xmin, xmax = ax.get_xlim()
                    ax.vlines(x_cdf, ymin=0, ymax = self.rv.cdf(x_cdf), 
                              color=self.colors['cdf_line_color'], 
                              linestyle=':', linewidth=1)
                    ax.hlines(self.rv.cdf(x_cdf), xmin=xmin, 
                              xmax = x_cdf, color=self.colors['cdf_line_color'], 
                              linestyle=':', linewidth=1)
                    ax.annotate(f'({x_cdf:.2f}, {self.rv.cdf(x_cdf):.2f})',
                                xy = (x_cdf, self.rv.cdf(x_cdf)), 
                                color=self.colors['cdf_line_color'])
            if select_sf:
                ax.plot(self.x, self.rv.sf(self.x), linestyle='-', 
                           color= 'plum', 
                           lw=1, label='SF')
                
                if select_sf_shine:
                    for n in range(1, n_lines):
                        ax.plot(self.x, self.rv.sf(self.x), '-', 
                                color = 'plum',
                                alpha=alpha_value, 
                                linewidth = (diff_linewidth*n)) 
        
        def boxplot(self, ax):
            """ Define the boxplot properties. """
                
            bp = ax.boxplot(self.r, patch_artist=True,
                                        vert=False,
                                        notch=False,
                                        showfliers=False
                                        )
            
            for element in ['boxes', 'whiskers', 'fliers', 'means', \
                            'medians', 'caps']:
                plt.setp(bp[element], color=self.colors['boxplot_lines_color'])
            for patch in bp['boxes']:
                patch.set(facecolor=self.colors['boxplot_face_color'])  
                              
            
            # Move x label below - this will be active if boxplot is shown
            ax.set_xlabel(self.xlabel)
        
            # In addition to the global rcParams, set plot options:
            ax.spines['left'].set_visible(False)
            ax.set_yticklabels([])
            ax.set_yticks([])
            ax.set_ylim(0.9, 1.1)
            
        def quantiles(self, ax):
            """ Quantiles and their plot properties. """
            
            def get_line(self, q):
                """ Compute the quantiles and set them as vertical lines. """
                
                # Compute
                quant = mquantiles(self.r)
                
                # Plot
                ax.vlines(quant[q-1], ymin=0, ymax=self.rv.pdf(quant[q-1]), 
                       color=self.colors[f'quant{q}_color'], 
                       dashes = self.lines['dashes_r'],
                    linewidth=2, label=f'Q{q} = {quant[q-1]:.2f}',
                    zorder=0, clip_on=False)

                # Label midway
                ax.text(quant[q-1], self.rv.pdf(quant[q-1])*0.5, f'Q{q}', 
                        ha='center', fontsize=10, 
                        color=self.colors[f'quant{q}_color'])
            
            # Streamlit control - checkboxes for Q1/2/3: on/off
            if q1:
                q=1
                get_line(self, q)
    
            if q2:
                q=2
                get_line(self, q)
            if q3:
                q=3
                get_line(self, q)
    
        def sigmas(self, ax):
            """ Sigmas and their plot properties. """
            
            # Need to calculate above with the function!
            def which_s(self, s):
                """
                Compute standard deviation and the mean. 
                Shade between: mean-std and mean+std which shows sigma.
                """

                x01 = s*self.r.std()
                # Select only x values in between sigma range
                x1 = self.x[ (self.x > (self.r.mean()-x01)) & \
                            (self.x < (x01+self.r.mean()))]
                # This will shade 1/2/3 sigma, limiting y on the PDF border
                ax.fill_between(x1, self.rv.pdf(x1), 0,
                                color=self.colors['pdf_line_color'],
                                alpha=0.2)
                
                
            # Streamlit control - checkboxes for sigma1/2/3: on/off
            if s1:
                s=1
                which_s(self, s)
    
            if s2:
                s=2
                which_s(self, s)
            if s3:
                s=3
                which_s(self, s)
                

        def histogram(self, ax):
            """ Histogram properties """
            
            ax.hist(self.r, density=True, bins=20, 
                       edgecolor=self.colors['hist_edge_color'], 
                       fill = False, #hatch='x',
                       linewidth=1, alpha=1, label='Sample distribution')

        
        def get_figure(self, fig_type):
            """
            Figure layout: single figure, or two as a subplot.
            I want this, because boxplot (which is placed as subplot) 
            can be set to: on/off.
            """
            
            if fig_type == 'dual':
                fig, ax = plt.subplots(2, 1, 
                                       gridspec_kw={'height_ratios': [9, 0.7]})
                
                
                return fig, ax
            
            else:
                fig, ax = plt.subplots(1,1)
                return fig, ax
    
        def figure_display_control(self):
            """
            Set dual figure: this will have distribution and boxplot. 
            In this case we have distributions and its properties on the 
            ax[0], while if boxplot is 'on' it will be set to ax[1].
            """
    
            plt.rcParams.update(self.global_rc_params)
            
            # Streamlit control - if boxplot is true
            if select_boxplot:
                fig, ax = Figure.get_figure(self, 'dual')

                Figure.pdf_cdf_lines(self, ax=ax[0])
    
                if q1 or q2 or q3:
                    Figure.quantiles(self, ax=ax[0])

                if s1 or s2 or s3:
                    Figure.sigmas(self, ax=ax[0])

                if select_hist:
                    Figure.histogram(self, ax=ax[0])
                
                legend = ax[0].legend(bbox_to_anchor=(0,1.02,1,0.2), 
                                      loc="lower left", mode="expand", 
                                      borderaxespad=0, ncol=3)    
                legend.get_frame().set_edgecolor("#525252")
                
                # In case all distribution prop. from ax[0] are off set the
                # boxplot on the ax[0] if the boxplot is on.
                if (select_cdf == False and select_pdf == False \
                    and select_hist == False and select_sf == False):
                    
                    fig, ax = Figure.get_figure(self, 'single')

                    Figure.boxplot(self, ax=ax)
                    
                else:

                    Figure.boxplot(self, ax=ax[1])  

                    # Get xlim from the upper image and port it to the lower 
                    # as we want to have x axis of the distributions and 
                    # boxplot aligned.
                    ax[1].set_xlim(ax[0].get_xlim())
                        
                    # Move y label to apropriate ax.
                    ax[0].set_ylabel(self.ylabel)

                
            else:
                # Single fig. mode
                fig, ax = Figure.get_figure(self, 'single')
                
                Figure.pdf_cdf_lines(self, ax=ax)
    
                if select_hist:
                    Figure.histogram(self, ax=ax)
    
                if q1 or q2 or q3:
                    Figure.quantiles(self, ax=ax)
    
                if s1 or s2 or s3:
                    Figure.sigmas(self, ax=ax)
    
                ax.set_xlabel(self.xlabel)
                ax.set_ylabel(self.ylabel)
                
                legend = ax.legend(bbox_to_anchor=(0,1.02,1,0.2), 
                                      loc="lower left", mode="expand", 
                                      borderaxespad=0, ncol=3)    
                legend.get_frame().set_edgecolor("#525252")
            
            # If nothing is selected from the 'What to show on the Figure'    
            if (select_cdf == False and select_pdf == False \
                    and select_hist == False and select_boxplot == False \
                        and select_sf == False):
                
                fig, ax = Figure.get_figure(self, 'single')
                
                ax.text(0.1, 0.5, 'Tabula rasa',
                        va='center', fontsize=20)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.set_yticklabels([])
                ax.set_yticks([])     
                ax.set_xticklabels([])
                ax.set_xticks([]) 
                
            return fig
    
    def port_to_streamlit():
        """
        Use the Figure class to obtain what will be displayed on the plot.
        Show Figure using Streamlit widget: pyplot
        """
        
        p = Figure(
            x1,
            r1,
            rv1,
            Figure.xlabel,
            Figure.ylabel,
            plot_mode,
            Figure.global_rc_params,
            Figure.lines,
            Figure.colors,
            )
        
        # To obtain whether Light/Dark Mode
        Figure.display_mode(p)
    
        # Parse to Streamlit
        st.pyplot(p.figure_display_control())

            
    # Output statistics into a Table
    def df_generate_statistics(r1):
        """
        Compute statistical information of the created distribution.
        Parse it to the Pandas dataframe.
        
        Use r1: array of float64 - generated random numbers using the 
                selected distribution.
        """
        
        df_data = pd.DataFrame(r1)
        stats = df_data.describe()
        stats.loc['var'] = df_data.var().tolist()
        stats.loc['skew'] = df_data.skew().tolist()
        stats.loc['kurt'] = df_data.kurtosis().tolist()

        # Parse to Streamlit
        st.dataframe(stats.rename(columns={0: 'Value'}))
    
    # Radio buttons that control Dark/Light mode
    if plot_mode == 'Dark Mode':
        port_to_streamlit()
    
    if plot_mode == 'Light Mode':
        port_to_streamlit()
        
    # Streamlit on/off to show/hide statistical information
    if df_stat: 
        df_generate_statistics(r1)
    
    
    #### Generate Python code ###
        
    def how_many_params():
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
    scale, loc, name, p = how_many_params()

    # Extracted p is in the form: ['a=4.3', 'b=4.0'] so I need to remove [',']
    a = str([i for i in p]).strip(" [] ").strip("'").replace("'", '').replace(", ",'\n')

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


    # To generate code, I pass: {loc} and {scale} which will have their
    # name and values, and {a} which will be shape parameters - they can
    # be many and each will be printed out in the new line
    # {select_distribution} - is passed from the slider
    # {name} - contains only name of the shape parameters, without values
    # Have to place un-indented, otherwise py code will be indented
    generate_code = f"""
# -*- coding: utf-8 -*-
# Generated using Distribution Analyser:
# https://github.com/rdzudzar/DistributionAnalyser
# {time.strftime("%Y%m%d_%H%M%S")}
# ---

import matplotlib.pyplot as plt #v3.2.2
import numpy as np #v1.18.5
from scipy.stats import {select_distribution} #v1.6.1

# Set random seed
np.random.seed(1)

# Function parameters
{scale}
{loc}
{a}              
     
# Generate evenly spaced numbers over a specified interval
size = 400
x = np.linspace({select_distribution}.ppf(0.001, {name} loc=loc, scale=scale ), 
                {select_distribution}.ppf(0.999, {name} loc=loc, scale=scale ),
                size)

# Freeze the distribution
rv = {select_distribution}({name} loc=loc, scale=scale)

# Generate random numbers
r = {select_distribution}.rvs({name} loc=loc, scale=scale, size=size)

# Make a plot
fig, ax = plt.subplots(2, 1,
                       gridspec_kw={{'height_ratios': [9, 0.7]}})

# Plot PDF, CDF and SF
ax[0].plot(x, rv.pdf(x), linestyle='-', color='#3182bd', lw=3, label='PDF')
ax[0].plot(x, rv.cdf(x), linestyle='-', color='k', lw=3, label='CDF')
ax[0].plot(x, rv.sf(x), linestyle='-', color='#df65b0', lw=3, label='SF')

# Plot Histogram
ax[0].hist(r, density=True, bins=20, color='lightgrey',
           edgecolor='k', label='Sample')

# Plot Boxplot
bp = ax[1].boxplot(r, patch_artist=True,
           vert=False,
           notch=False,
           showfliers=False,
           ) 
# Boxplot aestetic
for median in bp['medians']: median.set(color ='red', linewidth = 2) 
for patch in bp['boxes']: patch.set(facecolor='white') 

# Set legend
ax[0].legend(bbox_to_anchor=(0,1.1,1,0.2), 
             loc="center", 
             borderaxespad=0, ncol=2)

# Set Figure aestetics
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].set_title('Distribution: {select_distribution}')

ax[1].set_xlim(ax[0].get_xlim())
ax[1].set_ylim(0.9, 1.1)
ax[1].spines['left'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].set_yticklabels([])
ax[1].set_yticks([])

ax[1].set_xlabel('X value')
ax[0].set_ylabel('Density')

plt.show()

"""

    def get_code():
        """ Prints out the python formatted code"""
        
        st.code(f"{generate_code}")
    
    
    def py_file_downloader(py_file_text):
        """
        Strings <-> bytes conversions and creating a link which will
        download generated python script.
        """

        # Add a timestamp to the name of the saved file
        time_stamp = time.strftime("%Y%m%d_%H%M%S")

        # Base64 takes care of porting info to the data
        b64 = base64.b64encode(py_file_text.encode()).decode()
        
        # Saved file will have distribution name and the timestamp
        code_file = f"{select_distribution}_{time_stamp}.py"
        #st.markdown(f'** Download Python File **: \
        #            <a href="data:file/txt;base64,{b64}" \
        #                download="{code_file}">Click Here</a>', 
        #                unsafe_allow_html=True)
    
        st.download_button(
            label = 'Download .py file',
            data = f'{py_file_text}',
            file_name = f'{code_file}',
            mime = 'application/octet-stream')

    # Press the button to get the python code and download hyperlink option
    if export_code:
        get_code()
        

        py_file_downloader(f"{generate_code}")
