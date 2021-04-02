# -*- coding: utf-8 -*-

"""
@author: Robert https://github.com/rdzudzar

This file contains pre-computed items: such as equations, distribution names,
distribution proper names and parameters of the distribution, and links to
access documentation page for each distribution. 

This items are pre-computed to reduce the load of the app.
Functions which are used to obtain these are at the bottom of the lists.
* Note, as indicated, due to a certain reasons some outputs are manually
modified. For example, scraping - there are some duplicates functions that
are explained with multiple parameters, therefore these had to removed.

"""
from scipy import stats 
#from bs4 import BeautifulSoup 
from six.moves import urllib




def scipy_distribution_equations():
    """
    Scraped distribution equations from the SciPy website. Duplicates are 
    removed.
    
    """
    
    mod_equations = [
         'f(x, a) = \\frac{1}{x^2 \\Phi(a) \\sqrt{2\\pi}} * \\exp(-\\frac{1}{2} (a-1/x)^2)',
         'f(x) = \\sin(2x + \\pi/2) = \\cos(2x)',
         'f(x) = \\frac{1}{\\pi \\sqrt{x (1-x)}}',
         'f(x, \\chi) = \\frac{\\chi^3}{\\sqrt{2\\pi} \\Psi(\\chi)} x \\sqrt{1-x^2}  \\exp(-\\chi^2 (1 - x^2)/2)',
         'f(x, a, b) = \\frac{\\Gamma(a+b) x^{a-1} (1-x)^{b-1}}  {\\Gamma(a) \\Gamma(b)}',
         'f(x, a, b) = \\frac{x^{a-1} (1+x)^{-a-b}}{\\beta(a, b)}',
         'f(x, c) = \\frac{c}{\\log(1+c) (1+cx)}',
         'f(x, c, d) = c d x^{-c - 1} / (1 + x^{-c})^{d + 1}',
         'f(x, c, d) = c d x^{c-1} / (1 + x^c)^{d + 1}',
         'f(x) = \\frac{1}{\\pi (1 + x^2)}',
         'f(x, k) = \\frac{1}{2^{k/2-1} \\Gamma \\left( k/2 \\right)}  x^{k-1} \\exp \\left( -x^2/2 \\right)',
         'f(x, k) = \\frac{1}{2^{k/2} \\Gamma \\left( k/2 \\right)}  x^{k/2-1} \\exp \\left( -x/2 \\right)',
         'f(x) = \\frac{1}{2\\pi} (1+\\cos(x))',
         (r"$f(x, \beta, m) = \begin{cases} N \exp(-x^2 / 2) , & \text{for } x > - \beta \\  N A (B - x)^{-m}, & \text{for } x < -\beta \end{cases}$"),
         'f(x, a) = \\frac{1}{2\\Gamma(a)} |x|^{a-1} \\exp(-|x|)',
         'f(x, c) = c / 2 |x|^{c-1} \\exp(-|x|^c)',
         'f(x, a) = \\frac{x^{a-1} e^{-x}}{\\Gamma(a)}',
         'f(x) = \\exp(-x)',
         'f(x, K) = \\frac{1}{2K} \\exp\\left(\\frac{1}{2 K^2} - x / K \\right)  \\text{erfc}\\left(-\\frac{x - 1/K}{\\sqrt{2}}\\right)',
         'f(x, b) = b x^{b-1} \\exp(1 + x^b - \\exp(x^b))',
         'f(x, a, c) = a c [1-\\exp(-x^c)]^{a-1} \\exp(-x^c) x^{c-1}',
         (r"$f(x, df_1, df_2) = \frac{ df_2^{df_2/2} df_1^{df_1/2} x^{df_1 / 2-1}}{ (df_2+df_1 x)^{(df_1+df_2)/2} B(df_1/2, df_2/2)}$"),
         'f(x, c) = \\frac{x+1}{2c\\sqrt{2\\pi x^3}} \\exp(-\\frac{(x-1)^2}{2x c^2})',
         'f(x, c) = c x^{-c-1} (1 + x^{-c})^{-2}',
         'f(x, c) = \\frac{1}{\\pi (1+(x-c)^2)} + \\frac{1}{\\pi (1+(x+c)^2)}',
         'f(x, c) = \\sqrt{2/\\pi} cosh(c x) \\exp(-\\frac{x^2+c^2}{2})',
         'f(x, a) = \\frac{x^{a-1} e^{-x}}{\\Gamma(a)}',
         'f(x, a, b, c, z) = C x^{a-1} (1-x)^{b-1} (1+zx)^{-c}',
         'f(x, a, b, c) = (a + b (1 - \\exp(-c x)))  \\exp(-a x - b x + \\frac{b}{c}  (1-\\exp(-c x)))',
          (r"$f(x, c) = \begin{cases} \exp(-\exp(-x)) \exp(-x) ,  & \text{for } c = 0 \\  \exp(-(1-c x)^{1/c}) (1-c x)^{1/c-1},  & \text{for }   x \leq 1/c,  c > 0  \end{cases}$"),
         'f(x, a, c) = \\frac{|c| x^{c a-1} \\exp(-x^c)}{\\Gamma(a)}',
         #'f(x, a, c) = \\frac{|c| x^{c a-1} \\exp(-x^c)}{\\Gamma(a)}',
         'f(x, c) = \\frac{2 (1 - c x)^{1/(c-1)}}{[1 + (1 - c x)^{1/c}]^2}',
         'f(x, p, b) = x^{p-1} \\exp(-b (x + 1/x) / 2) / (2 K_p(b))',
         'f(x, c) = c \\frac{\\exp(-x)}  {(1 + \\exp(-x))^{c+1}}',
         'f(x, \\beta) = \\frac{\\beta}{2 \\Gamma(1/\\beta)} \\exp(-|x|^\\beta)',
         'f(x, c) = (1 + c x)^{-1 – 1/c}',
         'f(x) = \\frac{1}{x \\sqrt{2\\pi}} \\exp(-\\frac{1}{2} (\\log(x))^2)',
         'f(x, c) = c \\exp(x) \\exp(-c (e^x-1))',
         'f(x) = \\exp(x - e^x)',
         'f(x) = \\exp(-(x + e^{-x}))',
         'f(x) = \\frac{2}{\\pi (1 + x^2)}',
         'f(x, \\beta) = \\frac{\\beta}{\\Gamma(1/\\beta)} \\exp(-|x|^\\beta)',
         'f(x) = \\frac{ 2 e^{-x} }{ (1+e^{-x})^2 }             = \\frac{1}{2} \\text{sech}(x/2)^2',
         'f(x) = \\sqrt{2/\\pi} \\exp(-x^2 / 2)',
         'f(x) = \\frac{1}{\\pi} \\text{sech}(x)',
         'f(x, a) = \\frac{x^{-a-1}}{\\Gamma(a)} \\exp(-\\frac{1}{x})',
         'f(x, \\mu) = \\frac{1}{\\sqrt{2 \\pi x^3}}  \\exp(-\\frac{(x-\\mu)^2}{2 x \\mu^2})',
         'f(x, c) = c x^{-c-1} \\exp(-x^{-c})',
         'f(x, a, b) = \\frac{b}{x(1-x)}  \\phi(a + b \\log \\frac{x}{1-x} )',
         'f(x, a, b) = \\frac{b}{\\sqrt{x^2 + 1}}  \\phi(a + b \\log(x + \\sqrt{x^2 + 1}))',
         'f(x, a) = a (a + x^a)^{-(a + 1)/a}',
         #'f(x, h, k) = (1 - k x)^{1/k - 1} (1 - h (1 - k x)^{1/k})^{1/h-1}',
         #'f(x, h, k) = (1 - k x)^{1/k - 1} (1 - h (1 - k x)^{1/k})^{1/h-1}',
         #'f(x, h, k) = (1 - k x)^{1/k - 1} (1 - h (1 - k x)^{1/k})^{1/h-1}',
         'f(x, h, k) = (1 - k x)^{1/k - 1} (1 - h (1 - k x)^{1/k})^{1/h-1}',
         'D_n^+ = \\text{sup}_x (F_n(x) - F(x)), \\\\ D_n^- = \\text{sup}_x (F(x) - F_n(x)),\\\\',
         'D_n = \\text{sup}_x |F_n(x) - F(x)|',
         'D_n = \\text{sup}_x |F_n(x) - F(x)|',
         'f(x) = \\frac{1}{2} \\exp(-|x|)',
         '(x, \\kappa) = \\frac{1}{\\kappa+\\kappa^{-1}}\\exp(-x\\kappa),\\quad x\\ge0\\\\   = \\frac{1}{\\kappa+\\kappa^{-1}}\\exp(x/\\kappa),\\quad x<0\\\\',
         'f(x) = \\frac{1}{\\sqrt{2\\pi x^3}} \\exp\\left(-\\frac{1}{2x}\\right)',
        'f(x) = \\frac{1}{|x| \\sqrt{2\\pi |x|}} \\exp{ \\left(-\\frac{1}{2|x|} \\right)}',
         'f(x) = \\frac{1}{2\\pi}\\int_{-\\infty}^\\infty \\varphi(t)e^{-ixt}\\,dt',
         'f(x, c) = \\frac{\\exp(c x - \\exp(x))}                       {\\Gamma(c)}',
         'f(x) = \\frac{\\exp(-x)} {(1+\\exp(-x))^2}',
         (r"$f(x, c) = \begin{cases}\frac{c}{2} x^{ c-1}  & \text{for } 0 < x  < 1 \\ \frac{c}{2} x^{-c-1}  & \text{for } x \ge 1 \end{cases}$"),
         'f(x, s) = \\frac{1}{s x \\sqrt{2\\pi}}                  \\exp\\left(-\\frac{\\log^2(x)}{2s^2}\\right)',
         'f(x, a, b) = \\frac{1}{x \\log(b/a)}',
         'f(x, c) = \\frac{c}{(1+x)^{c+1}}',
         'f(x) = \\sqrt{2/\\pi}x^2 \\exp(-x^2/2)',
         'f(x, k, s) = \\frac{k x^{k-1}}{(1+x^s)^{1+k/s}}',
         'f(x) = \\exp(-(x + \\exp(-x))/2) / \\sqrt{2\\pi}',
         'f(x, \\nu) = \\frac{2 \\nu^\\nu}{\\Gamma(\\nu)} x^{2\\nu-1} \\exp(-\\nu x^2)',
         "f(x, n_1, n_2, \\lambda) = \\exp\\left(\\frac{\\lambda}{2} + \\lambda n_1 \\frac{x}{2(n_1 x + n_2)} \\right)  n_1^{n_1/2} n_2^{n_2/2} x^{n_1/2 - 1} \\\\                    (n_2 + n_1 x)^{-(n_1 + n_2)/2} \\gamma(n_1/2) \\gamma(1 + n_2/2) \\\\                    \\frac{L^{\\frac{n_1}{2}-1}_{n_2/2} \\left(-\\lambda n_1 \\frac{x}{2(n_1 x + n_2)}\\right)} {B(n_1/2, n_2/2) \\gamma\\left(\\frac{n_1 + n_2}{2}\\right)} \\\\",
         'X = \\frac{Y + c}{\\sqrt{V/k}}',
         'f(x, k, \\lambda) = \\frac{1}{2} \\exp(-(\\lambda+x)/2)            (x/\\lambda)^{(k-2)/4}  I_{(k-2)/2}(\\sqrt{\\lambda x})',
         'f(x) = \\frac{\\exp(-x^2/2)}{\\sqrt{2\\pi}}',
         'f(x, a, b) = \\frac{a \\, K_1(a \\sqrt{1 + x^2})}{\\pi \\sqrt{1 + x^2}} \\,                     \\exp(\\sqrt{a^2 - b^2} + b x)',
         'f(x, b) = \\frac{b}{x^{b+1}}',
         'f(x, \\kappa) = \\frac{|\\beta|}{\\Gamma(\\alpha)}  (\\beta (x - \\zeta))^{\\alpha - 1} \\exp(-\\beta (x - \\zeta))',
         'f(x, a) = a x^{a-1}',
         'f(x, c, s) = \\frac{c}{x s} \\phi(\\log(x)/s)                     (\\Phi(-\\log(x)/s))^{c-1}',
         'f(x, c) = c \\phi(x) (\\Phi(-x))^{c-1}',
         'f(x) = x \\exp(-x^2/2)',
         'f(x, c) = \\frac{(1-x^2)^{c/2-1}}{B(1/2, c/2)}',
         'f(x, \\mu) = \\frac{1}{\\sqrt{2\\pi x}}  \\exp\\left(\\frac{-(1-\\mu x)^2}{2\\mu^2x}\\right)',
         'f(x, a, b) = \\frac{1}{x \\log(b/a)}',
         'f(x, b) = x \\exp(- \\frac{x^2 + b^2}{2}) I_0(x b)',
         'f(x) = \\frac{2}{\\pi} \\sqrt{1-x^2}',
        'f(x) = {\\frac {2}{\\omega {\\sqrt {2\\pi }}}} \\exp^{-{\\frac {(x-\\xi )^{2}}{2\\omega ^{2}}}}\\int _{-\\infty }^{\\alpha \\left({\\frac {x-\\xi }{\\omega }}\\right)}{\\frac {1}{\\sqrt {2\\pi }}} \\exp^{-{\\frac {t^{2}}{2}}}\ dt',
         'f(x, \\nu) = \\frac{\\Gamma((\\nu+1)/2)}  {\\sqrt{\\pi \\nu} \\Gamma(\\nu/2)} (1+x^2/\\nu)^{-(\\nu+1)/2}',
        '\\text{An instance of the rvcontinuous class}',
        '\\text{An instance of the rvcontinuous class}',
         'f(x, b) = \\frac{\\exp(-x)}{1 - \\exp(-b)}',
        '\\text{An instance of the rvcontinuous class}',
        '\\text{An instance of the rvcontinuous class}',
        #'\\text{An instance of the rvcontinuous class}',
        '\\text{An instance of the rvcontinuous class}',
        'f(x, \\kappa) = \\frac{ \\exp(\\kappa \\cos(x)) }{ 2 \\pi I_0(\\kappa) }',
         'f(x, \\kappa) = \\frac{ \\exp(\\kappa \\cos(x)) }{ 2 \\pi I_0(\\kappa) }',
         'f(x) = \\frac{1}{\\sqrt{2\\pi x^3}} \\exp(- \\frac{ (x-1)^2 }{ 2x })',
         'f(x, c) = c (-x)^{c-1} \\exp(-(-x)^c)',
         'f(x, c) = c x^{c-1} \\exp(-x^c)',
         'f(x, c) = \\frac{1-c^2}{2\\pi (1+c^2 - 2c \\cos(x))}',
         ]
         
    return mod_equations



def scipy_distribution_proper_names():
    """
    Scraped proper names.Duplicates removed.
    """


    names = [
            'Alpha distribution',
            'Anglit distribution',
            'Arcsine distribution',
            'Argus distribution',
            'Beta',
            'Beta prime',
            'Bradford',
            'Burr (Type III)',
            'Burr (Type XII)',
            'Cauchy',
            'Chi',
            'Chi-squared',
            'Cosine',
            'Crystalball distribution',
            'Double gamma',
            'Double Weibull',
            'Erlang',
            'Exponential',
            'Exponentially modified Normal',
            'Exponential power',
            'Exponentiated Weibull',
            'F',
            'Fatigue-life (Birnbaum-Saunders)',
            'Fisk',
            'Folded Cauchy',
            'Folded normal',
            'Gamma',
            'Gauss hypergeometric',
            'Generalized exponential',
            'Generalized extreme value',
            'Generalized gamma',
            #'Generalized gamma',
            'Generalized half-logistic',
            'Generalized Inverse Gaussian',
            'Generalized logistic',
            'Generalized normal',
            'Generalized Pareto',
            'Gilbrat',
            'Gompertz (or truncated Gumbel)',
            'Left-skewed Gumbel',
            'Right-skewed Gumbel',
            'Half-Cauchy',
            'The upper half of a generalized normal',
            'Half-logistic',
            'Half-normal',
            'Hyperbolic secant',
            'Inverted gamma',
            'Inverse Gaussian',
            'Inverted Weibull',
            'Johnson SB',
            'Johnson SU',
            'Kappa 3 parameter distribution',
            'Kappa 4 parameter distribution',
            #'Kappa 4 parameter distribution',
            #'Kappa 4 parameter distribution',
            #'Kappa 4 parameter distribution',
            'Kolmogorov-Smirnov one-sided test statistic distribution',
            'Kolmogorov-Smirnov two-sided test statistic distribution',
            'Limiting distribution of scaled Kolmogorov-Smirnov two-sided test statistic',
            'Laplace',
            'Asymmetric Laplace',
            'Levy',
            'Left-skewed Levy',
            'Levy-stable',
            'Log gamma',
            'Logistic (or Sech-squared)',
            'Log-Laplace',
            'Lognormal',
            'Loguniform or reciprocal',
            'Lomax (Pareto of the second kind)',
            'Maxwell',
            'Mielke Beta-Kappa / Dagum',
            'Moyal',
            'Nakagami',
            'Non-central F distribution',
            "Non-central Student's t",
            'Non-central chi-squared',
            'Normal',
            'Normal Inverse Gaussian',
            'Pareto',
            'Pearson type III',
            'Power-function',
            'Power log-normal',
            'Power normal',
            'Rayleigh',
            'R-distributed (symmetric beta)',
            'Reciprocal inverse Gaussian',
            'Loguniform or reciprocal',
            'Rice',
            'Semicircular',
            'Skew-normal random variable',
            "Student's t",
            'Trapezoidal',
            'Triangular',
            'Truncated exponential',
            'Truncated normal',
            #'Truncated normal',
            'Tukey-Lamdba',
            'Uniform',
            'Von Mises',
            'Von Mises',
            'Wald',
            'Weibull maximum',
            'Weibull minimum',
            'Wrapped Cauchy'
                ]

    return names

def distr_selectbox_names():
    """
    Accessing stats.name. 
    """
    
    names =  ['alpha',
             'anglit',
             'arcsine',
             'argus',
             'beta',
             'betaprime',
             'bradford',
             'burr',
             'burr12',
             'cauchy',
             'chi',
             'chi2',
             'cosine',
             'crystalball',
             'dgamma',
             'dweibull',
             'erlang',
             'expon',
             'exponnorm',
             'exponpow',
             'exponweib',
             'f',
             'fatiguelife',
             'fisk',
             'foldcauchy',
             'foldnorm',
             'gamma',
             'gausshyper',
             'genexpon',
             'genextreme',
             'gengamma',
             #'gengamma',
             'genhalflogistic',
             'geninvgauss',
             'genlogistic',
             'gennorm',
             'genpareto',
             'gilbrat',
             'gompertz',
             'gumbel_l',
             'gumbel_r',
             'halfcauchy',
             'halfgennorm',
             'halflogistic',
             'halfnorm',
             'hypsecant',
             'invgamma',
             'invgauss',
             'invweibull',
             'johnsonsb',
             'johnsonsu',
             'kappa3',
             'kappa4',
             #'kappa4',
             #'kappa4',
             #'kappa4',
             'ksone',
             'kstwo',
             'kstwobign',
             'laplace',
             'laplace_asymmetric',
             'levy',
             'levy_l',
             'levy_stable',
             'loggamma',
             'logistic',
             'loglaplace',
             'lognorm',
             'loguniform',
             'lomax',
             'maxwell',
             'mielke',
             'moyal',
             'nakagami',
             'ncf',
             'nct',
             'ncx2',
             'norm',
             'norminvgauss',
             'pareto',
             'pearson3',
             'powerlaw',
             'powerlognorm',
             'powernorm',
             'rayleigh',
             'rdist',
             'recipinvgauss',
             'reciprocal',
             'rice',
             'semicircular',
             'skewnorm',
             't',
             'trapezoid',
             'triang',
             'truncexpon',
             'truncnorm',
             #'truncnorm',
             'tukeylambda',
             'uniform',
             'vonmises',
             'vonmises_line',
             'wald',
             'weibull_max',
             'weibull_min',
             'wrapcauchy']
   
    return names


def extracted_scipy_urls():
    """ 
    Scraped url links to each distribution.
    """
    
    
    all_url_links = ['https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.alpha.html#scipy.stats.alpha',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anglit.html#scipy.stats.anglit', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.arcsine.html#scipy.stats.arcsine', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.argus.html#scipy.stats.argus', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html#scipy.stats.beta', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betaprime.html#scipy.stats.betaprime',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bradford.html#scipy.stats.bradford',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.burr.html#scipy.stats.burr', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.burr12.html#scipy.stats.burr12',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.cauchy.html#scipy.stats.cauchy', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi.html#scipy.stats.chi', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html#scipy.stats.chi2', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.cosine.html#scipy.stats.cosine', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.crystalball.html#scipy.stats.crystalball',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dgamma.html#scipy.stats.dgamma',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dweibull.html#scipy.stats.dweibull',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.erlang.html#scipy.stats.erlang', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html#scipy.stats.expon', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.exponnorm.html#scipy.stats.exponnorm',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.exponpow.html#scipy.stats.exponpow',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.exponweib.html#scipy.stats.exponweib', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f.html#scipy.stats.f', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fatiguelife.html#scipy.stats.fatiguelife',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisk.html#scipy.stats.fisk', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.foldcauchy.html#scipy.stats.foldcauchy', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.foldnorm.html#scipy.stats.foldnorm',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html#scipy.stats.gamma',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gausshyper.html#scipy.stats.gausshyper',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genexpon.html#scipy.stats.genexpon',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genextreme.html#scipy.stats.genextreme',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gengamma.html#scipy.stats.gengamma', 
     #'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gengamma.html#scipy.stats.gengamma',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genhalflogistic.html#scipy.stats.genhalflogistic', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.geninvgauss.html#scipy.stats.geninvgauss',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genlogistic.html#scipy.stats.genlogistic', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gennorm.html#scipy.stats.gennorm', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genpareto.html#scipy.stats.genpareto', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gilbrat.html#scipy.stats.gilbrat', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gompertz.html#scipy.stats.gompertz', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gumbel_l.html#scipy.stats.gumbel_l',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gumbel_r.html#scipy.stats.gumbel_r',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.halfcauchy.html#scipy.stats.halfcauchy',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.halfgennorm.html#scipy.stats.halfgennorm',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.halflogistic.html#scipy.stats.halflogistic',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.halfnorm.html#scipy.stats.halfnorm',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypsecant.html#scipy.stats.hypsecant', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invgamma.html#scipy.stats.invgamma',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invgauss.html#scipy.stats.invgauss', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invweibull.html#scipy.stats.invweibull',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.johnsonsb.html#scipy.stats.johnsonsb',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.johnsonsu.html#scipy.stats.johnsonsu', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kappa3.html#scipy.stats.kappa3', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kappa4.html#scipy.stats.kappa4', 
     #'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kappa4.html#scipy.stats.kappa4', 
     #'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kappa4.html#scipy.stats.kappa4', 
     #'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kappa4.html#scipy.stats.kappa4', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ksone.html#scipy.stats.ksone', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstwo.html#scipy.stats.kstwo', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstwobign.html#scipy.stats.kstwobign',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.laplace.html#scipy.stats.laplace', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.laplace_asymmetric.html#scipy.stats.laplace_asymmetric', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levy.html#scipy.stats.levy', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levy_l.html#scipy.stats.levy_l',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levy_stable.html#scipy.stats.levy_stable',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.loggamma.html#scipy.stats.loggamma',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.logistic.html#scipy.stats.logistic',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.loglaplace.html#scipy.stats.loglaplace', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html#scipy.stats.lognorm',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.loguniform.html#scipy.stats.loguniform',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lomax.html#scipy.stats.lomax',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.maxwell.html#scipy.stats.maxwell',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mielke.html#scipy.stats.mielke',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.moyal.html#scipy.stats.moyal',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nakagami.html#scipy.stats.nakagami',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ncf.html#scipy.stats.ncf', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html#scipy.stats.nct', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ncx2.html#scipy.stats.ncx2', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html#scipy.stats.norm', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norminvgauss.html#scipy.stats.norminvgauss',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pareto.html#scipy.stats.pareto', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearson3.html#scipy.stats.pearson3', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.powerlaw.html#scipy.stats.powerlaw',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.powerlognorm.html#scipy.stats.powerlognorm',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.powernorm.html#scipy.stats.powernorm',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rayleigh.html#scipy.stats.rayleigh', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rdist.html#scipy.stats.rdist',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.recipinvgauss.html#scipy.stats.recipinvgauss',
     'https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.reciprocal.html',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rice.html#scipy.stats.rice',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.semicircular.html#scipy.stats.semicircular',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewnorm.html#scipy.stats.skewnorm',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html#scipy.stats.t', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.trapezoid.html#scipy.stats.trapezoid',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.triang.html#scipy.stats.triang',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncexpon.html#scipy.stats.truncexpon',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html#scipy.stats.truncnorm', 
     #'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html#scipy.stats.truncnorm', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.tukeylambda.html#scipy.stats.tukeylambda', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html#scipy.stats.uniform', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.vonmises.html#scipy.stats.vonmises',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.vonmises_line.html#scipy.stats.vonmises_line',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wald.html#scipy.stats.wald', 
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_max.html#scipy.stats.weibull_max',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html#scipy.stats.weibull_min',
     'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wrapcauchy.html#scipy.stats.wrapcauchy']
   
    return all_url_links

def obtained_all_dist_params_dict():
    """
    Dictionary with all continuous distributions and their default parameters.
    
    """
    
    obtained_dict = {
            'alpha': {'a': '3.57', 'loc': '0.00', 'scale': '1.00'},
             'anglit': {'loc': '0.00', 'scale': '1.00'},
             'arcsine': {'loc': '0.00', 'scale': '1.00'},
             'argus': {'chi': '1.00', 'loc': '0.00', 'scale': '1.00'},
             'beta': {'a': '2.31', 'b': '0.63', 'loc': '0.00', 'scale': '1.00'},
             'betaprime': {'a': '5.00', 'b': '6.00', 'loc': '0.00', 'scale': '1.00'},
             'bradford': {'c': '0.30', 'loc': '0.00', 'scale': '1.00'},
             'burr': {'c': '10.50', 'd': '4.30', 'loc': '0.00', 'scale': '1.00'},
             'burr12': {'c': '10.00', 'd': '4.00', 'loc': '0.00', 'scale': '1.00'},
             'cauchy': {'loc': '0.00', 'scale': '1.00'},
             'chi': {'df': '78.00', 'loc': '0.00', 'scale': '1.00'},
             'chi2': {'df': '55.00', 'loc': '0.00', 'scale': '1.00'},
             'cosine': {'loc': '0.00', 'scale': '1.00'},
             'crystalball': {'beta': '2.00', 'm': '3.00', 'loc': '0.00', 'scale': '1.00'},
             'dgamma': {'a': '1.10', 'loc': '0.00', 'scale': '1.00'},
             'dweibull': {'c': '2.07', 'loc': '0.00', 'scale': '1.00'},
             'erlang': {'a': '10.00', 'loc': '0.00', 'scale': '1.00'},
             'expon': {'loc': '0.00', 'scale': '1.00'},
             'exponnorm': {'K': '1.50', 'loc': '0.00', 'scale': '1.00'},
             'exponpow': {'b': '2.70', 'loc': '0.00', 'scale': '1.00'},
             'exponweib': {'a': '2.89', 'c': '1.95', 'loc': '0.00', 'scale': '1.00'},
             'f': {'dfn': '29.00', 'dfd': '18.00', 'loc': '0.00', 'scale': '1.00'},
             'fatiguelife': {'c': '29.00', 'loc': '0.00', 'scale': '1.00'},
             'fisk': {'c': '3.09', 'loc': '0.00', 'scale': '1.00'},
             'foldcauchy': {'c': '4.72', 'loc': '0.00', 'scale': '1.00'},
             'foldnorm': {'c': '1.95', 'loc': '0.00', 'scale': '1.00'},
             'gamma': {'a': '1.99', 'loc': '0.00', 'scale': '1.00'},
             'gausshyper': {'a': '13.76',
              'b': '3.12',
              'c': '2.51',
              'z': '5.18',
              'loc': '0.00',
              'scale': '1.00'},
             'genexpon': {'a': '9.13',
              'b': '16.23',
              'c': '3.28',
              'loc': '0.00',
              'scale': '1.00'},
             'genextreme': {'c': '-0.10', 'loc': '0.00', 'scale': '1.00'},
             'gengamma': {'a': '4.42', 'c': '3.12', 'loc': '0.00', 'scale': '1.00'},
             'genhalflogistic': {'c': '0.77', 'loc': '0.00', 'scale': '1.00'},
             'geninvgauss': {'p': '2.30', 'b': '1.50', 'loc': '0.00', 'scale': '1.00'},
             'genlogistic': {'c': '0.41', 'loc': '0.00', 'scale': '1.00'},
             'gennorm': {'beta': '1.30', 'loc': '0.00', 'scale': '1.00'},
             'genpareto': {'c': '0.10', 'loc': '0.00', 'scale': '1.00'},
             'gilbrat': {'loc': '0.00', 'scale': '1.00'},
             'gompertz': {'c': '0.95', 'loc': '0.00', 'scale': '1.00'},
             'gumbel_l': {'loc': '0.00', 'scale': '1.00'},
             'gumbel_r': {'loc': '0.00', 'scale': '1.00'},
             'halfcauchy': {'loc': '0.00', 'scale': '1.00'},
             'halfgennorm': {'beta': '0.67', 'loc': '0.00', 'scale': '1.00'},
             'halflogistic': {'loc': '0.00', 'scale': '1.00'},
             'halfnorm': {'loc': '0.00', 'scale': '1.00'},
             'hypsecant': {'loc': '0.00', 'scale': '1.00'},
             'invgamma': {'a': '4.07', 'loc': '0.00', 'scale': '1.00'},
             'invgauss': {'mu': '0.15', 'loc': '0.00', 'scale': '1.00'},
             'invweibull': {'c': '10.58', 'loc': '0.00', 'scale': '1.00'},
             'johnsonsb': {'a': '4.32', 'b': '3.18', 'loc': '0.00', 'scale': '1.00'},
             'johnsonsu': {'a': '2.55', 'b': '2.25', 'loc': '0.00', 'scale': '1.00'},
             'kappa3': {'a': '1.00', 'loc': '0.00', 'scale': '1.00'},
             'kappa4': {'h': '0.10', 'k': '0.00', 'loc': '0.00', 'scale': '1.00'},
             'ksone': {'n': '1000.00', 'loc': '0.00', 'scale': '1.00'},
             'kstwo': {'n': '10.00', 'loc': '0.00', 'scale': '1.00'},
             'kstwobign': {'loc': '0.00', 'scale': '1.00'},
             'laplace': {'loc': '0.00', 'scale': '1.00'},
             'laplace_asymmetric': {'kappa': '2.00', 'loc': '0.00', 'scale': '1.00'},
             'levy': {'loc': '0.00', 'scale': '1.00'},
             'levy_l': {'loc': '0.00', 'scale': '1.00'},
             'levy_stable': {'alpha': '1.80',
              'beta': '-0.50',
              'loc': '0.00',
              'scale': '1.00'},
             'loggamma': {'c': '0.41', 'loc': '0.00', 'scale': '1.00'},
             'logistic': {'loc': '0.00', 'scale': '1.00'},
             'loglaplace': {'c': '3.25', 'loc': '0.00', 'scale': '1.00'},
             'lognorm': {'s': '0.95', 'loc': '0.00', 'scale': '1.00'},
             'loguniform': {'a': '0.01', 'b': '1.00', 'loc': '0.00', 'scale': '1.00'},
             'lomax': {'c': '1.88', 'loc': '0.00', 'scale': '1.00'},
             'maxwell': {'loc': '0.00', 'scale': '1.00'},
             'mielke': {'k': '10.40', 's': '4.60', 'loc': '0.00', 'scale': '1.00'},
             'moyal': {'loc': '0.00', 'scale': '1.00'},
             'nakagami': {'nu': '4.97', 'loc': '0.00', 'scale': '1.00'},
             'ncf': {'dfn': '27.00',
              'dfd': '27.00',
              'nc': '0.42',
              'loc': '0.00',
              'scale': '1.00'},
             'nct': {'df': '14.00', 'nc': '0.24', 'loc': '0.00', 'scale': '1.00'},
             'ncx2': {'df': '21.00', 'nc': '1.06', 'loc': '0.00', 'scale': '1.00'},
             'norm': {'loc': '0.00', 'scale': '1.00'},
             'norminvgauss': {'a': '1.00', 'b': '0.50', 'loc': '0.00', 'scale': '1.00'},
             'pareto': {'b': '2.62', 'loc': '0.00', 'scale': '1.00'},
             'pearson3': {'skew': '0.10', 'loc': '0.00', 'scale': '1.00'},
             'powerlaw': {'a': '1.66', 'loc': '0.00', 'scale': '1.00'},
             'powerlognorm': {'c': '2.14', 's': '0.45', 'loc': '0.00', 'scale': '1.00'},
             'powernorm': {'c': '4.45', 'loc': '0.00', 'scale': '1.00'},
             'rayleigh': {'loc': '0.00', 'scale': '1.00'},
             'rdist': {'c': '1.60', 'loc': '0.00', 'scale': '1.00'},
             'recipinvgauss': {'mu': '0.63', 'loc': '0.00', 'scale': '1.00'},
             'reciprocal': {'a': '0.01', 'b': '1.00', 'loc': '0.00', 'scale': '1.00'},
             'rice': {'b': '0.77', 'loc': '0.00', 'scale': '1.00'},
             'semicircular': {'loc': '0.00', 'scale': '1.00'},
             'skewnorm': {'a': '4.00', 'loc': '0.00', 'scale': '1.00'},
             't': {'df': '2.74', 'loc': '0.00', 'scale': '1.00'},
             'trapezoid': {'c': '0.20', 'd': '0.80', 'loc': '0.00', 'scale': '1.00'},
             'triang': {'c': '0.16', 'loc': '0.00', 'scale': '1.00'},
             'truncexpon': {'b': '4.69', 'loc': '0.00', 'scale': '1.00'},
             'truncnorm': {'a': '0.10', 'b': '2.00', 'loc': '0.00', 'scale': '1.00'},
             'tukeylambda': {'lam': '3.13', 'loc': '0.00', 'scale': '1.00'},
             'uniform': {'loc': '0.00', 'scale': '1.00'},
             'vonmises': {'kappa': '3.99', 'loc': '0.00', 'scale': '1.00'},
             'vonmises_line': {'kappa': '3.99', 'loc': '0.00', 'scale': '1.00'},
             'wald': {'loc': '0.00', 'scale': '1.00'},
             'weibull_max': {'c': '2.87', 'loc': '0.00', 'scale': '1.00'},
             'weibull_min': {'c': '1.79', 'loc': '0.00', 'scale': '1.00'},
             'wrapcauchy': {'c': '0.03', 'loc': '0.00', 'scale': '1.00'}
             }

    return obtained_dict


def stats_options():
    
    stats_all = {'alpha': stats.alpha, 
                 'anglit': stats.anglit, 
                 'arcsine': stats.arcsine, 
                 'argus': stats.argus, 
                 'beta': stats.beta, 
                 'betaprime': stats.betaprime, 
                 'bradford': stats.bradford, 
                 'burr': stats.burr, 
                 'burr12': stats.burr12, 
                 'cauchy': stats.cauchy, 
                 'chi': stats.chi, 
                 'chi2': stats.chi2, 
                 'cosine': stats.cosine, 
                 'crystalball': stats.crystalball, 
                 'dgamma': stats.dgamma, 
                 'dweibull': stats.dweibull, 
                 'erlang': stats.erlang, 
                 'expon': stats.expon, 
                 'exponnorm': stats.exponnorm, 
                 'exponpow': stats.exponpow, 
                 'exponweib': stats.exponweib, 
                 'f': stats.f, 
                 'fatiguelife': stats.fatiguelife, 
                 'fisk': stats.fisk, 
                 'foldcauchy': stats.foldcauchy, 
                 'foldnorm': stats.foldnorm, 
                 'gamma': stats.gamma, 
                 'gausshyper': stats.gausshyper, 
                 'genexpon': stats.genexpon, 
                 'genextreme': stats.genextreme, 
                 'gengamma': stats.gengamma, 
                 'genhalflogistic': stats.genhalflogistic, 
                 'geninvgauss': stats.geninvgauss, 
                 'genlogistic': stats.genlogistic, 
                 'gennorm': stats.gennorm, 
                 'genpareto': stats.genpareto, 
                 'gilbrat': stats.gilbrat, 
                 'gompertz': stats.gompertz, 
                 'gumbel_l': stats.gumbel_l, 
                 'gumbel_r': stats.gumbel_r, 
                 'halfcauchy': stats.halfcauchy, 
                 'halfgennorm': stats.halfgennorm, 
                 'halflogistic': stats.halflogistic, 
                 'halfnorm': stats.halfnorm, 
                 'hypsecant': stats.hypsecant, 
                 'invgamma': stats.invgamma, 
                 'invgauss': stats.invgauss, 
                 'invweibull': stats.invweibull, 
                 'johnsonsb': stats.johnsonsb, 
                 'johnsonsu': stats.johnsonsu, 
                 'kappa3': stats.kappa3, 
                 'kappa4': stats.kappa4, 
                 'ksone': stats.ksone, 
                 'kstwo': stats.kstwo, 
                 'kstwobign': stats.kstwobign, 
                 'laplace': stats.laplace, 
                 'laplace_asymmetric': stats.laplace_asymmetric, 
                 'levy': stats.levy, 
                 'levy_l': stats.levy_l, 
                 'levy_stable': stats.levy_stable, 
                 'loggamma': stats.loggamma, 
                 'logistic': stats.logistic, 
                 'loglaplace': stats.loglaplace, 
                 'lognorm': stats.lognorm, 
                 'loguniform': stats.loguniform, 
                 'lomax': stats.lomax, 
                 'maxwell': stats.maxwell, 
                 'mielke': stats.mielke, 
                 'moyal': stats.moyal, 
                 'nakagami': stats.nakagami, 
                 'ncf': stats.ncf, 
                 'nct': stats.nct, 
                 'ncx2': stats.ncx2, 
                 'norm': stats.norm, 
                 'norminvgauss': stats.norminvgauss, 
                 'pareto': stats.pareto, 
                 'pearson3': stats.pearson3, 
                 'powerlaw': stats.powerlaw, 
                 'powerlognorm': stats.powerlognorm, 
                 'powernorm': stats.powernorm, 
                 'rayleigh': stats.rayleigh, 
                 'rdist': stats.rdist, 
                 'recipinvgauss': stats.recipinvgauss, 
                 'reciprocal': stats.reciprocal, 
                 'rice': stats.rice, 
                 'semicircular': stats.semicircular, 
                 'skewnorm': stats.skewnorm, 
                 't': stats.t, 
                 'trapezoid': stats.trapezoid, 
                 'triang': stats.triang, 
                 'truncexpon': stats.truncexpon, 
                 'truncnorm': stats.truncnorm, 
                 'tukeylambda': stats.tukeylambda, 
                 'uniform': stats.uniform, 
                 'vonmises': stats.vonmises, 
                 'vonmises_line': stats.vonmises_line, 
                 'wald': stats.wald, 
                 'weibull_max': stats.weibull_max, 
                 'weibull_min': stats.weibull_min, 
                 'wrapcauchy': stats.wrapcauchy
                 }
    
    return stats_all


def scipy_distribution_names_and_docstrings():
    """
    Compile a list of distributions with their names and names
    of how to access docstrings.
    
    However, this function returns some distributions multiple times
    because they are repeated with the different parameter setup,
    mostly with negative/positive parameter values.

    Wanting to have only one function, I removed duplicates from 
    the list that I'm using in web-app -- uncommented lines.

    Returns
    -------
    access_docstrings : In the form ['stats.alpha.__doc__', 
                                     ' ... ', ... ]
    distribution_names : In the form ['alpha', 
                                      '....', ...]

    """
    
    access_docstrings = []
    distribution_names = []
    for i, name in enumerate(sorted(stats._distr_params.distcont)):
        docstrings = 'stats.'+str(name[0])+('.__doc__')
        access_docstrings.append(docstrings)
        distribution_names.append(name[0])
    
    return access_docstrings, distribution_names


def get_scipy_distribution_parameters():
    """
    Modified from: https://stackoverflow.com/questions/37559470/what-do-all-the-distributions-available-in-scipy-stats-look-like

    Returns
    -------
    names, all_params_names, all_params

    """
    
    names, all_params_names, all_params = [], [], []

    # Get names and parameters of each scipy distribution 
    # from stats._distr_params.distcont 
    for name, params in sorted(stats._distr_params.distcont):
        
        names.append(name)
        
        # Add loc and scale to the parameters as they are not listed
        loc, scale = 0.00, 1.00
        params = list(params) + [loc, scale]
        all_params.append(params)
        
        # Get distribution information
        dist = getattr(stats, name)
        
        # Join parameters of each distribution with loc and scale
        p_names = ['loc', 'scale']
        if dist.shapes:
            p_names = [sh.strip() for sh in dist.shapes.split(',')] + ['loc', 'scale']
            all_params_names.append(p_names)
        else:
            all_params_names.append(['loc', 'scale'])

    return names, all_params_names, all_params



# This script goes throught the SciPy website and extracts the equations
# for each continuous distribution.
# Note: idx 87 - reciprocal - distribution doesn't have a webpage in 
# SciPy 1.6.1; So the equation is obtained from 1.4 version

# When using these equations make sure to check them, since I compiled a 
# list of them and noticed a number of errors, either due to multiple
# equations on the website, or due to missing equations

# These are crosschecked with the output from the function below this one
# And final equations are majority from the 
# function:extract_equations_from_docstrings()
def scrape():
    """ Get all url's to scipy sites; scrape equations"""
    
    all_urls = []
    for i, name in enumerate(sorted(stats._distr_params.distcont)):
        # Skipping this because its reciprocal - doesn't have a page here
        if i == 87:
            pass
        else:
            add = 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.'+str(name[0])+'.html#scipy.stats.'+str(name[0])
            all_urls.append(add)
    
    
    distribution = []
    for idx, i in enumerate(all_urls):
    
        if idx == 87:
            print(' \[f(x, a, b) = \frac{1}{ \left(x*log( \frac{b}{a} \right) }')
        
        else:
            html_doc = urllib.request.urlopen(i).read()
            soup = BeautifulSoup(html_doc, 'html.parser')
      
    
            divTag = soup.find_all("div", class_="math notranslate nohighlight")
    
    
            for tag in divTag:
                distribution.append(tag.text)

def extract_equations_from_docstrings():
    """
    First part gets all docstrings, and stores them in [all_names].
    docstrings are listed in l which are then accessed (see below)
    and from each, equation is extracted.
    This requires later manual check, due to complexity of the 
    structure around the equations.
    """
    all_names = []
    for i, name in enumerate(sorted(stats._distr_params.distcont)):
        add = 'stats.'+str(name[0])+('.__doc__')
        all_names.append(add)
        
        
    l = [stats.alpha.__doc__,
     stats.anglit.__doc__,
     stats.arcsine.__doc__,
     stats.argus.__doc__,
     stats.beta.__doc__,
     stats.betaprime.__doc__,
     stats.bradford.__doc__,
     stats.burr.__doc__,
     stats.burr12.__doc__,
     stats.cauchy.__doc__,
     stats.chi.__doc__,
     stats.chi2.__doc__,
     stats.cosine.__doc__,
     stats.crystalball.__doc__,
     stats.dgamma.__doc__,
     stats.dweibull.__doc__,
     stats.erlang.__doc__,
     stats.expon.__doc__,
     stats.exponnorm.__doc__,
     stats.exponpow.__doc__,
     stats.exponweib.__doc__,
     stats.f.__doc__,
     stats.fatiguelife.__doc__,
     stats.fisk.__doc__,
     stats.foldcauchy.__doc__,
     stats.foldnorm.__doc__,
     stats.gamma.__doc__,
     stats.gausshyper.__doc__,
     stats.genexpon.__doc__,
     stats.genextreme.__doc__,
     stats.gengamma.__doc__,
     stats.gengamma.__doc__,
     stats.genhalflogistic.__doc__,
     stats.geninvgauss.__doc__,
     stats.genlogistic.__doc__,
     stats.gennorm.__doc__,
     stats.genpareto.__doc__,
     stats.gilbrat.__doc__,
     stats.gompertz.__doc__,
     stats.gumbel_l.__doc__,
     stats.gumbel_r.__doc__,
     stats.halfcauchy.__doc__,
     stats.halfgennorm.__doc__,
     stats.halflogistic.__doc__,
     stats.halfnorm.__doc__,
     stats.hypsecant.__doc__,
     stats.invgamma.__doc__,
     stats.invgauss.__doc__,
     stats.invweibull.__doc__,
     stats.johnsonsb.__doc__,
     stats.johnsonsu.__doc__,
     stats.kappa3.__doc__,
     stats.kappa4.__doc__,
     stats.kappa4.__doc__,
     stats.kappa4.__doc__,
     stats.kappa4.__doc__,
     stats.ksone.__doc__,
     stats.kstwo.__doc__,
     stats.kstwobign.__doc__,
     stats.laplace.__doc__,
     stats.laplace_asymmetric.__doc__,
     stats.levy.__doc__,
     stats.levy_l.__doc__,
     stats.levy_stable.__doc__,
     stats.loggamma.__doc__,
     stats.logistic.__doc__,
     stats.loglaplace.__doc__,
     stats.lognorm.__doc__,
     stats.loguniform.__doc__,
     stats.lomax.__doc__,
     stats.maxwell.__doc__,
     stats.mielke.__doc__,
     stats.moyal.__doc__,
     stats.nakagami.__doc__,
     stats.ncf.__doc__,
     stats.nct.__doc__,
     stats.ncx2.__doc__,
     stats.norm.__doc__,
     stats.norminvgauss.__doc__,
     stats.pareto.__doc__,
     stats.pearson3.__doc__,
     stats.powerlaw.__doc__,
     stats.powerlognorm.__doc__,
     stats.powernorm.__doc__,
     stats.rayleigh.__doc__,
     stats.rdist.__doc__,
     stats.recipinvgauss.__doc__,
     stats.reciprocal.__doc__,
     stats.rice.__doc__,
     stats.semicircular.__doc__,
     stats.skewnorm.__doc__,
     stats.t.__doc__,
     stats.trapezoid.__doc__,
     stats.triang.__doc__,
     stats.truncexpon.__doc__,
     stats.truncnorm.__doc__,
     stats.truncnorm.__doc__,
     stats.tukeylambda.__doc__,
     stats.uniform.__doc__,
     stats.vonmises.__doc__,
     stats.vonmises_line.__doc__,
     stats.wald.__doc__,
     stats.weibull_max.__doc__,
     stats.weibull_min.__doc__,
     stats.wrapcauchy.__doc__]
    
    
    # Get equations, tracing places after math
    # The end of equation is not marked in any special way, so
    # get more lines just in sace some equations are longer 
    eq = []
    reg = []
    for j in range(len(sorted(l))):
        for i, line in enumerate(l[j].split('\n')):
            if 'math::' in line:
                eq.append(l[j].split('\n')[i+2:i+4])
                reg.append(l[j].split('\n')[i+2:i+6])
     
    # Join each extraction together, this will remove the extra
    # blank lines from the equation
    class OurList(list): 
        def join(self, s):
            return s.join(self)
    
    joined_eq = []
    for i in eq:
        li = OurList(i)
        li = li.join('')
        joined_eq.append(li)
    
    #print(joined_eq)
    return

# Creating dictionaries
def creating_dictionaries():
    """
    Final results from the helper functions are various dictionaries that
    I use in the code.
    For every dictionary, there is an example how it looks like.
    """
    
    # Dictionary containing distribution name and how to access docstrings    
    access_docstrings, distribution_names = scipy_distribution_names_and_docstrings()
    name_docstring_dict = {distribution: access_docstrings[i] for i, distribution in enumerate(distribution_names)}
    
    """
    {'alpha': 'stats.alpha.__doc__',
     'anglit': 'stats.anglit.__doc__',
     ...
    """
    
    # Dictionary containing distribution name and its PDF    
    eq = scipy_distribution_equations()
    name_eq_dict = {distribution: eq[i] for i, distribution in enumerate(distr_selectbox_names())}
    
    """
    {'alpha': 'f(x, a) = \\frac{1}{x^2 \\Phi(a) \\sqrt{2\\pi}} * \\exp(-\\frac{1}{2} (a-1/x)^2)',
     'anglit': 'f(x) = \\sin(2x + \\pi/2) = \\cos(2x)',
     ...
    """
    
    # Dictionary containing distribution name and its proper name    
    proper_names = scipy_distribution_proper_names()
    name_proper_dict = {distribution: proper_names[i] for i, distribution in enumerate(distr_selectbox_names())}
    
    """
    {'alpha': 'Alpha distribution',
     'anglit': 'Anglit distribution',
     ...
    """
    
    # Nested dictionary containing distribution name and its paramaters&values  
    names, all_params_names, all_params = get_scipy_distribution_parameters()
    all_dist_params_dict = {function: {param:  f"{all_params[i][j]:.2f}"  for j, param in enumerate(all_params_names[i])} for i, function in enumerate(names)}

    """
        {'alpha': {'a': 3.57,
                   'loc': 0.0,
                   'scale': 1.0},
         'anglit': {'loc': 0.0,
                    'scale': 1.0},
         etc...
    """

    # Names and url-s
    name_url_dict = {distribution: [extracted_scipy_urls()[i], scipy_distribution_proper_names()[i]] for i, distribution in enumerate(distr_selectbox_names())}

    """
    {'alpha': ['https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.alpha.html#scipy.stats.alpha',
               'Alpha distribution'],
     'anglit': ['https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anglit.html#scipy.stats.anglit',
                'Anglit distribution'],
     ...
    """


    return name_docstring_dict, name_eq_dict, name_proper_dict, all_dist_params_dict, name_url_dict
