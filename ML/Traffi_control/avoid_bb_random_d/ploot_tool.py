import matplotlib
import matplotlib.ticker as mt

inline_rc = matplotlib.rcParams
import matplotlib.colors as colors

colors.colorConverter.cache = {}
import math

from IPython.core.display import clear_output
import pickle
# from igraph import *
# import igraph.test
# print(igraph.__version__)
# igraph.test.run_tests()
# import cairo
import matplotlib.pyplot as plt
from pprint import pprint
# import psycopg2
# import numba

import numexpr
from collections import defaultdict
from sklearn.cluster import *
# from sklearn import metrics
from sklearn.decomposition import *

import pandas as pd
import numpy as np
import scipy
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import scipy.stats as ss
import scipy.io as sio
import statsmodels.api as sm
import sklearn as sl

pd.options.mode.chained_assignment = None  # default='warn'
matplotlib.rcParams.update(inline_rc)
from matplotlib.font_manager import FontProperties

import warnings

warnings.filterwarnings('ignore')

# print(matplotlib.rcParams)
import matplotlib.font_manager

[f for f in matplotlib.font_manager.fontManager.ttflist]
# print(matplotlib.rcParamsDefault)

# create colormaps
import seaborn as sns

sns.reset_orig()

sns.palplot(sns.color_palette('deep', 10))
sns.palplot(sns.color_palette("Set2", 10))
sns.palplot(sns.color_palette("Paired", 12))
[plblue, pblue, plgreen, pgreen, plred, pred, plorange, porange, plpurple, ppurple, plbrown,
 pbrown] = sns.color_palette("Paired", 12)

import palettable
from cycler import cycler

pal = palettable.colorbrewer.qualitative.Set1_9
colors = pal.mpl_colors
sns.palplot(colors)
[cred, cblue, cgreen, cpurple, corange, cyellow, cbrown, cpink, cgray] = colors
matplotlib.rc("axes", prop_cycle=cycler(color=list(colors)))

import matplotlib.colors as colors

colors.colorConverter.colors['cblue'] = cblue
colors.colorConverter.colors['cred'] = cred
colors.colorConverter.colors['cgreen'] = cgreen
colors.colorConverter.colors['corange'] = corange
colors.colorConverter.colors['cpink'] = cpink
colors.colorConverter.colors['cbrown'] = cbrown
colors.colorConverter.colors['cgray'] = cgray
colors.colorConverter.colors['cpurple'] = cpurple
colors.colorConverter.colors['cyellow'] = cyellow

colors.colorConverter.colors['plblue'] = plblue
colors.colorConverter.colors['pblue'] = pblue
colors.colorConverter.colors['plgreen'] = plgreen
colors.colorConverter.colors['pgreen'] = pgreen
colors.colorConverter.colors['plred'] = plred
colors.colorConverter.colors['pred'] = pred
colors.colorConverter.colors['plorange'] = plorange
colors.colorConverter.colors['porange'] = porange
colors.colorConverter.colors['plpurple'] = plpurple
colors.colorConverter.colors['ppurple'] = ppurple
colors.colorConverter.colors['plbrown'] = plbrown
colors.colorConverter.colors['pbrown'] = pbrown

colors.colorConverter.cache = {}

# import ipyparallel as ipp

import colormaps as cmaps

cmap_viridis = cmaps.viridis
cmap_magma = cmaps.magma
cmap_inferno = cmaps.inferno
cmap_plasma = cmaps.plasma
cmap_spectral = palettable.colorbrewer.diverging.Spectral_11_r.mpl_colormap
cmap_viridis_r = matplotlib.colors.ListedColormap(cmaps.viridis.colors[::-1])


def universal_fig(figsize=(3, 3), fontsize=12, axislinewidth=1, markersize=5, text=None, limits=[-7, 7],
                  offset=[-44, 12], projection=None, fontfamily=["Helvetica", "Arial"], contain_latex=False):
    '''
    Create universal figure settings with publication quality
    returen fig, ax (similar to plt.plot)
    fig, ax = universal_fig()
    '''
    # ----------------------------------------------------------------
    if projection is None:
        fig, ax = plt.subplots(frameon=False)
    else:
        fig, ax = plt.subplots(frameon=False, subplot_kw=dict(projection=projection))
    fig.set_size_inches(figsize)
    matplotlib.rc("font", **{"family": "sans-serif", "sans-serif": fontfamily, "size": fontsize})
    matplotlib.rc('pdf', fonttype=42, use14corefonts=True, compression=6)
    matplotlib.rc('ps', useafm=True, usedistiller='none', fonttype=42)
    matplotlib.rc("axes", unicode_minus=False, linewidth=axislinewidth, labelsize='medium')
    matplotlib.rc("axes.formatter", limits=limits)
    matplotlib.rc('savefig', bbox='tight', format='eps', frameon=False, pad_inches=0.05)
    matplotlib.rc('legend')
    matplotlib.rc('lines', marker=None, markersize=markersize)
    matplotlib.rc('text', usetex=False)
    matplotlib.rc('xtick', direction='in')
    matplotlib.rc('xtick.major', size=4)
    matplotlib.rc('xtick.minor', size=2)
    matplotlib.rc('ytick', direction='in')
    matplotlib.rc('lines', linewidth=1)
    matplotlib.rc('ytick.major', size=4)
    matplotlib.rc('ytick.minor', size=2)
    matplotlib.rcParams['lines.solid_capstyle'] = 'butt'
    matplotlib.rcParams['lines.solid_joinstyle'] = 'bevel'
    matplotlib.rc('mathtext', fontset='stixsans')

    if contain_latex:
        matplotlib.rc('ps', useafm=False, usedistiller='none', fonttype=3)
        matplotlib.rc('pdf', fonttype=3, use14corefonts=True, compression=6)

    matplotlib.rc('legend', fontsize='medium', frameon=False,
                  handleheight=0.5, handlelength=1, handletextpad=0.4, numpoints=1)
    if text is not None:
        w = ax.annotate(text, xy=(0, 1), xycoords='axes fraction', fontsize='large', weight='bold',
                        xytext=(offset[0] / 12 * fontsize, offset[1] / 12 * fontsize), textcoords='offset points',
                        ha='left', va='top')
        print(w.get_fontname())
    # ----------------------------------------------------------------
    # end universal settings
    return fig, ax