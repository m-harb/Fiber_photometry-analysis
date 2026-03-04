import numpy as np
from scipy.interpolate import interp1d


def full_clean_ax(ax):
    # Hide the spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.yaxis.set_ticklabels([])
    ax.xaxis.set_ticklabels([])


def clean_ax(ax):
    """to improve the display of the axes in our figures"""
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


def compute_mean_sem(snips):
    avg = snips.mean(0)
    sem = snips.std(0) / np.sqrt(snips.shape[0] - 1)
    return avg, sem


def time_interpolation(ref_time, orig_time, signal):
    ip = interp1d(orig_time, signal, bounds_error=False, fill_value='extrapolate')
    ip_sig = ip(ref_time)
    return ip_sig
