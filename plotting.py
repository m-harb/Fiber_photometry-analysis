from typing import Optional
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from pathlib import Path

from shapely.plotting import plot_polygon

import utilities as u
from settings import colors
import seaborn as sns
DELTA_F_F = r'$\frac{\Delta\ F}{F} $'


def plot_raw_data(recording):
    """
    Plot the raw signals from a given recording.

    Parameters
    ----------
    recording: dict
        Dictionary regrouping all info on a recording.
        As returned by open_data.get_data

    Returns
    -------
    fig: matplotlib.Figure
    """
    data = recording['signals']
    t = recording['time']
    n_sites = len(data)
    fig, axs = plt.subplots(n_sites, 1, sharex='all', squeeze=False)
    site_ix = 0
    for site_name, signals in data.items():
        c_ax = axs[site_ix, 0]
        c_ax.set_title(f'Signals from {site_name}')
        for wvl, signal in signals.items():
            c_ax.plot(t, signal, label=wvl)
        c_ax.legend()
        c_ax.set_xlabel('Time (s)')
        c_ax.set_ylabel('Raw data (AU)')
        add_events(c_ax, recording)
        site_ix += 1
    md_title(fig, recording)
    fig.set_tight_layout(True)
    return fig


def add_events(ax, recording):
    """
    Plot events as vertical bars on a given set of axes.
    The event times are taken from a recording dictionary that itself is usually returned by open_data.get_data


    Parameters
    ----------
    ax: matplotlib.Axes
    recording: dict
        As returned by open_data.get_data. Needs at least a key 'events' which value
        is a numpy array of timestamps

    """
    for ev in recording['events']:
        ax.axvline(ev, color='k', zorder=-1)


def md_title(fig, recording):
    """
    Set the title of a given figure according to the metdata of a recording

    Parameters
    ----------
    fig
    recording dict
        As returned by open_data.get_data. Needs at least a key 'metadata' which value is a dictionary
        This metadata dict needs keys: mouse, genotype, injection and task


    Returns
    -------

    """
    md = recording['metadata']
    fig.suptitle(f'Animal {md["mouse"]} - {md["genotype"]} - {md["injection"]} - {md["task"]}',
                 fontsize=18)


def plot_dff(recording, save_dir: Optional[str] = None):
    """
    Plot the Delta F / F from a given recording.
    This recording dictionary is returned by pm_analysis.compute_dff_recording

    Parameters
    ----------
    recording: dict
        As returned by pm_analysis.compute_dff_recording

    save_dir: str, Optional
        Path to the directory in which we want to save the figure
        A subdirectory will be created to store only dff figures.



    """
    dffs = recording['dff']
    t = recording['dff_time']
    n_sites = len(dffs)
    if n_sites == 0:
        return
    fig, axs = plt.subplots(n_sites, 1, sharex='all', figsize=(14.22, 5.3), squeeze=False)
    site_ix = 0
    for site_name, signal in dffs.items():
        c_ax = axs[site_ix, 0]
        c_ax.set_title(f'{site_name}', fontsize=16)
        c_ax.plot(t, signal, c='green')
        site_ix += 1
        c_ax.set_xlabel('Time (s)')
        c_ax.set_ylabel(DELTA_F_F)
        add_events(c_ax, recording)
    md_title(fig, recording)
    fig.set_tight_layout(True)
    save_figure(fig, recording['metadata'], 'dff', prefix='dff', save_dir=save_dir)
    plt.close(fig)



def save_figure(fig, metadata, kind='dff', prefix=None, save_dir=None):
    """
    Saves a figure in a given directory. Creates a subdirectory according to the kind of figure
    to be saved (dff, peth...). The name of the file depends on the metadata from the recording


    Parameters
    ----------
    fig: matplotlib.Figure
        Figure to be saved
    metadata: dict
        Metadata dictionary from the recording dictionary which is itself returned by open_data.get_data
    kind: str
        Kind of data we plotted (dff, PETH...)
    prefix: str, optional
        Prefix to add at the beginning of the file name. If left to None (default) uses kind
    save_dir: str, optional
        Directory to save the figure in. If left to None, (the default) nothing is saved


    """
    if save_dir is None:
        return
    prefix = prefix if prefix is not None else kind
    task = metadata['task']
    mid = metadata['mid']
    save_dir = Path(save_dir)
    save_dir_task = save_dir / f'{kind}/{task}'
    save_dir_task.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_dir_task / f'{prefix}_{mid}.png')


def plot_peth_recording(recording, save_dir: Optional[str] = None):
    """
    Plot the PETH from a given recording.
    This recording dictionary is returned by pm_analysis.compute_dff_recording

    Parameters
    ----------
    recording: dict
        As returned by pm_analysis.compute_dff_recording

    save_dir: str, Optional
        Path to the directory in which we want to save the figure
        A subdirectory will be created to store only dff figures.

    Returns
    -------
    fig: matplotlib.Figure

    """
    z_snips = recording['z_snips']
    t_snips = recording['t_snips']
    if t_snips is None:
        return
    n_sites = len(z_snips)
    fig, axs = plt.subplots(n_sites, 1, sharex='all', figsize=(9.5, 9.5), squeeze=False)
    site_ix = 0
    for site_name, c_snips in z_snips.items():
        if len(c_snips) < 1:
            continue
        c_ax = axs[site_ix, 0]
        plot_peth(t_snips, c_snips, site_name, c_ax)
        site_ix += 1
    md_title(fig, recording)
    fig.set_tight_layout(True)
    save_figure(fig, recording['metadata'], 'PETH', save_dir)
    return fig


def plot_peth(t_snips, snips, site_name='', ax=None):
    """
    Plot a PETH. Basic function that is independant of the recording data structure

    Parameters
    ----------
    t_snips: numpy array
    snips: numpy array
    site_name: str
    ax: matplotlib.Axes, optional

    Returns
    -------
    ax: matplotlib.Axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    hm_ax = divider.append_axes('top', size='30%', pad='2%')
    hm_ax.set_title(f'{site_name}', fontsize=16)
    # ax.plot(t_snips, snips.T, c='k', lw=.5, alpha=.5)
    hm_ax.imshow(snips, extent=(t_snips[0], t_snips[-1], 0, snips.shape[0]), interpolation='none', aspect='auto')
    u.full_clean_ax(hm_ax)
    avg, sem = u.compute_mean_sem(snips)
    ax.axvline(0, color='.3')
    ax.fill_between(t_snips, avg - sem, avg + sem, facecolor='green', edgecolor='none', alpha=.3)
    ax.plot(t_snips, avg, lw=2, c='green')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel("Z-scored " + DELTA_F_F)
    return ax


def plot_maps_recording(recording, fig_path=None):
    map_keys = ('s_map', 's_occ_map')
    if any([k not in recording['video'] for k in map_keys]):
        return

    fig, axs = plt.subplots(1, 2, sharex='all', sharey='all', figsize=(15, 12))
    axs = axs.reshape(-1)
    for ix, mk in enumerate(map_keys):
        c_ax = axs[ix]
        divider = make_axes_locatable(c_ax)
        o_ax = divider.append_axes('bottom', size='100%', pad='25%')
        site_ax = (c_ax, o_ax)
        site_ix = 0
        for site, c_map in recording['video'][mk].items():
            s_ax = site_ax[site_ix]
            s_ax.set_title(f'{site}', fontsize=18)
            s_ax.imshow(c_map, origin='lower')
            site_ix += 1
            if 'occ' in mk and site_ix > 0:
                s_ax.set_title('Occupancy')
                break
    md_title(fig, recording)
    save_figure(fig, recording['metadata'], kind='maps', prefix='act_occ', save_dir=fig_path)


def plot_roi_pos(centroids, fig_path, rois, metadata):
    fig, ax = plt.subplots()
    for name, roi in rois.items():
        plot_polygon(roi, ax=ax, color=colors[name])
    ax.plot(*centroids.T, '.', ms=1, c='k')
    ax.set_ylim(ax.get_ylim()[::-1])
    md_title(fig, {'metadata': metadata})
    save_figure(fig, metadata, 'maps', prefix='rois_map', save_dir=fig_path)


def group_boxplot(full_df, group_by=('site', 'genotype', 'injection'), key='time_center', norm_key=None):
    loc_full_df = full_df.copy()
    orders = {'site': ('BNST', 'CeA'), 'genotype': ('WT', 'Glong'), 'injection': ('CTRL', 'CNO')}
    if norm_key is not None:
        loc_full_df[f'norm_{key}'] = loc_full_df[key] / loc_full_df[norm_key]
        key = f'norm_{key}'
    group_by = list(group_by)
    # gp = full_df.groupby(group_by)
    sns.catplot(loc_full_df, x=group_by[-1], y=key, hue=group_by[1], row=group_by[0], kind='box',
                order=orders[group_by[-1]], hue_order=orders[group_by[1]], row_order=orders[group_by[0]],
                width=.5)


if __name__ == '__main__':
    plt.ion()
