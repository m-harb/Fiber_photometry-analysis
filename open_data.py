import pandas as pd
import numpy as np
from pathlib import Path

from pims import Video

from utilities import time_interpolation


def open_photometry(filepath, sites_names=None):
    """
    Open photometry data file that contains the signal for each region (ie ROI)

    Parameters
    ----------
    filepath: str or Path
        Path to the data_photometry*.csv file. It needs columns:
            LedState, RegionXXXG, FrameCounter
    sites_names: dict, Optional
        Associates a region column name (Region0G, Region1G...) and the brain region name
        If a brain region name is missing, the column name is kept.
        Same if sites_names is left as None, the default value.

    Returns
    -------
    data: dict
        {brain_region_name: {wavelength: signal as numpy array} }
    frames: dict
        {wavelength: frame index}
    """
    # Get the column names from the header
    with open(filepath, 'r') as fh:
        s_header = fh.readline().strip()
    header = s_header.split(',')
    # Open the raw numerical data
    raw_data = pd.read_csv(filepath, delimiter=',').to_numpy()
    # Get the signals per site
    ph_data = {col_name: raw_data[:, col_ix]
               for col_ix, col_name in enumerate(header)
               if 'Region' in col_name}
    # Rename the sites if site names were given
    if sites_names is not None:
        ph_data = {sites_names.get(col_name, col_name): signal for col_name, signal in ph_data.items()}
    # Get the led state column, to split the wavelengths
    led_ix = header.index('LedState')
    led = raw_data[:, led_ix]
    # Frame counter column
    fc_ix = header.index('FrameCounter')
    framecounter = raw_data[:, fc_ix]
    # Split data per wavelength
    wvl_codes = {470: 2, 415: 1}
    data = {site: {wvl: signal[led == w_code]
                   for wvl, w_code in wvl_codes.items()}
            for site, signal in ph_data.items()}
    frames = {wvl: framecounter[led == w_code].astype(int) for wvl, w_code in wvl_codes.items()}
    # Trim the end in case we don't have as many samples per wavelength
    min_length = min([len(cf) for cf in frames.values()])
    data = {site: {wvl: c_signal[:min_length]
                   for wvl, c_signal in signal.items()}
            for site, signal in data.items()}
    frames = {wvl: f_ix[:min_length] for wvl, f_ix in frames.items()}

    return data, frames


def open_photometry_ts(filepath, frames):
    raw_ts = pd.read_csv(filepath, delimiter=',', header=None).to_numpy()
    ts = {wvl: raw_ts[np.in1d(raw_ts[:, 0], f_ix), 1] for wvl, f_ix in frames.items()}
    return ts


def open_keydown_ts(filepath):
    try:
        raw_ts = pd.read_csv(filepath, delimiter=',', header=None).to_numpy()
    except pd.errors.EmptyDataError:
        return np.array([])

    ts = raw_ts[:, 1]
    # At least 10 seconds between two events
    d_ts = np.diff(ts, prepend=0)
    ts = ts[d_ts > 10*1000]

    return ts


def open_video_ts(filepath):
    raw_ts = pd.read_csv(filepath, delimiter=',', header=None).to_numpy()
    return raw_ts[:, 1]


def get_metadata(filepath):
    filepath = Path(filepath)
    parts = filepath.parts
    fields = ("mouse", "genotype", "injection", "task")
    d = {}
    for i, new_key in enumerate(fields):
        d[new_key] = parts[-2 - i]
    d['mid'] = f'{d["mouse"]}_{d["genotype"]}_{d["injection"]}'
    d['folderpath'] = filepath.parent
    file_prefix = ('data_behvideo', 'ts_behvideo', 'data_photometry', 'ts_photometry', 'ts_keydown')
    filepaths = {}
    for prefix in file_prefix:
        tmp = d['folderpath'].glob(prefix + '*')
        for f in tmp:
            filepaths[prefix] = f
    d.update(filepaths)
    return d


def find_quality_check(photometry_path):
    photometry_path = Path(photometry_path)
    if photometry_path.is_file():
        photometry_path = photometry_path.parent
    quality_paths = list(photometry_path.glob('quality_check.csv'))
    if len(quality_paths) == 1:
        return quality_paths[0]
    if photometry_path.parent != photometry_path:
        return find_quality_check(photometry_path.parent)
    return None


def assess_quality(metadata, batch_num=(2, ), quality_th=2, quality_path=None):
    pm_path = metadata['folderpath']
    if quality_path is None:
        quality_path = find_quality_check(pm_path)
    if quality_path is None:
        return {}, quality_path
    quality = pd.read_csv(quality_path, dtype={'mouse': str})
    mouse = metadata["mouse"]
    genotype = metadata["genotype"]
    inj = metadata["injection"]
    task = metadata["task"]
    quality['valid'] = (quality['quality'] >= quality_th) & (quality['batch'].isin(batch_num))
    rec_quality = quality.query(f'mouse=="{mouse}" and genotype=="{genotype}" and injection=="{inj}" and task=="{task}"')
    q = {row['site']: row['valid'] for ix, row in rec_quality.iterrows()}

    return q, quality_path


def filter_quality(data_dict, mask_dict):
    """
    Filters items from data_dict based on the values of mask_dict.
    data_dict and mask_dict keys should be shared. If a key is missing in mask_dict
    then the associated value is supposed to be True, so the corresponding value in data_dict is kept.

    Parameters
    ----------
    data_dict: dict
        {k: value}
    mask_dict: dict
        {k: boolean}

    Returns
    -------
    qd: dict
        Dictionary with the same structure as data_dict but some items could be gone
    """
    qd = {k: v for k, v in data_dict.items() if mask_dict.get(k, True)}
    return qd


def get_data(dir_path, sites_names=None, ref_time=470, batch_num=(1, 2), quality_th=2, quality_path=None):
    dir_path = Path(dir_path)
    pm_path = list(dir_path.glob('data_photometry*.csv'))[0]
    md = get_metadata(pm_path)
    quality, quality_path = assess_quality(md, batch_num, quality_th, quality_path)
    md['quality_path'] = quality_path
    #md['quality'] = quality
    ts_pm_path = md['ts_photometry']
    ts_video_path = md['ts_behvideo']
    ts_keydown_path = md['ts_keydown']
    raw_data, frames = open_photometry(pm_path, sites_names)
    raw_data = filter_quality(raw_data, quality)
    frames = filter_quality(frames, quality)
    ts_pm = open_photometry_ts(ts_pm_path, frames)
    # Time interpolation to compensate for the delay between 415 and 470 acquisition
    ts = ts_pm[ref_time]
    data = {}
    for site_name, site_sig in raw_data.items():
        data[site_name] = {}
        for wvl, signal in site_sig.items():
            if wvl != ref_time:
                signal = time_interpolation(ts, ts_pm[wvl], signal)
            data[site_name][wvl] = signal

    ts_keydown = open_keydown_ts(ts_keydown_path)
    ts_video = open_video_ts(ts_video_path)
    v = Video(md['data_behvideo'])
    n_frames = len(v)
    v.close()
    assert len(ts_video) == n_frames
    time = (ts - ts[0]) / 1000
    ts_keydown = (ts_keydown - ts[0]) / 1000
    ts_video = (ts_video - ts[0]) / 1000
    recording = {'signals': data, 'time': time, 'events': ts_keydown,
                 'video': {'time': ts_video},
                 'metadata': md, 'quality': quality}
    return recording


if __name__ == '__main__':
    bp = Path('C:/Users/malek/Desktop/Python codes/Aquineuro/2024/')
    fp = bp #/ 'revision batch 1/data/OFT/CNO/WT/4'
    fig_path = Path('C:/Users/malek/Desktop/Python codes/Aquineuro/2024/revision batch 2/figures')
    sites = {'Region0G': 'BNST', 'Region1G': 'CeA'}
    rec = get_data(fp, sites_names=sites)
