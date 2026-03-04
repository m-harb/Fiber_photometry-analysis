from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from scipy.integrate import simps

import plotting
from open_data import get_data
from pm_space import process_map
from settings import upaths
from videotracking import track_positions
import roi_analysis as ra


def trim(t, signal, duration_s=15):
    """
    Trim the beginning of a signal, to take care of the photobleaching

    Parameters
    ----------
    t: numpy.ndarray
        Time vector
    signal: numpy.ndarray
        Signal (iso or physio)
    duration_s: float
        Duration to trim, in seconds
        Default to 15 seconds

    Returns
    -------
    masked_time: : numpy.ndarray
        Time, after trimming (no longer starts at 0)
    masked_signal: : numpy.ndarray
        Signal after trimming
    """
    mask = t > t[0] + duration_s
    return t[mask], signal[mask]


def fit_iso(iso, physio):
    """
    Fit the isosbestic signal to the physio signal, for DF/F computation

    Parameters
    ----------
    iso: numpy.ndarray
        Iso signal after trimming
    physio: numpy.ndarray
        Physio signal

    Returns
    --------
    iso_fitted: numpy.ndarray
        Iso after fit is applied
    """
    p = np.polyfit(iso, physio, deg=1)
    iso_fitted = np.polyval(p, iso)
    return iso_fitted


def compute_dff(iso_fitted, physio):
    """
    Compute DF/F by normalizing the physio signal by the iso signal, usually the fitted version (Lerner et al 2015)

    Parameters
    ----------
    iso_fitted: numpy.ndarray
    physio: numpy.ndarray

    Returns
    -------
    delta: numpy.ndarray
        DF/F
    """
    delta = (physio - iso_fitted) / iso_fitted
    return delta


def analyse_signal(t, iso, physio, trim_duration_s=15):
    """
    Analyze the photometry signal: trim, fit and compute DF/F

    Parameters
    ----------
    t: np.ndarray
        Time vector
    iso: np.ndarray
        Isosbestic signal
    physio: np.ndarray
        Physiology signal (GCaMP, GRAB-DA...)
    trim_duration_s: float
        Duration to cut from the start of the signal before fitting.

    Returns
    -------
    t_trim: np.ndarray
        Time vector with initial duration cut, to be the same size as the DF/F
    delta_f: np.ndarray
        DF/F signal.
    """
    t_trim, iso_trim = trim(t, iso, trim_duration_s)
    _, physio_trim = trim(t, physio, trim_duration_s)
    iso_fit = fit_iso(iso_trim, physio_trim)
    delta_f = compute_dff(iso_fit, physio_trim)
    return t_trim, delta_f


def compute_dff_recording(recording, trim_duration_s=15, show=False, fig_path=None):
    """
    Compute all the DF/F for a given recording, which can have multiple recording sites

    Parameters
    ----------
    recording: dict
    trim_duration_s: float
        Duration to cut from the start of the signal before fitting.
    show: bool
        Shall we plot the DF/F for all sites?
        Default to False
    fig_path: Path, optional
        Where to save the figure
        Default to None
    Returns
    -------

    """
    signals = recording['signals']
    recording['dff'] = {}
    t_trim = None
    for site_name, site_signals in signals.items():
        t_trim, dff = analyse_signal(recording['time'], site_signals[415], site_signals[470], trim_duration_s)
        recording['dff'][site_name] = dff
    recording['dff_time'] = t_trim
    if show:
        plotting.plot_dff(recording, fig_path)
    return recording


def get_snippets(t, dff, events, t_left_s=10., t_right_s=20.):
    """
    Extract pieces of DF/F signal around events.
    
    Parameters
    ----------
    t: np.ndarray
        Time vector
    dff: np.ndarray
        DF/F signal
    events: np.ndarray
        Time stamps of events
    t_left_s: float
        Time in seconds to get before an event for the snippet extraction / PETH
        Default to 10
    t_right_s: float
        Time in seconds to get after an event for the snippet extraction / PETH
        Default to 20

    Returns
    -------
    t_snips: : np.ndarray
        Time vector with event time at 0
    snipps_arr: np.ndarray
        2D array. One event per row, one column per time point

    """
    dt = np.median(np.diff(t))
    pts_left = int(t_left_s / dt)
    pts_right = int(t_right_s / dt)
    events = [ev for ev in events if ev + t_right_s < t[-1] and ev - t_left_s > t[0]]
    ev_ix = np.searchsorted(t, events)
    left_ix = ev_ix - pts_left
    right_ix = ev_ix + pts_right
    snips = [dff[l_ix:r_ix] for l_ix, r_ix in zip(left_ix, right_ix)]
    if len(snips) > 0:
        snips_arr = np.vstack(snips)
        t_snips = (np.arange(snips_arr.shape[1]) * dt) - t_left_s
    else:
        snips_arr = np.array([])
        t_snips = np.array([])

    return t_snips, snips_arr


def z_score(t, signals):
    """
    Z-scoring a signal using negative times as the baseline

    Parameters
    ----------
    t: np.ndarray
        Time vector
    signals:
        Signal to normalize

    Returns
    -------
    z_sig: np.ndarray
        z-scored signal
    """
    is_bsl = t < 0
    if np.sum(is_bsl) == 0:
        return signals
    bsl = signals[:, is_bsl]
    avg = bsl.mean(1)
    std = bsl.std(1)
    z_sig = ((signals.T - avg) / std).T
    return z_sig


def extract_snippets_recording(recording, t_left_s=10., t_right_s=20.):
    """
    Extract pieces of DF/F signal around events in a recording, that can have multiple recording sites

    Parameters
    ----------
    recording: dict
    t_left_s: float
        Time in seconds to get before an event for the snippet extraction / PETH
        Default to 10
    t_right_s: float
        Time in seconds to get after an event for the snippet extraction / PETH
        Default to 20

    Returns
    -------
    recording: dict
        Updated recording dictionary with snippets included
    """
    t = recording['dff_time']
    events = recording['events']
    recording['snips'] = {}
    recording['z_snips'] = {}
    recording['t_snips'] = None
    if len(events) == 0:
        return recording
    for site_name,  dff in recording['dff'].items():
        t_snips, snips = get_snippets(t, dff, events, t_left_s, t_right_s)
        z_snips = z_score(t_snips, snips)
        recording['snips'][site_name] = snips
        recording['z_snips'][site_name] = z_snips
        recording['t_snips'] = t_snips
    return recording


def peth_amplitude(t_snips, snips, z_th=3, delay_s=1):
    """
    Quantify the amplitude of the response in the PETH by peak detection

    Parameters
    ----------
    t_snips
    snips
    z_th
    delay_s

    Returns
    -------
    peaks: np.ndarray
        Indices of peaks
    all_heights: np.ndarray
        Heights of all peaks
    """
    dt = np.median(np.diff(t_snips))
    peaks = []
    all_heights = []
    start_ev_ix = np.nonzero(t_snips >= 0)[0][0]
    for c_snip in snips:
        pk, infos = find_peaks(c_snip, height=z_th, distance=int(delay_s / dt))
        pk_ix = [ix for ix, c_pk in enumerate(pk) if c_pk > start_ev_ix]
        if len(pk_ix) == 0:
            continue
        # We only keep the index of the biggest peak (larger peak_height)
        c_pk_ix = pk_ix[np.argmax(infos['peak_heights'][pk_ix])]
        peaks.append(pk[c_pk_ix])
        all_heights.append(infos['peak_heights'][c_pk_ix])

    return np.array(peaks), np.array(all_heights)


def peth_auc(t_snips, snips):
    """
    Compute the AUC of snippets for any time after 0 (event time)

    Parameters
    ----------
    t_snips
    snips

    Returns
    -------

    """
    is_resp = t_snips > 0
    resp = snips[:, is_resp]
    t_resp = t_snips[is_resp]
    auc = simps(resp, t_resp, axis=1)
    return auc


def peth_amp_recording(recording, z_th=3, delay_s=1):
    """
    Quantifying all PETH of a recording

    Parameters
    ----------
    recording
    z_th
    delay_s

    Returns
    -------

    """
    t = recording['t_snips']
    recording['peaks'] = {}
    recording['amplitudes'] = {}
    recording['resp_amp'] = {}
    recording['resp_auc'] = {}
    for site_name, snips in recording['z_snips'].items():
        peaks, amp = peth_amplitude(t, snips, z_th, delay_s)
        recording['peaks'][site_name] = peaks
        recording['amplitudes'][site_name] = amp
        recording['resp_amp'][site_name] = np.mean(amp)
        recording['resp_auc'][site_name] = np.mean(peth_auc(t, snips))
    return recording


def get_recording(folder_path, sites_names, analysis_path, trim_duration_s=15, t_left_s=10, t_right_s=20, z_th=3,
                  delay_s=1, batch_num=(1, 2), quality_th=2, bin_spacing=5., quality_path=None, force=True,
                  output_video=False, bilateral_filter=False, periph_width_mm=70, make_plots=True):
    """
    This function loads a given recording and apply all the analysis/preprocessing steps defined so far
    1. Load Neurophotometrics data
    2. Compute DF/F
    3. Video tracking (ROI definition, tracking...)
    4. Activity and Occupancy maps
    5. ROI analysis: time spent per ROI + average signal per ROI
    6. Extract snippets of signal around events, if any
    7. PETH computation and quantification

    Parameters
    ----------
    folder_path: Path
        Path to the folder containing the recording
    sites_names: dict
        Correspondence between the region names from Neurophotometrics (Region0G, Region1G) and brain region names
    analysis_path: str or Path
        Path to save the intermediate results
    trim_duration_s: float
        Duration of the signal to be removed at the beginning of the photometry recording before computing DF/F
        to discard the fast photobleaching period. In seconds.
    t_left_s: float
        Time in seconds to get before an event for the snippet extraction / PETH
        Default to 10
    t_right_s: float
        Time in seconds to get after an event for the snippet extraction / PETH
        Default to 20
    z_th: float
        Threshold, in height, for peak detection when quantifying the amplitude of the response in the PETH
        This is done on a zscore version of the PETH
    delay_s: float
        Minimal delay, in seconds, between two consecutive peaks.
        Defaut to 1 second.
    batch_num: tuple of int
        To select animals based on their batch number
    quality_th: int
        Quality threshold to select animals on. 0: no signal, 1: weak, 2: medium, 3: good
    bin_spacing: float
        For the activity and occupancy maps, the bin size in mm
    quality_path: Path, optional
        Path to the CSV file that contains quality estimates for each recording
    force: bool
        Shall we re-run the analysis even if it is already done, ignoring previously saved files
    output_video: bool
        Shall we produce a video with the video tracking output overlayed on it.
        At the moment requires an NVidia GPU
    bilateral_filter: bool
        Preprocess the frames with a bilateral filter. Can help but slows down the processing.
    periph_width_mm: float
        Width in mm of the periphery in the openfield.
        Default to 70 mm.
    make_plots: bool
        Shall we produce and save figures? Default to False

    Returns
    -------
    recording: dict
        Group all of the data and metadata extracted from this recording
    """
    recording = get_data(folder_path, sites_names, batch_num=batch_num, quality_th=quality_th,
                         quality_path=quality_path)
    recording = compute_dff_recording(recording, trim_duration_s, make_plots, fig_path=upaths['figures'])

    task = recording['metadata']['task'].lower()
    if task in {'oft', 'epm', 'social'}:
        _, track_paths = track_positions(recording['metadata']['data_behvideo'], analysis_path,
                                         force, output_video, bilateral_filter)
        recording['metadata'].update(track_paths)
    if 'tail' not in task:
        if 'oft' in task or 'social' in task:
            real_corners = np.array([[0, 0],
                                     [605, 0],
                                     [605, 395],
                                     [0, 395]])
        else:
            raise ValueError('Not implemented yet')
        recording = process_map(recording, real_corners, bin_spacing, show=make_plots, fig_path=upaths['figures'])
        recording['real_corners'] = real_corners
        recording = ra.roi_analysis(recording, periph_width_mm, make_plots, upaths['figures'])
    recording = extract_snippets_recording(recording, t_left_s, t_right_s)
    recording = peth_amp_recording(recording, z_th, delay_s)
    return recording


if __name__ == '__main__':

    # fp = upaths['data'] / 'revision batch 2/data/Tail1/CTRL/Glong/581'
    fp = upaths['data'] / 'OFT 0507/CNO/Glong/581'
    sites = {'Region0G': 'BNST', 'Region1G': 'CeA'}
    rec = get_recording(fp, sites_names=sites, analysis_path=upaths['analysis'], bin_spacing=10, make_plots=True)
    # rec = compute_dff_recording(rec)
    # rec = extract_snippets_recording(rec)

