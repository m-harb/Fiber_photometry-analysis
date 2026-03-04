import numpy as np
from skimage import transform
from scipy.stats import binned_statistic_2d
from astropy.convolution import convolve

import plotting
import utilities as u
from geometry import point_sorter


def load_space_data(centroid_path, crop_path, real_corners):
    """
    Reload the animal positions, from video tracking, normalize them and convert them to real units
    using the outline of the maze as a reference.
    We estimate a transform that can bring the crop coordinates to the real world frame of reference
    and we apply it to the centroid coordinates

    Parameters
    ----------
    centroid_path: Path or str
        Path to the centroids, as saved by the video tracking procedure in
        videotracking.track_positions
    crop_path: Path or str
        Path to the coordinates used for cropping the image around the maze
    real_corners: np.ndarray
        Array containing the coordinates of the maze corners, in real units

    Returns
    -------
    real_centroids: np.ndarray
        Animal coordinates in the real world frame of reference
    trans:
        Matrix to go from real world to image frame of reference.
        We used the inverse to transform the centroid coordinates
    """
    try:
        centroids = np.load(centroid_path)
        crop_coords = np.load(crop_path)
    except FileNotFoundError:
        print(centroid_path, crop_path)
        return None
    # We then normalize the coordinates into real units and with a maze with its lower left corner at 0, 0
    centroids += crop_coords.min(0)
    order = point_sorter(crop_coords)
    crop_coords = crop_coords[order]
    order = point_sorter(real_corners)
    real_corners = real_corners[order]

    # Estimate transformation to normalize the representation
    trans = transform.estimate_transform('projective', real_corners, crop_coords)
    real_centroids = trans.inverse(centroids)
    return real_centroids, trans


def align_photometry(recording, coords):
    """
    For a given recording interpolate the DF/F of all sites so that it is sampled at the same time that
    video frames are taken. This way we can relate animal position and photometry signal

    Parameters
    ----------
    recording: dict
        Recording dictionary
    coords: np.ndarray
        Array with animal coordinates. Will be trimmed to restrict it to a period
        in which we have photometry

    Returns
    -------
    recording: dict
        Updated recording
    """
    v_time = recording['video']['time']
    pm_time = recording['dff_time']
    if pm_time is None:
        return recording
    g_vt = v_time >= pm_time[0]
    v_time = v_time[g_vt]
    coords = coords[g_vt, :]
    recording['i_dff'] = {site: u.time_interpolation(v_time, pm_time, sig)
                          for site, sig in recording['dff'].items()}
    recording['video']['coords'] = coords
    recording['video']['t_time'] = v_time
    return recording


def activity_map(recording, real_corners, bin_spacing=5, kernel_width=3):
    """
    Computes the activity map: for each animal position, gives the average photometry signal in each recording site

    Parameters
    ----------
    recording: dict
    real_corners: np.ndarray
        Maze corners in real world coordinates
    bin_spacing: float
        Spatial bin size in mm
    kernel_width: int
        Smoothing neighboring size

    Returns
    -------
    recording: dict
        Updated recording. Adds the following keys under 'video':
        b_map       : Binned activity map
        s_map       : Smoothed activity map
        occ_map     : Occupancy map
        s_occ_map   : Smoothed occupancy map
    """
    if 'coords' not in recording['video']:
        return recording
    coords = recording['video']['coords']
    coords_min = np.nanmin(real_corners, 0)
    coords_max = np.nanmax(real_corners, 0)
    x_bins = np.arange(coords_min[1], coords_max[1], bin_spacing)
    y_bins = np.arange(coords_min[0], coords_max[0], bin_spacing)
    recording['video']['b_map'] = {}
    recording['video']['s_map'] = {}
    recording['video']['occ_map'] = {}
    recording['video']['s_occ_map'] = {}
    for site, signal in recording['i_dff'].items():
        binned_pm_map, xe, ye, bn = binned_statistic_2d(coords[:, 1], coords[:, 0], signal,
                                                        bins=[x_bins, y_bins], statistic='mean')
        binned_occ_map, xe, ye, bn = binned_statistic_2d(coords[:, 1], coords[:, 0], np.ones(coords.shape[0]),
                                                         bins=[x_bins, y_bins], statistic='count')
        recording['video']['b_map'][site] = binned_pm_map
        recording['video']['occ_map'][site] = binned_occ_map
        kernel = np.ones((kernel_width, kernel_width)) / kernel_width ** 2
        recording['video']['s_map'][site] = convolve(binned_pm_map, kernel)
        recording['video']['s_occ_map'][site] = convolve(binned_occ_map, kernel)
    return recording


def process_map(recording, real_corners, bin_spacing=5, kernel_width=3, show=False, fig_path=None):
    """
    Computing activity and occupancy maps for a given recording, taking  care of coordinates
    normalization and eventually plotting them.

    Parameters
    ----------
    recording
    real_corners
    bin_spacing
    kernel_width
    show
    fig_path

    Returns
    -------

    """
    coords, trans = load_space_data(recording['metadata']['centroid'], recording['metadata']['crop'], real_corners)
    if coords is None:
        # TODO: Create other keys (map, coords)
        recording['video']['s_map'] = {site: None for site in recording['dff'].keys()}
        return recording
    recording['video']['transform'] = trans
    recording = align_photometry(recording, coords)
    recording = activity_map(recording, real_corners, bin_spacing, kernel_width)
    if show:
        plotting.plot_maps_recording(recording, fig_path)
    return recording

