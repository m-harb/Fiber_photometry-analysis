import matplotlib.pyplot as plt
import numpy as np

import shapely
from shapely.geometry import Polygon
from shapely.plotting import plot_polygon

from plotting import plot_roi_pos
from settings import colors, gunmetal
from geometry import oft_rois, find_crop_roi, social_rois


def roi_presence(centroids, roi_dict, roi_colors, show=False, figure_path=None):
    is_in = dict()
    centroid_points = np.array([shapely.geometry.Point(p) for p in centroids])

    for roi_name, roi_poly in roi_dict.items():
        # plot_polygon(roi_poly, color=colors[roi_name], ax=ax)
        c_mask = roi_poly.contains(centroid_points)
        is_in[roi_name] = c_mask
        # plot_points(centroid_points[c_mask], color=colors[roi_name], markersize=1)
    if show:
        fig, ax = plt.subplots()
        for name, poly in roi_dict.items():
            plot_polygon(poly, color=roi_colors[name], ax=ax)
        ax.plot(*centroids.T, '.', ms=1, c=gunmetal, alpha=.5)
        if figure_path is not None:
            fig.savefig(figure_path)

    return is_in


def roi_analysis(recording, periph_width_mm=70, show=False, fig_path=None):
    if not 'coords' in recording['video']:
        return recording
    rois_coords = np.load(recording['metadata']['rois'])
    task = recording['metadata']['task'].lower()
    trans = recording['video']['transform']
    real_rois_coords = {name: trans.inverse(coords) for name, coords in rois_coords.items()}
    maze_coords = find_crop_roi(real_rois_coords)
    rois = {}
    if 'oft' in task or 'social' in task:
        rois = oft_rois(maze_coords, periph_width_mm)
    if 'social' in task:
        rois.update(social_rois(real_rois_coords))
    centroids = recording['video']['coords']
    recording['video']['is_in'] = roi_presence(centroids, rois, colors)
    dt = np.median(np.diff(recording['video']['time']))
    # Time spent in each roi
    time_spent = {name: np.sum(is_in) * dt for name, is_in in recording['video']['is_in'].items()}
    recording['video']['time_spent'] = time_spent
    # Average photometry signal per recording site per roi
    all_roi_sig = {}
    for site, signal in recording['i_dff'].items():
        all_roi_sig[site] = {}
        for roi, is_in in recording['video']['is_in'].items():
            sig_roi = signal[is_in].mean()
            all_roi_sig[site][roi] = sig_roi
    recording['video']['roi_dff'] = all_roi_sig
    if show:
        plot_roi_pos(centroids, fig_path, rois, recording['metadata'])

    return recording


