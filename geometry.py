from typing import Dict

import numpy as np
import shapely as shp
from shapely import Polygon
from shapely.affinity import scale, rotate


def find_closest(points, pt):
    dist = np.linalg.norm(points - pt, axis=1)
    c_ix = np.argmin(dist)
    return c_ix


def point_sorter(points):
    points = np.atleast_2d(np.array(points).copy())
    m_point = np.mean(points, 0)
    points = points - m_point
    theta = np.arctan2(points[:, 1], points[:, 0])
    neg = points[:, 1] < 0
    theta[neg] = 2 * np.pi + theta[neg]
    return np.argsort(theta)


def oft_rois(of_corners, periph_width_mm=70):
    of_poly = shp.Polygon(of_corners)
    center_poly = shp.Polygon(shp.offset_curve(of_poly, -periph_width_mm))
    periph_poly = of_poly - center_poly
    return {'maze': of_poly, 'center': center_poly, 'periphery': periph_poly}


def find_crop_roi(roi_coords):
    roi_poly: Dict[str, Polygon] = {name: Polygon(coords) for name, coords in roi_coords.items()}
    for fname, first_roi in roi_poly.items():
        first_is_crop = True
        for sname, sec_roi in roi_poly.items():
            if not first_roi.contains(sec_roi):
                first_is_crop = False
                break
        if first_is_crop is True:
            f_coords = np.array(first_roi.exterior.coords.xy).astype(int)
            f_coords = f_coords[:, :-1]
            return f_coords.T
    raise ValueError('No crop ROI found')


def create_ellipse_geom(center, radius, angle):
    # Let create a circle of radius 1 around center point:
    circ = shp.geometry.Point(center).buffer(1)
    # Let create the ellipse along x and y:
    ell = scale(circ, radius[0], radius[1])
    # Let rotate the ellipse (clockwise, x axis pointing right):
    ellr = rotate(ell, angle)
    return ellr


def ellipse_from_points(points):
    center = points[0, :]
    vector = points[1, :] - center
    l_radius = np.linalg.norm(vector)
    s_radius = np.linalg.norm(points[2, :] - center)
    alpha = 180 * np.arctan2(vector[1], vector[0]) / np.pi
    return center, (l_radius, s_radius), alpha


def social_rois(rois, ring_width_mm=40):
    social_boxes = {}
    for roi_name, coords in rois.items():
        if coords.shape[0] != 3:
            continue
        ell = ellipse_from_points(coords)
        ell_geom = create_ellipse_geom(*ell)
        social_boxes[roi_name] = ell_geom
        d_ell = Polygon(shp.offset_curve(ell_geom, ring_width_mm))
        ring = d_ell - ell_geom
        social_boxes[f'{roi_name}_ring'] = ring
    return social_boxes
