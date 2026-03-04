from typing import Dict
from pathlib import Path
from pims import Video
import cv2
import numpy as np
from tqdm import tqdm
from imageio_ffmpeg import write_frames
from geometry import find_crop_roi, ellipse_from_points
from open_data import get_metadata
from roi_interface import UI
from settings import upaths


def get_bg(video):
    """
    process the median of 20 frames randomly taken from the first 1200 frames
    Parameters
    ----------
    video: a pims object, a handle on an open video file

    Returns
    -------
    bg: the median frame
    """
    frame = video[0]
    n = len(video)
    rng = np.random.default_rng()
    i_frames = np.sort(rng.integers(low=0, high=min(n - 1, 100 * 120), size=20))
    gray_stack = np.zeros((frame.shape[0], frame.shape[1], 20))
    for i, ifr in tqdm(enumerate(i_frames), desc='computing background'):
        gray_stack[:, :, i] = cv2.cvtColor(video[ifr], cv2.COLOR_BGR2GRAY)
    bg = np.median(gray_stack, axis=2)
    return bg


def get_social_boxes(crop_bg):
    """
    to automatically fit an ellispe on each social cages
    Parameters
    ----------
    crop_bg: the bg framed crop to remove the walls of the Open Field

    Returns
    -------
    e, e2 the parameters of each ellipse
    """
    edge = cv2.Canny(np.uint8(crop_bg), 50, 150)
    edge_contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cages = sorted(edge_contours, key=cv2.contourArea)[-2:]
    e = list(cv2.fitEllipse(cages[0]))
    e2 = list(cv2.fitEllipse(cages[1]))
    e[1] = [x / 2 for x in e[1]]
    e2[1] = [x / 2 for x in e2[1]]
    return e, e2


def define_rois(metadata, path_dict, force=False):
    """
    GUI to pick the corners of each ROIS
    the maze, the crop (rectangle to use to crop the video), both social rois (left and right)
    Parameters
    ----------
    metadata: contains all the info on thez recording (mouse, task, injection, paths, ...)
    path_dict : the path to the already saved roi information
    force : if you want to overwrite this already saved information

    Returns
    -------
        roi_coords: the dictionnary with all rois
        crop_coords: the 4 coordinates to crop the video
        path_dict: all the paths to the already saved information regarding the rois
    """
    rois_path = path_dict['rois']
    crop_path = path_dict['crop']
    social_rois = path_dict['social_rois']
    last_social_rois = path_dict['last_social_rois']
    video = Video(metadata['data_behvideo'])
    frame0 = video[0]

    if (not force) and rois_path.exists():
        roi_coords = np.load(rois_path)
        roi_coords = dict(roi_coords)
        crop_coords = np.load(crop_path)
    else:
        roi_ui = UI(frame0, roi_name='maze')
        roi_coords = roi_ui.rois_coords
        if len(roi_coords) != 0:
            np.savez(rois_path, **roi_coords)
            crop_coords = find_crop_roi(roi_coords)
            np.save(crop_path, crop_coords)

    np.save(path_dict['frame0'], frame0)

    if metadata['task'] == 'Social':
        if social_rois.exists():
            social_roi_dict = np.load(social_rois)
        else:
            if last_social_rois.exists():
                boxes_dict = np.load(last_social_rois)
                boxes_dict = {side: ellipse_from_points(pts) for side, pts in boxes_dict.items()}
            else:
                default_ell = ((frame0.shape[1] / 2, frame0.shape[0] / 2), (50, 50), 0)
                boxes_dict = {'left': default_ell, 'right': default_ell}
            social_roi_dict = dict()
            for side, b in boxes_dict.items():
                roi_ui = UI(frame0, roi_name=side, roi_type='Ellipse', roi_values=b)
                c_roi = roi_ui.rois_coords
                social_roi_dict[side] = c_roi[side]
            np.savez(last_social_rois, **social_roi_dict)
            np.savez(social_rois, **social_roi_dict)

        roi_coords.update(** social_roi_dict)

    return roi_coords, crop_coords, path_dict


def subtract_bg(gray, bg):
    """
    To improve the video tracking we substract the bg (a frame without the mouse) to each frame of the video (gray here)
    Parameters
    ----------
    gray : each frame of the video
    bg :a frame without the mouse

    Returns
    -------
    The substraction of gray - bg

    """
    if bg is None:
        return gray
    # return cv2.subtract(gray, bg, dtype=cv2.CV_8U)
    tmp = gray - bg
    tmp = tmp - tmp.min()
    tmp = tmp / tmp.max()
    tmp *= 255
    gray_c = np.uint8(tmp)
    return gray_c


def track_positions(video_path, analysis_path, force=False, output_video=False, bilateral_filter=False):
    """
    perform the video tracking on the entire behavioral video and retourn the position (centroid) of the mouse at each frame
    Parameters
    ----------
    video_path: a path to the behavioral video file
    analysis_path: to place to store the results
    force : if you want to recalculate everything (as it is time consumming we save each step of the analysis)
    output_video : if you want to create a new_video with the centroid position on top of each frame
    bilateral_filter : if you want to apply the bilateral filter (time consumming)

    Returns
    -------
    centroids: the position (centroid) of the mouse at each frame
    paths_dict: the path to the saved analysis files for future use in the pipeline
    """
    video_path = Path(video_path)
    metadata = get_metadata(video_path)
    subanalysis_path = analysis_path / "tracking" / metadata['task']
    subanalysis_path.mkdir(parents=True, exist_ok=True)
    centroid_path = subanalysis_path / f"centroids_{metadata['mid']}.npy"
    new_video_path = subanalysis_path / f"video_{metadata['mid']}.mp4"

    rois_path = subanalysis_path / f"rois_{metadata['mid']}.npz"
    crop_path = subanalysis_path / f"crop_{metadata['mid']}.npy"
    frame0_path = subanalysis_path / f"frame0_{metadata['mid']}.npy"
    last_social_rois = subanalysis_path / f'last_social_rois.npz'
    social_rois = subanalysis_path / f"social_rois_{metadata['mid']}.npz"

    paths_dict = {'centroid': centroid_path, 'rois': rois_path, 'crop': crop_path, 'frame0': frame0_path,
                  'social_rois': social_rois, 'last_social_rois': last_social_rois}

    roi_coords, crop_coords, paths_dict = define_rois(metadata, paths_dict, force=force)
    np.savez(rois_path, **roi_coords)
    x_min, y_min = crop_coords.min(axis=0)
    x_max, y_max = crop_coords.max(axis=0)

    video = Video(metadata['data_behvideo'])
    frame = video[0]
    bg = None

    if metadata['task'] == 'Social':
        bg_path = subanalysis_path / f"bg_{metadata['mid']}.npy"
        if bg_path.exists():
            bg = np.load(bg_path)
        else:
            bg = get_bg(video)
            np.save(bg_path, bg)
        bg = bg[y_min:y_max, x_min:x_max]

    if (not force) and centroid_path.exists():
        return np.load(centroid_path), paths_dict

    centroids = []

    writer = None
    if output_video:
        if new_video_path.exists() and (not force):
            output_video = False

    if output_video:
        frame0 = frame[y_min:y_max, x_min:x_max, :]
        writer = write_frames(new_video_path,
                              (frame0.shape[1], frame0.shape[0]), fps=73, codec='h264_nvenc',
                              bitrate='2M')
        writer.send(None)  # seed the generator

    for frame in tqdm(video, position=1):
        frame0 = frame[y_min:y_max, x_min:x_max, :]
        gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        gray = subtract_bg(gray, bg)
        blur = gray  # if we don't do the bilateralFilter
        if bilateral_filter:
            blur = cv2.bilateralFilter(gray, d=-11, sigmaColor=10, sigmaSpace=7)
        _, thresh = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY_INV)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours):
            contour = max(contours, key=cv2.contourArea)
            centroid = np.round(np.mean(np.squeeze(contour), 0)).astype(int)
        else:
            centroid = (np.nan, np.nan)
            contour = []
        centroids.append(centroid)

        if writer is not None:
            frame0 = cv2.drawContours(frame0, [contour], 0, (255, 255, 255))
            frame0 = cv2.drawMarker(frame0, centroid, (255, 0, 0), markerType=cv2.MARKER_CROSS)
            writer.send(np.ascontiguousarray(frame0))

    if output_video:
        writer.close()

    centroids = np.array(centroids)
    np.save(centroid_path, centroids)

    video.close()

    return centroids, paths_dict


if __name__ == '__main__':
    # for vp in tqdm(upaths['data'].rglob('*.avi'), position=0):
    #     c = track_positions(vp, upaths['analysis'], output_video=False, force=False)

    vp = Path(upaths['data'] / "batch try/OFT/CNO/Glong/624/data_behvideo_2024-05-14T11_30_17.avi")
    c = track_positions(vp, upaths['analysis'], output_video=True, force=True)
