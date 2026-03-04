import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from matplotlib.backend_bases import MouseButton
from matplotlib.patches import PathPatch
from geometry import find_closest, point_sorter, create_ellipse_geom
from shapely.geometry import Polygon
import shapely as shp
from shapely.plotting import plot_polygon, patch_from_polygon


class ROI:
    def __init__(self, name='') -> None:
        super().__init__()
        self.name = name
        self.points = []
        self.geometry = None
        self._patch = None
        self.picked = None
        self.picked_ix = None
        self._text = None
        self._ax = None
        self.geometry_type = None

    @property
    def mid_pt(self):
        return shp.centroid(self.geometry)

    @property
    def text(self):
        if self._text is None and self.ax is not None:
            self._text = self.ax.text(self.mid_pt.x, self.mid_pt.y, self.name)
        return self._text

    @property
    def patch(self):
        return self._patch

    @patch.setter
    def patch(self, value):
        self._patch = value

    @property
    def ax(self):
        if self.patch is None:
            return None
        return self.patch.axes

    def add_point(self, new_point, m_point):
        self.points.append(new_point)
        order = point_sorter(self.points)
        self.points = [self.points[ix] for ix in order]
        self._update_geometry()

    def remove_point(self, point_ix):
        self.points.pop(point_ix)
        self._update_geometry()

    def update_point(self, point_ix, new_point):
        self.points[point_ix] = new_point
        self._update_geometry()

    def _update_geometry(self):
        pass


class PolyROI(ROI):
    def __init__(self, name='') -> None:
        super().__init__(name)
        self.geometry = Polygon(self.points)
        self.geometry_type = 'Polygon'

    def _update_geometry(self):
        if len(self.points) < 3:
            return
        self.geometry = Polygon(self.points).normalize()


class EllipseROI(ROI):
    def __init__(self, name='') -> None:
        super().__init__(name)
        self._angle = None
        self._radius = None
        self._center = None
        self.geometry_type = 'Ellipse'
        self.points = [(0, 0) for _ in range(3)]
        self.m_points = []

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, value):
        self._center = value
        self.points[0] = value
        self._compute_points()

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = tuple([int(x) for x in value])
        self._compute_points()

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        self._angle = value
        self._compute_points()

    def _update_geometry(self):
        ellr = create_ellipse_geom(self.center, self.radius, self.angle)

        self.geometry = ellr
        for pt, mp in zip(self.points, self.m_points):
            mp.set_data(pt)

    def add_point(self, new_point, m_point):
        self.m_points.append(m_point)

    def update_point(self, point_ix, new_point):
        self.points[point_ix] = new_point
        if point_ix == 0:
            self.center = new_point
        elif point_ix == 1:
            v = np.array(self.points[1]) - np.array(self.center)
            self.radius = (np.linalg.norm(v), self.radius[1])
            self.angle = 180 * np.arctan2(v[1], v[0])  / np.pi
        elif point_ix == 2:
            v = np.array(self.points[2]) - np.array(self.center)
            self.radius = (self.radius[0], np.linalg.norm(v))
        self._update_geometry()

    def _compute_points(self):
        if self.angle is None or self.radius is None or self.center is None:
            return
        alpha = np.pi * self.angle / 180
        x = self.radius[0] * np.cos(alpha) + self.center[0]
        y = self.radius[0] * np.sin(alpha) + self.center[1]
        self.points[1] = (x, y)
        x = self.radius[1] * np.cos(alpha + np.pi / 2) + self.center[0]
        y = self.radius[1] * np.sin(alpha + np.pi / 2) + self.center[1]
        self.points[2] = (x , y)
        self._update_geometry()


class UI:
    def __init__(self, img=None, roi_name='ROI1', roi_type='Polygon', roi_values=None) -> None:
        self.fig, self.ax = plt.subplots(1, 1, figsize=(18, 12))
        self.fig.set_tight_layout(False)
        plt.subplots_adjust(right=.65)
        if img is None:
            self.ax.set_xlim(-10, 10)
            self.ax.set_ylim(-10, 10)
        else:
            self.ax.imshow(img)
        self.ax_name = self.fig.add_axes([.7, .7, .15, .02])
        self.name_tb = TextBox(self.ax_name, 'ROI name', roi_name)
        self.ax_new = self.fig.add_axes([.87, .7, 0.05, .02])
        self.new_btn = Button(self.ax_new, 'New')
        self.new_btn.on_clicked(self.new)

        self.roi_type = roi_type
        self.roi = None
        self.rois = []
        self.c_patch = None
        self.moving = False

        if roi_values is not None:
            if roi_type == 'Ellipse':
                self.roi = EllipseROI(roi_name)
                self.roi.center = roi_values[0]
                self.roi.radius = roi_values[1]
                self.roi.angle = roi_values[2]
                self.rois.append(self.roi)
                for p in self.roi.points:
                    self.add_roi_point(*p)
                # self.add_roi_point(*self.roi.points[1])
                # self.add_roi_point(self.)

                self.update_patch()

        self.connect()
        plt.show(block=True)

    @property
    def rois_coords(self):
        coords = {roi.name: np.array(roi.points) for roi in self.rois}
        return coords

    def new(self, event):
        if self.roi_type == 'Polygon':
            self.roi = PolyROI(self.name_tb.text)
        if self.roi_type == 'Ellipse':
            pass
        self.rois.append(self.roi)

    def connect(self):
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key)

    def on_key(self, event):
        if event.key == 'n':
            self.new(None)

    def on_move(self, event):
        if event.inaxes is not self.ax:
            return
        if self.roi is None:
            return
        if self.roi.picked is None:
            return
        if not self.moving:
            return
        self.roi.picked.set_data([event.xdata, event.ydata])
        self.roi.update_point(self.roi.picked_ix, (event.xdata, event.ydata))
        self.update_patch()

    def on_release(self, event):
        if self.roi is None:
            return
        if event.button != MouseButton.LEFT or event.inaxes is not self.ax:
            return
        if self.moving:
            self.moving = False
            self.roi.picked = None
            self.roi.picked_ix = None
            self.update_patch()
        else:
            if self.roi_type != 'Ellipse':   # Ugly but easy
                self.add_roi_point(event.xdata, event.ydata)

    def on_pick(self, event):
        x, y = event.artist.get_data()
        pt = np.array([x, y]).squeeze()
        picked_ix = find_closest(self.roi.points, pt)
        if event.mouseevent.button == MouseButton.LEFT:
            self.roi.picked = event.artist
            self.roi.picked_ix = picked_ix
            self.moving = True
        else:
            self.roi.remove_point(picked_ix)
            self.update_patch()
            event.artist.remove()
            # self.remove_roi_point(self.ax.lines.index(event.artist))

    def remove_roi_point(self, ix):
        self.ax.lines.pop(ix)
        self.fig.canvas.draw()

    def add_roi_point(self, x, y):
        p, = self.ax.plot(x, y, 'o', picker=True)
        self.fig.canvas.draw()
        self.roi.add_point((x, y), p)
        self.update_patch()

    def update_patch(self):
        if len(self.roi.points) <= 2:
            return
        if self.roi.patch is not None:
            self.c_patch.remove()

        if self.roi.geometry_type == 'Polygon':
            self.roi.patch = patch_from_polygon(self.roi.geometry, facecolor='#56CFF033')
        elif self.roi.geometry_type == 'Ellipse':
            self.roi.patch = patch_from_polygon(self.roi.geometry, facecolor='#56CFF033')
        self.c_patch = self.ax.add_patch(self.roi.patch)

        self.roi.text.set_position((self.roi.mid_pt.x, self.roi.mid_pt.y))
        self.fig.canvas.draw()


if __name__ == '__main__':
    w = UI()

#
# class ROI:
#     def __init__(self, name='') -> None:
#         super().__init__()
#         self.name = name
#         self.points = []
#         self.polygon = Polygon(self.points)
#         self._patch = None
#         self.picked = None
#         self.picked_ix = None
#         self._text = None
#         self._ax = None
#
#     @property
#     def mid_pt(self):
#         return shp.centroid(self.polygon)
#
#     @property
#     def text(self):
#         if self._text is None and self.ax is not None:
#             self._text = self.ax.text(self.mid_pt.x, self.mid_pt.y, self.name)
#         return self._text
#
#     @property
#     def patch(self):
#         return self._patch
#
#     @patch.setter
#     def patch(self, value):
#         self._patch = value
#
#     @property
#     def ax(self):
#         if self.patch is None:
#             return None
#         return self.patch.axes
#
#     def add_point(self, new_point):
#         self.points.append(new_point)
#         order = point_sorter(self.points)
#         self.points = [self.points[ix] for ix in order]
#         self._update_polygon()
#
#     def remove_point(self, point_ix):
#         self.points.pop(point_ix)
#         self._update_polygon()
#
#     def update_point(self, point_ix, new_point):
#         self.points[point_ix] = new_point
#         self._update_polygon()
#
#     def _update_polygon(self):
#         if len(self.points) < 3:
#             return
#         self.polygon = Polygon(self.points).normalize()
#
#
#
#
#
