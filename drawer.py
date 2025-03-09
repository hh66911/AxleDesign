# drawer.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
from numpy import ndarray

import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Drawer:
    def __init__(self, figsize):
        self.fig = plt.figure(figsize=figsize)
        self.ax = plt.gca()

    def line(self, pt1, pt2, color='black'):
        """绘制从pt1到pt2的线段"""
        x = [pt1[0], pt2[0]]
        y = [pt1[1], pt2[1]]
        self.ax.plot(x, y, color=color)

    def rect(self, pt1, pt2, color='black'):
        """绘制以pt1和pt2为对角的矩形"""
        x1, y1 = pt1
        x2, y2 = pt2
        min_x, max_x = sorted([x1, x2])
        min_y, max_y = sorted([y1, y2])
        width = max_x - min_x
        height = max_y - min_y
        rectangle = patches.Rectangle(
            (min_x, min_y), width, height,
            linewidth=1, edgecolor=color, facecolor='none'
        )
        self.ax.add_patch(rectangle)

    def polyline(self, *pts, color='black'):
        """绘制连接多个点的折线"""
        x = [pt[0] for pt in pts]
        y = [pt[1] for pt in pts]
        self.ax.plot(x, y, color=color)

    def arc(self, center, radius, start_angle, stop_angle, color='black'):
        """绘制以center为中心、radius为半径的圆弧，角度范围从start到stop"""
        arc_patch = patches.Arc(
            center, 2*radius, 2*radius,
            angle=0, theta1=start_angle, theta2=stop_angle,
            edgecolor=color
        )
        self.ax.add_patch(arc_patch)

    def wipeout(self, *pts, color='white'):
        """填充由多个点定义的多边形"""
        polygon = patches.Polygon(
            pts,
            edgecolor='black',  # 黑色边框
            facecolor=color,    # 填充颜色
            linewidth=1         # 边框宽度
        )
        self.ax.add_patch(polygon)


class Path2D:
    def __init__(self, start_pos=np.zeros(2)):
        if not isinstance(start_pos, ndarray):
            start_pos = np.array(start_pos)
        self.points = [start_pos]

    def __repr__(self):
        return f'Path2D({self.points})'

    def __str__(self):
        fmt_pts = ' -> '.join([f'({p[0]: .2f}, {p[1]: .2f})' for p in self.points])
        return f'{fmt_pts}'

    def offset(self, x_or_seq, y=None):
        if y is not None:
            off = np.array((x_or_seq, y))
        elif not isinstance(x_or_seq, ndarray):
            off = np.array(x_or_seq)
        else:
            raise ValueError('offset维数不对')
        self.points.append(self.points[-1] + off)

    def goto(self, x_or_seq, y=None):
        if y is not None:
            pt = np.array((x_or_seq, y))
        elif not isinstance(x_or_seq, ndarray):
            pt = np.array(x_or_seq)
        else:
            raise ValueError('point维数不对')
        self.points.append(pt)

    def draw(self, drawer: Drawer):
        if len(self.points) == 2:
            return drawer.line(*self.points)
        else:
            return drawer.polyline(*self.points)

    def wipeout(self, drawer: Drawer):
        return drawer.wipeout(*self.points)
