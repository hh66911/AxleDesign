from enum import Enum
import math
import warnings
from dataclasses import dataclass

import numpy as np
from numpy import ndarray

from drawer import Drawer, Path2D


class BadDesignWarning(UserWarning):
    pass


class Angle:
    def __init__(self, degrees):
        # 初始化角度值，以度为单位
        self._degrees = degrees

    def __repr__(self):
        # 输出时只输出角度值
        return f'A({self._degrees})'

    def __str__(self):
        # 字符串表示，输出度分秒
        degrees = self._degrees
        deg = int(degrees)
        minutes = (degrees - deg) * 60
        min = int(minutes)
        seconds = (minutes - min) * 60
        return f"{deg}°{min}'{seconds : <.2f}″"

    def __float__(self):
        # 数字表示，输出角度值
        return float(self._degrees)

    def to_radians(self):
        # 将角度转换为弧度
        return math.radians(self._degrees)

    def sin(self):
        # 计算正弦值
        return math.sin(self.to_radians())

    def cos(self):
        # 计算余弦值
        return math.cos(self.to_radians())

    def tan(self):
        return math.tan(self.to_radians())


@dataclass
class Fillet:
    radius: float
    center: tuple
    pts: list
    start_angle: float
    stop_angle: float


@dataclass
class _StepFeature:
    position: float
    size: float
    is_abs: bool

    def __iter__(self):
        return iter((self.position, self.size, self.is_abs))


@dataclass
class _ShoulderFeature:
    position: float
    width: float


@dataclass
class _BushingFeature:
    position: float
    height: float
    width: float


@dataclass
class _KeywayFeature:
    position: float
    width: float
    length: float
    forward: bool


@dataclass
class _GearFeature:
    position: float
    bold: float
    da: float
    forward: bool


@dataclass
class _BearingFeature:
    position: float
    bold: float
    da: float
    forward: bool


class PutSide(Enum):
    AFTER = 'after'
    BEFORE = 'before'


def _get_offset(feat, halfl, put_side):
    if isinstance(feat, _StepFeature):
        offset = -halfl
        if put_side == PutSide.AFTER:
            offset = -offset
    elif isinstance(feat, _ShoulderFeature):
        offset = -halfl
        if put_side == PutSide.AFTER:
            offset = -offset + feat.width
    else:
        # 确定特征宽度
        if isinstance(feat, _BushingFeature):
            feature_width = feat.width / 2
        elif isinstance(feat, _GearFeature):
            feature_width = feat.gear.half_bold
        elif isinstance(feat, _BearingFeature):
            feature_width = feat.bearing.b / 2
        else:
            raise ValueError("不支持的特征类型。")

        # 计算偏移量
        if put_side == PutSide.BEFORE:
            offset = -halfl - feature_width
        else:
            offset = halfl + feature_width
    return offset


class Shaft:
    CR_TABLE = {
        (0, 3): 0.2, (3, 6): 0.4,
        (6, 10): 0.6, (10, 18): 0.8,
        (18, 30): 1.0, (30, 50): 1.6,
        (50, 80): 2.0, (80, 120): 2.5,
        (120, 180): 3.0, (180, 250): 4.0,
        (250, 320): 5.0, (320, 400): 6.0,
        (400, 500): 8.0, (500, 630): 10,
        (630, 800): 12,  (800, 1000): 16,
        (1000, 1250): 20,  (1250, 1600): 25,
    }

    @staticmethod
    def _get_chamfer_radius(diameter):
        for k, v in Shaft.CR_TABLE.items():
            if k[0] < diameter <= k[1]:
                return v
        warnings.warn(f"直径 {diameter} 超出了倒角半径计算的范围。", BadDesignWarning)

    def __init__(self, init_diam):
        self.initial_diameter = init_diam
        self.length = None
        self.steps: list[_StepFeature] = []

        self.contour = []            # 原始轮廓
        self.chamfered_contour = []  # 倒角处理后的轮廓
        self.chamfer_mode = {
            'fillet': None,
            'chamfer': None
        }
        self.need_refresh = True

        self.gears: list[_GearFeature] = []
        self.keyways: list[_KeywayFeature] = []
        self.bearings: list[_BearingFeature] = []
        self.bushings: list[_BushingFeature] = []

    def add_step(self, position, height=None, diameter=None):
        if height is not None:
            feat = _StepFeature(position, height, False)
        elif diameter is not None:
            feat = _StepFeature(position, diameter, True)
        else:
            raise ValueError("必须提供高度或直径。")
        self.need_refresh = True  # 需要重新计算轮廓
        self.steps.append(feat)
        return feat
    
    def end_at(self, position):
        self.steps.append(_StepFeature(position, self.initial_diameter, True))

    def add_shoulder(self, position, height, width):
        if width < height * 1.4:
            warnings.warn(f"{position} 处的肩部过窄，一般应大于高度的 1.4 倍。",
                          BadDesignWarning)
        d = self._get_diameter_at(position, False)
        if not 0.07 * d <= height <= 0.1 * d:
            warnings.warn(
                f"{position} 处的肩部高度 {height} 不在推荐范围 ({0.07 * d}, {0.1 * d}) 内。",
                BadDesignWarning)

        self.need_refresh = True  # 需要重新计算轮廓
        self.steps.append(_StepFeature(position, height, False))
        self.steps.append(_StepFeature(position + width, -height, False))
        return _ShoulderFeature(position, width)

    def add_bushing(self, feat, height, width, put_side=PutSide.BEFORE):
        offset = _get_offset(feat, width / 2, put_side)
        pos = feat.position + offset
        d1 = self._get_diameter_at(pos, False)
        result = _BushingFeature(pos, height, width)
        self.bushings.append(result)
        return result

    def add_keyway(self, feat, length, width, forward=True):
        if isinstance(feat, (_GearFeature,)):
            pos = feat.position
        else:
            raise ValueError("不支持的特征类型。")
        self.keyways.append(_KeywayFeature(
            pos, width, length, forward
        ))
        return self.keyways[-1]

    def add_gear(self, pos_or_feat, da, bold,
                 forward=True, put_side=PutSide.BEFORE):
        if not isinstance(pos_or_feat, float):
            pos_or_feat = pos_or_feat.position + _get_offset(
                pos_or_feat, bold / 2, put_side)

        self.gears.append(
            _GearFeature(pos_or_feat, bold, da, forward
                         ))
        return self.gears[-1]

    def add_bearing(self, feat, da, bold,
                    forward=True, put_side=PutSide.BEFORE):
        if not isinstance(feat, _BushingFeature):
            raise NotImplementedError(f"不支持的特征类型: {type(feat)}")
        pos = feat.position + _get_offset(
            feat, bold / 2, put_side)
        self.bearings.append(_BearingFeature(pos, bold, da, forward))
        return self.bearings[-1]

    def _get_diameter_at(self, pos, check_length=True,
                         put_side=PutSide.BEFORE):
        self.process_features(False, False)
        if check_length and (pos < 0 or pos > self.length):
            raise ValueError(f"位置 {pos} 超出了轴的长度范围。")
        if put_side == PutSide.BEFORE:
            def _check(x0, p, x1):
                return x0 <= p < x1
        else:
            def _check(x0, p, x1):
                return x0 < p <= x1
        for i in range(len(self.contour) - 1):
            x0, d0 = self.contour[i]
            x1, _ = self.contour[i + 1]
            if _check(x0, pos, x1):
                return d0
        return self.contour[-1][1]

    def process_features(self, do_fillet=False, do_chamfer=True, num_pt_per_arc=5):
        if not (self.need_refresh) and (
            self.chamfer_mode['fillet'] == do_fillet and
            self.chamfer_mode['chamfer'] == do_chamfer
        ):
            return

        events = []
        events.append((0, self.initial_diameter))

        # 处理阶梯特征
        current_diam = self.initial_diameter
        self.steps.sort(key=lambda x: x.position)
        for pos, l, absolute in self.steps:
            if absolute:
                current_diam = l
            else:
                current_diam += l
            events.append((pos, current_diam))

        # 生成基础轮廓
        self.contour = []
        current_diam = self.initial_diameter
        self.contour.append((0, current_diam))
        for pos, diam in events:
            if diam != current_diam:
                self.contour.extend([(pos, current_diam), (pos, diam)])
            else:
                self.contour.append((pos, diam))
            current_diam = diam

        # 合并相同点
        merged_contour = []
        for i, c in enumerate(self.contour):
            if i == 0 or c != merged_contour[-1]:
                merged_contour.append(c)
        self.contour = merged_contour

        if do_chamfer:
            self.length = self.contour[-1][0]
            pos, d = self.contour[0]
            self.contour[0] = (pos + 1, d)
            self.contour.insert(0, (pos, d - 2))
            pos, d = self.contour[-1]
            self.contour[-1] = (pos - 1, d)
            self.contour.append((pos, d - 2))
        self.chamfer_mode['chamfer'] = do_chamfer
        # 倒角处理
        if do_fillet:
            self._apply_chamfers(num_pt_per_arc)
        else:
            self.chamfered_contour = [(x, y / 2) for x, y in self.contour]
        self.chamfer_mode['fillet'] = do_fillet

    def _apply_chamfers(self, num_pt_per_arc):
        self.chamfered_contour = []

        i = 0
        while i < len(self.contour) - 1:
            x0, d0 = self.contour[i]
            x1, d1 = self.contour[i+1]

            if x0 == x1 and d0 != d1:  # 垂直段（直径变化点）
                # 转换为半径单位进行计算
                r0, r1 = d0 / 2, d1 / 2
                delta_r = r1 - r0
                fradius = Shaft._get_chamfer_radius(min(r0, r1))

                # 内侧
                cx = x0 - np.sign(delta_r) * fradius
                cy = min(r0, r1) + fradius
                if delta_r > 0:
                    start_angle = -np.pi / 2
                    stop_angle = 0
                else:
                    start_angle = -np.pi
                    stop_angle = -np.pi / 2

                # 生成内圆角坐标（直径单位）
                theta = np.linspace(start_angle, stop_angle, num_pt_per_arc)
                pts = zip(cx + fradius * np.cos(theta),
                          cy + fradius * np.sin(theta))

                # 合并并排序坐标点
                if delta_r > 0:
                    self.chamfered_contour.append((cx, d0 / 2))
                else:
                    self.chamfered_contour.append((x0, d0 / 2))
                    self.chamfered_contour.append((x0, cy))
                self.chamfered_contour.append(Fillet(
                    fradius, (cx, cy), list(pts),
                    start_angle, stop_angle,
                ))
                if delta_r > 0:
                    self.chamfered_contour.append((x1, cy))
                    self.chamfered_contour.append((x1, d1 / 2))
                else:
                    self.chamfered_contour.append((cx, d1 / 2))
                i += 1
            else:  # 水平段
                self.chamfered_contour.append((x0, d0 / 2))
            i += 1

        # 添加轮廓末端
        last = self.contour[-1]
        self.chamfered_contour.append((last[0], last[1] / 2))

    def _draw_half_contour(self, drawer: Drawer, mirrored=False):
        path = Path2D((0, 0))
        for segment in self.chamfered_contour:
            if isinstance(segment, Fillet):
                path.draw(drawer)
                path = None
                center = (
                    segment.center[0],
                    -segment.center[1]
                ) if mirrored else segment.center
                sa = np.degrees(segment.start_angle)
                ea = np.degrees(segment.stop_angle)
                drawer.arc(center, segment.radius,
                           -sa if mirrored else sa,
                           -ea if mirrored else ea)
            else:
                p = (
                    segment[0], -segment[1]
                ) if mirrored else segment
                if path is None:
                    path = Path2D(p)
                else:
                    path.goto(p)
        path.goto(self.length, 0)
        path.draw(drawer)

    def draw(self, drawer: Drawer,
             do_fillet=False):
        self.process_features(do_fillet)

        for feat in self.bearings:
            pos = feat.position
            da = feat.da
            bold = feat.bold
            pt1 = (pos - bold / 2, da / 2)
            pt2 = (pos + bold / 2, -da / 2)
            drawer.rect(pt1, pt2, 'blue')

        for feat in self.gears:
            pos = feat.position
            da = feat.da
            bold = feat.bold
            pt1 = (pos - bold / 2, da / 2)
            pt2 = (pos + bold / 2, -da / 2)
            drawer.rect(pt1, pt2, 'green')

        # WIPEOUT
        pts_list = map(lambda x: x.pts if isinstance(x, Fillet)
                       else [x], self.chamfered_contour)
        pts_list = sum(pts_list, start=[])
        pts = pts_list + [(pt[0], -pt[1]) for pt in reversed(pts_list)]
        drawer.wipeout(*pts)

        # Contour
        self._draw_half_contour(drawer)
        self._draw_half_contour(drawer, True)

        for feat in self.bushings:
            pos = feat.position
            height = feat.height
            di = self._get_diameter_at(pos)
            width = feat.width
            pt1 = (pos - width / 2, height / 2)
            pt2 = (pos + width / 2, di / 2)
            drawer.rect(pt1, pt2, 'red')
            pt1 = (pos - width / 2, -height / 2)
            pt2 = (pos + width / 2, -di / 2)
            drawer.rect(pt1, pt2, 'red')

        for feat in self.keyways:
            pos = feat.position
            width = feat.width
            length = feat.length
            pt1 = (pos - length / 2, width / 2)
            pt2 = (pos + length / 2, -width / 2)
            drawer.rect(pt1, pt2, 'gold')
