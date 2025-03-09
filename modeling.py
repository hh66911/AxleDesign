from enum import Enum
import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy import ndarray
from mygraph import m_graph, t_graph

from drawer import Drawer, Path2D
from matplotlib import pyplot as plt


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


class Bearing:
    size_df = pd.read_excel(r"D:\BaiduSyncdisk\球轴承尺寸.xlsx")

    def __init__(self, code):
        self.code = code
        if code.startswith('16'):
            bearing_type = '6'
        else:
            bearing_type = code[0]
        match bearing_type:
            case '7':
                name = '角接触球轴承'
                if code.endswith('AC'):
                    angle = 25
                elif code.endswith('B'):
                    raise ValueError('不支持B型角接触球轴承')
                elif code.endswith('C'):
                    angle = 15
                    code = code[:-1]
                else:
                    raise ValueError('角度信息缺失')
            case '6':
                name = '深沟球轴承'
                angle = None
            case _:
                raise ValueError('不支持的轴承类型')

        code = '7' + self.code[1:]
        codes = Bearing.size_df[["a1", 'a2']]
        idx = codes.stack()[codes.stack() == code]
        idx = list(set(i[0] for i in idx.index))
        if len(idx) > 1:
            raise ValueError(f'错误的型号，多个值找到：{idx}')
        elif len(idx) == 0:
            raise ValueError(f'错误的型号，未找到：{code}')
        size_data = Bearing.size_df.loc[idx[0], :]
        (
            self.d, self.da,
            self.b, self.c,
            self.c1
        ) = size_data[['d', 'D', 'B', 'c', 'c1']]

        self.name = name
        self.angle = np.deg2rad(angle) if angle is not None else 0

        length = (self.da - self.d) / 2
        ball_radius = length / 4
        self.inner_thick = length / 2 - ball_radius * np.cos(np.pi / 3)

    def check_attach(self, height):
        if height > self.inner_thick * 4 / 5:
            warnings.warn(
                f'高度 {height} 超过了轴承内圈厚度的 4/5', BadDesignWarning)
            return False
        return True

    def __repr__(self):
        return f'Bearing({self.code})'

    def __str__(self):
        return f'{self.name} {self.code} {self.d}x{self.da}x{self.b}'

    @staticmethod
    def parse_code_size(digits):
        # 初始化变量
        A = B = C = D = None

        def get_size(d1, d2):
            d = int(d1) * 10 + int(d2)
            if d <= 3:
                d = [10, 12, 15, 17][d]
            else:
                d = d * 5
            return d

        # 匹配五位数字开头的情况：ABCDD
        five_digit_pattern = re.match(r'^(\d)(\d)(\d)(\d)$', digits)
        if five_digit_pattern:
            A, B, C, D1, D2 = five_digit_pattern.groups()
            A, B, C = int(A), int(B), int(C)
            D = get_size(D1, D2)
        # 匹配四位数开头的情况：ABDD
        else:
            four_digit_pattern = re.match(r'^(\d)(\d)(\d)$', digits)
            if four_digit_pattern:
                A, C, D1, D2 = four_digit_pattern.groups()
                A, C = int(A), int(C)
                B = 0 if C != 0 else 1
                D = get_size(D1, D2)
            # 匹配三位数开头并跟随一个斜杠和数值的情况：ABC/D
            else:
                three_digit_with_slash_pattern = re.match(
                    r'^(\d)(\d)/([\d.]+)$', digits)
                if three_digit_with_slash_pattern:
                    A, B, C, D = three_digit_with_slash_pattern.groups()
                    A, B, C = int(A), int(B), int(C)
                    D = float(D)

        # 返回结果
        return A, B, C, D


class KeywayType(Enum):
    A = 'A'  # 半圆键
    B = 'B'  # 方键
    C = 'C'  # 单半圆键


class Keyway:
    SIZE_LOOKUP_TABLE = [
        (6, 8), (8, 10), (10, 12), (12, 17),
        (17, 22), (22, 30), (30, 38), (38, 44),
        (44, 50), (50, 58), (58, 65), (65, 75),
        (75, 85), (85, 95), (95, 110), (110, 130),
        (130, 150), (150, 170), (170, 200),
        (200, 230), (230, 260), (260, 290),
        (290, 330), (330, 380), (380, 440), (440, 500)
    ]
    BOLD_TABLE = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16,
                  18, 20, 22, 25, 28, 32, 36, 40, 45,
                  50, 56, 63, 70, 80, 90, 100]  # H8 配合
    ALL_LENGTH = [6, 8, 10, 12, 14, 16, 18, 20, 22,
                  25, 28, 32, 36, 40, 45, 50, 56, 63,
                  70, 80, 90, 100, 110, 125, 140, 160,
                  180, 200, 220, 250, 280, 320, 360,
                  400, 450, 500]
    LENGTH_LOOKUP_RANGE = [
        (0, 8), (0, 13), (1, 15), (2, 17), (4, 19), (6, 21),
        (8, 23), (10, 25), (12, 26), (14, 27), (15, 28),
        (16, 29), (17, 30), (18, 31), (19, 32), (20, 33),
        (20, 34), (21, 34), (22, 35), (23, 36), (24, 36),
        (25, 36), (26, 36), (27, 36), (28, 36)
    ]
    HEIGHT_TABLE = [2, 3, 4, 5, 6, 7, 8, 8, 9, 10,
                    11, 12, 14, 14, 16, 18, 20, 22,
                    25, 28, 32, 32, 36, 40, 45, 50]  # H11 或 H8 配合
    T_SHAFT_TABLE = [
        1.2, 1.8, 2.5, 3.0, 3.5, 4.0, 5.0, 5.0, 5.5, 6.0, 7.0, 7.5, 9.0, 9.0, 10.0,
        11.0, 12.0, 13.0, 15.0, 17.0, 20.0, 20.0, 22.0, 25.0, 28.0, 31.0
    ]
    T_HUB_TABLE = [
        1.0, 1.4, 1.8, 2.3, 2.8, 3.3, 3.3, 3.3, 3.8, 4.3, 4.4, 4.9, 5.4, 5.4, 6.4,
        7.4, 8.4, 9.4, 10.4, 11.4, 12.4, 12.4, 14.4, 15.4, 17.4, 19.5
    ]

    def __init__(self, length, diameter, ktype=KeywayType.A):
        self.length = length
        self.r = diameter / 2

        if length > diameter * 1.6:
            raise ValueError("长度超过了给定直径的最大允许长度。")

        for idx, (min_d, max_d) in enumerate(Keyway.SIZE_LOOKUP_TABLE):
            if min_d <= diameter < max_d:
                self.width = Keyway.BOLD_TABLE[idx]
                self.height = Keyway.HEIGHT_TABLE[idx]
                self.t_shaft = Keyway.T_SHAFT_TABLE[idx]
                self.t_hub = Keyway.T_HUB_TABLE[idx]
                start, end = Keyway.LENGTH_LOOKUP_RANGE[idx]
                valid_lengths = Keyway.ALL_LENGTH[start:end]
                if length not in valid_lengths:
                    raise ValueError(f"长度 {length} 不在有效范围 {valid_lengths} 内。")
                break
        else:
            raise ValueError(f"直径 {diameter} 超出了键槽尺寸查找的范围。")

        self.type = ktype
        self.left_top = (-self.width / 2, self.length / 2)
        self.right_bottom = (self.width / 2, -self.length / 2)


@dataclass
class Fillet:
    radius: float
    center: tuple
    pts: list
    start_angle: float
    stop_angle: float


@dataclass
class _FixedFeature:
    position: float


@dataclass
class _StepFeature(_FixedFeature):
    size: float
    is_abs: bool

    def __iter__(self):
        return iter((self.position, self.size, self.is_abs))


@dataclass
class _ShoulderFeature(_FixedFeature):
    width: float


@dataclass
class _BushingFeature(_FixedFeature):
    height: float
    width: float


@dataclass
class _KeywayFeature(_FixedFeature):
    inst: Keyway
    forward: bool


@dataclass
class _GearFeature(_FixedFeature):
    bold: float
    da: float


@dataclass
class _BearingFeature(_FixedFeature):
    inst: Bearing
    forward: bool


def get_feature_name(feat, chinese=False):
    """
        根据特征对象的类型返回相应的特征名称。
    """
    if isinstance(feat, _StepFeature):
        return '阶梯' if chinese else "Step"
    elif isinstance(feat, _ShoulderFeature):
        return "轴肩" if chinese else "Shoulder"
    elif isinstance(feat, _BushingFeature):
        return "轴套" if chinese else "Bushing"
    elif isinstance(feat, _KeywayFeature):
        return "键槽" if chinese else "Keyway"
    elif isinstance(feat, _GearFeature):
        return "齿轮" if chinese else "Gear"
    elif isinstance(feat, _BearingFeature):
        return "轴承" if chinese else "Bearing"
    else:
        raise TypeError('未知特征名')


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

        self.coupling_pos = None
        self.forces = {'y': [], 'z': []}    # 受力 [位置, 数量]
        self.bends = {'y': [], 'z': []}    # 受弯矩 [位置, 数量]
        self.twists = []                    # 受转矩

        self.gears: list[_GearFeature] = []
        self.keyways: list[_KeywayFeature] = []
        self.bearings: list[_BearingFeature] = []
        self.bushings: list[_BushingFeature] = []

    def add_force(self, pos, val_y=+0., val_z=+0.):
        '''
        添加受力

        Args:
            pos: float
                位置
            val_y: float
                y方向数量（正方向向下）
            val_z: float
                z方向数量（正方向向下）
        '''
        if abs(val_y) > 0.1:
            self.forces['y'].append((pos, val_y))
        if abs(val_z) > 0.1:
            self.forces['z'].append((pos, val_z))

    def add_bend(self, pos, val_y=+0., val_z=+0.):
        '''
        添加弯矩

        Args:
            pos: float
                位置
            val_y: float
                y方向数量（正方向为逆时针）
            val_z: float
                z方向数量（正方向为逆时针）
        '''
        if abs(val_y) > 0.1:
            self.bends['y'].append((pos, val_y))
        if abs(val_z) > 0.1:
            self.bends['z'].append((pos, val_z))

    def add_twist(self, pos, val):
        '''
        添加扭矩

        Args:
            pos: float
                位置
            val: float
                数量
        '''
        if val != 0:
            self.twists.append((pos, val))

    def add_coupling(self, pos, fqy=0, fqz=0):
        twist_sum = sum(map(lambda x: x[1], self.twists))
        self.add_twist(pos, -twist_sum)
        self.coupling_pos = pos
        self.add_force(pos, fqy, fqz)

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
        result = _BushingFeature(pos, height, width)
        self.bushings.append(result)
        return result

    def add_keyway(self, feat, length, ktype=KeywayType.A, forward=True):
        if isinstance(feat, (_GearFeature,)):
            pos = feat.position
        else:
            raise ValueError("不支持的特征类型。")
        self.keyways.append(_KeywayFeature(
            pos, Keyway(
                length, self._get_diameter_at(pos, False), ktype
            ), forward
        ))
        return self.keyways[-1]

    def add_gear(self, pos_or_feat, da, bold, fr, ft, fa, bend_plane,
                 put_side=PutSide.BEFORE):
        if not isinstance(pos_or_feat, float):
            position = pos_or_feat.position + _get_offset(
                pos_or_feat, bold / 2, put_side)
        else:
            position = pos_or_feat

        self.add_force(position, ft, fr)
        if bend_plane == 'z':
            self.add_bend(position, 0, -fa * da / 2)
        else:
            self.add_bend(position, -fa * da / 2, 0)
        self.add_twist(position, ft * da / 2)

        self.gears.append(_GearFeature(position, bold, da))
        return self.gears[-1]

    def add_bearing(self, feat, code,
                    forward=True, put_side=PutSide.BEFORE):
        if not isinstance(feat, _BushingFeature):
            raise NotImplementedError(f"不支持的特征类型: {type(feat)}")
        b = Bearing(code)
        pos = feat.position + _get_offset(
            feat, b.b / 2, put_side)
        self.bearings.append(_BearingFeature(pos, b, forward))
        return self.bearings[-1]

    def solve_bearings(self):
        if len(self.bearings) != 2:
            raise ValueError("轴承数量不正确。")
        b1, b2 = self.bearings[0], self.bearings[1]
        p1, p2 = b1.position, b2.position
        d1, d2 = b1.inst.d, b2.inst.d
        if d1 != self._get_diameter_at(p1, True):
            raise ValueError("轴承1的直径不匹配。")
        if d2 != self._get_diameter_at(p2, True):
            raise ValueError("轴承2的直径不匹配。")
        # 计算轴承 y 平面的力
        f_sum = sum(map(lambda x: x[1], self.forces['y']))
        m_sum = sum(map(lambda x: x[0] * x[1], self.forces['y'])) - \
            sum(map(lambda x: x[1], self.bends['y']))
        f1y = (m_sum - p2 * f_sum) / (p1 - p2)
        f2y = f_sum - f1y
        # 计算轴承 z 平面的力
        f_sum = sum(map(lambda x: x[1], self.forces['z']))
        m_sum = sum(map(lambda x: x[0] * x[1], self.forces['z'])) - \
            sum(map(lambda x: x[1], self.bends['z']))
        f1z = (m_sum - p2 * f_sum) / (p1 - p2)
        f2z = f_sum - f1z
        # 添加轴承力
        self.add_force(p1, -f1y, -f1z)
        self.add_force(p2, -f2y, -f2z)

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

    def plot(self, figsize=(12, 6), do_fillet=False):
        self.process_features(do_fillet)

        drawer = Drawer(figsize)

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
            kw = feat.inst
            width = kw.width
            length = kw.length
            pt1 = (pos - length / 2, width / 2)
            pt2 = (pos + length / 2, -width / 2)
            drawer.rect(pt1, pt2, 'gold')

        if self.coupling_pos is not None:
            d = self._get_diameter_at(self.coupling_pos)
            drawer.line((self.coupling_pos, -d),
                        (self.coupling_pos, d), 'brown')

        # 绘制力
        def draw_force(dir, val):
            if dir == 'y':
                draw_dir = (1 if val < 0 else -1) * self.length / 20
                drawer.ax.arrow(pos, 0, draw_dir, draw_dir, head_width=2,
                                head_length=4, fc='r', ec='r')
            elif dir == 'z':
                draw_dir = (1 if val < 0 else -1) * self.length / 20
                drawer.ax.arrow(pos, 0, 0, draw_dir, head_width=2,
                                head_length=4, fc='r', ec='r')

        def draw_bend(plane: str, val):
            # 生成圆弧的点
            if val > 0:
                theta = np.linspace(-np.pi / 2, 0, 100)
            else:
                theta = np.linspace(np.pi, np.pi / 2, 100)
            if plane == 'y':  # xoy 平面
                x = pos + 10 * np.cos(theta)
                # 为了区分 y 和 z 方向，将 z 方向的曲线在 y 轴上偏移
                y_offset = -12 if val > 0 else 10
                y = y_offset + 10 * np.sin(theta) / 2
                hw, hl = 2, 2
            elif plane == 'z':  # xoz 平面
                x = pos + 10 * np.cos(theta)
                y = 10 * np.sin(theta)
                hw, hl = 3, 4
            else:
                raise ValueError('未知的平面')
            if val > 0:
                arror_d = (0, 1)
            else:
                arror_d = (1, 0)
            drawer.ax.plot(x, y, color='r', lw=2)
            arrow_start = (x[-1], y[-1])
            drawer.ax.arrow(
                arrow_start[0], arrow_start[1], arror_d[0], arror_d[1],
                head_width=hw, head_length=hl, fc='r', ec='r'
            )

        for pos, val in self.forces['y']:
            draw_force('y', val)
        for pos, val in self.forces['z']:
            draw_force('z', val)
        for pos, val in self.bends['y']:
            draw_bend('y', val)
        for pos, val in self.bends['z']:
            draw_bend('z', val)

        return drawer.fig

    def get_sampled_W(self, n=5000):
        """
        返回从整个轴体均匀采样的n个点的直径
        """
        self.process_features(False, False)
        # 确定轴体的总长度
        total_length = self.contour[-1][0]
        positions = np.linspace(0, total_length, n)

        diameters = np.zeros(n)
        W, Wt = np.zeros(n), np.zeros(n)
        i_contour = 0  # 初始化contour的索引

        def _by_contour(p, i):
            while i < len(self.contour) - 1:
                x0, d0 = self.contour[i]
                x1, _ = self.contour[i + 1]
                if x0 <= p < x1:
                    return d0, i
                elif p >= x1:
                    i += 1  # 移动到下一个线段
                else:
                    raise ValueError('未排序的数据')
            # 如果pos超出了contour的范围，使用最后一个点的直径
            return self.contour[-1][1], i

        def _calc_W(p, d):
            W = np.pi * d**3 / 32
            Wt = W * 2
            k = None
            for kw in self.keyways:
                left = kw.position - kw.inst.length / 2
                right = kw.position + kw.inst.length / 2
                if left <= p < right:
                    k = kw.inst
                    break
            if k is not None:
                b, t = k.width, k.t_shaft
                adj_val = b * t * (d - t)**2 / 2 / d
                W -= adj_val
                Wt -= adj_val
            return W, Wt

        for j in range(n):
            pos = positions[j]
            d, i_contour = _by_contour(pos, i_contour)
            diameters[j] = d
            W[j], Wt[j] = _calc_W(pos, d)

        return diameters, W, Wt

    def get_sampled_pos(self, n=5000):
        return np.linspace(0, self.length, n)


def calc_typeA(P: float, n: float, tT: float, c: int):
    '''
    按**扭转剪切强度**计算轴的直径

    Args:
        P: float
            功率
        n: float
            转速
        tT: float
            扭矩
        c: int
            键槽数量

    Returns:
        d: float
            轴的直径
    '''
    d = (9.55e6 / 0.2 / tT * P / n) ** (1/3)
    if d <= 100:
        if c == 1:
            d = d * 1.06
        elif c == 2:
            d = d * 1.14
        else:
            d = None
    else:
        d = d * (1 + 0.03 * c)
    return d


def calc_typeB(s: Shaft, sigT: float, alpha=0.6, sample_N=5000):
    '''
    按**弯扭组合**计算轴的直径

    Args:
        shaft_data: dict
            轴的数据
        alpha: float
            安全系数：当扭转切应力为静应力时，取 0.3；
            当扭转切应力为脉动循环变应力时，取 0.6
            当扭转切应力为对称循环变应力时，取 1.0
    '''
    # 计算弯矩
    length = s.length
    (_, bend_y), byfig = m_graph(s.forces['y'], s.bends['y'], length, sample_N)
    (_, bend_z), bzfig = m_graph(s.forces['z'], s.bends['z'], length, sample_N)
    bend_total = np.sqrt(bend_y**2 + bend_z**2)
    # 计算扭矩
    (_, twist), tfig = t_graph(s.twists, length, sample_N)
    # 计算弯扭组合应力
    _, W, Wt = s.get_sampled_W(sample_N)  # 计算截面模量
    sigma_bend = bend_total / W  # MPa
    sigma_t = alpha * twist / Wt  # MPa
    sigma_total = (sigma_bend**2 + sigma_t**2)**0.5
    sigma_fig = plt.figure(figsize=(16, 6))
    pos = np.linspace(0, length, sample_N)
    plt.plot(pos, sigma_bend, label='Bending Stress')
    plt.plot(pos, sigma_t, label='Twisting Stress')
    plt.plot(pos, sigma_total, label='Total Stress')
    max_pos, max_stress = pos[np.argmax(sigma_total)], np.max(sigma_total)
    plt.text(max_pos, max_stress,
             f'{max_stress:.2f}', fontsize=14, color='red')
    plt.plot([0, length], [sigT, sigT], 'r--', label='Allowable Stress')
    plt.xlabel('Position [mm]')
    plt.ylabel('Stress [MPa]')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    error = sigma_total > sigT
    return not (np.any(error)), byfig, bzfig, tfig, sigma_fig
