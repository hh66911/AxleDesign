import streamlit as st
from streamlit_cookies_controller import CookieController
from mygraph import m_graph
from modeling import Shaft, PutSide, calc_typeA, calc_typeB, get_feature_name
import pandas as pd
import numpy as np


st.title("【展开式】轴和轴承计算")


def input_params_ui():
    controller = CookieController(key='cookies')
    if 'input_params' not in st.session_state:
        st.session_state.input_params = controller.get('axle_params')
    else:
        input_params = st.session_state.input_params
    INIT_PARAMS = {
        'sig_f': 100., 't_T': 100., 'Pi': 2., 'ni': 500.,
        'eta': [1., 1.], 'i': [1., 1.], 'c': [1, 1, 1],
        'P': [2., 2., 2.], 'n': [500., 500., 500.]
    }
    if input_params is None:
        input_params = INIT_PARAMS
    else:
        para_keys = INIT_PARAMS.keys()
        for k in para_keys:
            if k not in input_params:
                input_params[k] = INIT_PARAMS[k]
    # print(input_params)

    st.header('输入')
    with st.container(border=True):
        st.subheader(body='材料参数')
        input_params['t_T'] = st.number_input(
            r"$\tau_{T}$", value=float(input_params['t_T']))
        input_params['sig_f'] = st.number_input(
            r"$\sigma_{F}$", value=input_params['sig_f'])

        tabs = st.tabs(['分组输入', '一次输入'])
        axles = []
        with tabs[0]:
            for i in range(3):
                axle_name = ['I', 'II', 'III'][i]
                axle_name = rf'轴 $\text{{{axle_name}}}$'
                with st.container(border=True):
                    st.subheader(f'轴 {axle_name} 参数')
                    p = st.number_input(r'$P$ (kW)', value=float(
                        input_params['P'][i]), key=f'P{i}')
                    n = st.number_input(
                        r'$n$ (RPM)', value=input_params['n'][i], key=f'n{i}')
                    c = st.number_input(
                        r'键槽数', value=input_params['c'][i], key=f'c{i}')
                    axles.append((p, n, c))
        with tabs[1]:
            with st.container(border=True):
                p_i = st.number_input(
                    r'$P_i$ (kW)', value=float(input_params['Pi']))
                n_i = st.number_input(r'$n_i$ (RPM)', value=input_params['ni'])
                eta1 = st.number_input(
                    r'$\eta_{\text{I}}$', value=input_params['eta'][0])
                eta2 = st.number_input(
                    r'$\eta_{\text{II}}$', value=input_params['eta'][1])
                i1 = st.number_input(
                    r'$i_{\text{I}}$', value=input_params['i'][0])
                i2 = st.number_input(
                    r'$i_{\text{II}}$', value=input_params['i'][1])
                c1 = st.number_input(
                    r'轴 $\text{I}$ 键槽数', value=input_params['c'][0])
                c2 = st.number_input(
                    r'轴 $\text{II}$ 键槽数', value=input_params['c'][1])
                c3 = st.number_input(
                    r'轴 $\text{III}$ 键槽数', value=input_params['c'][2])
                axles = [
                    (p_i, n_i, c1),
                    (p_i * eta1, n_i / i1, c2),
                    (p_i * eta1 * eta2, n_i / i1 / i2, c3),
                ]

        input_params['P'] = [a[0] for a in axles]
        input_params['n'] = [a[1] for a in axles]
        input_params['c'] = [a[2] for a in axles]
        input_params['Pi'] = axles[0][0]
        input_params['ni'] = axles[0][1]
        eta1 = axles[1][0] / axles[0][0]
        eta2 = axles[2][0] / axles[1][0]
        input_params['eta'] = [eta1, eta2]
        i1 = axles[0][1] / axles[1][1]
        i2 = axles[1][1] / axles[2][1]
        input_params['i'] = [i1, i2]
        # print(input_params)

    if st.button('保存'):
        controller.set('axle_params', input_params)
    return input_params


input_params = input_params_ui()


def show_basics(ps, ns, ts):
    table = pd.DataFrame({
        '功率': [f'{p:.2f} kW' for p in ps],
        '转速': [f'{n:.2f} RPM' for n in ns],
        '扭矩': [f'{t:.2f} Nmm' for t in ts]
    }, index=['I', 'II', 'III'])
    table.index.name = '轴'
    st.table(table)
    return table


def show_diameters(ds):
    table = pd.DataFrame({
        '直径': [f'{d:.2f} mm' for d in ds],
    }, index=['I', 'II', 'III'])
    table.index.name = '轴'
    st.table(table)
    return table


# ---------------------------------------
# region 初算直径
POWER = input_params['P']
SPEED = input_params['n']
TORQUE = [9550e3 * P / n for P, n in zip(POWER, SPEED)]
show_basics(POWER, SPEED, TORQUE)

st.header('初算直径')
TT = input_params['t_T']
diameters = [calc_typeA(p, n, TT, c)
             for p, n, c in zip(POWER, SPEED, input_params['c'])]
show_diameters(diameters)

for i in range(3):
    diameters[i] = st.number_input(
        f'轴 {i+1} 初选直径', value=int(diameters[i] + 0.5))
# endregion 初算直径


# ---------------------------------------
# region 导入齿轮参数
st.header('齿轮参数')
xlsx_file = st.file_uploader('上传齿轮参数文件', type=['xlsx'])
if xlsx_file is None:
    st.write('请上传齿轮参数文件')
    index_names = [
        "模数 (mm)", "法向压力角 (°)", "螺旋角 (度分秒)",
        "分度圆直径 (mm)", "齿根高 (mm)", "齿顶高 (mm)",
        "全齿高 (mm)", "齿顶圆直径 (mm)", "齿根圆直径 (mm)",
        "顶隙 (mm)", "中心距 (mm)", "节圆直径 (mm)",
        "传动比", "齿宽 (mm)", "齿数", "转速 (rpm)",
        "切向力 (N)", '径向力 (N)', '轴向力 (N)',
    ]
    gear_names = ['高速级小齿轮', '高速级大齿轮', '低速级小齿轮', '低速级大齿轮']
    df = pd.DataFrame(columns=gear_names, index=index_names)
    df.index.name = '项目'
else:
    df = pd.read_excel(xlsx_file)
    df.set_index('项目', inplace=True)
df = st.data_editor(df, use_container_width=True)
# endregion 导入齿轮参数


# ---------------------------------------
# region 确定轴尺寸
def get_default_value(session_state, key, default, min_val=1, max_val=None):
    """从session_state获取值，若不存在则返回默认值并裁剪到有效范围"""
    current_value = session_state.get(key, default)
    if max_val is not None:
        return np.clip(current_value, min_val, max_val)
    return current_value


def choose_feature_ui(features, label, key=None, default=None):
    if len(features) == 0:
        st.subheader('暂无可以放置齿轮的位置')
        return None

    def _f(f):
        name = get_feature_name(f, True)
        return f'位于 {f.position} 处的{name}'
    if default is not None and default in features:
        default = features.index(default)
    else:
        default = 0
    return st.selectbox(
        label, features, default, format_func=_f, key=key)


def design_shaft_ui(session_state, shaft, length_range, s_length):
    """创建轴设计UI并返回Shaft对象"""
    features = get_default_value(session_state, 'features', [])
    steps, shoulders, bushings = [], [], []
    placable_feats = []

    def _feat_step_tuple(f, s=None):
        pos = f['pos']
        if 'h' in f:
            r1 = (pos, f['h'], '高度')
            if s is not None:
                return r1, s.add_step(pos, f['h'])
            return r1
        elif 'd' in f:
            r1 = (pos, f['h'], '高度')
            if s is not None:
                return r1, s.add_step(pos, diameter=f['d'])
            return r1
        else:
            raise ValueError

    for f in features:
        if f['type'] == 'bushing':
            feat = f['feat']
            if feat in steps:
                pmt = f'位于 {feat.pos} 处的阶梯'
            elif feat in shoulders:
                pmt = f'位于 {feat.pos} 处的轴环'
            else:
                continue
            bushings.append((pmt, f['h'], f['w'], f['side']))
            placable_feats.append(shaft.add_bushing(
                feat, f['h'], f['w'], PutSide(f['side'])))
            continue

        pos = f['pos']
        match f['type']:
            case 'step':
                ss, pf = _feat_step_tuple(f, shaft)
                steps.append(ss)
                placable_feats.append(pf)
            case 'shoulder':
                shoulders.append((pos, f['h'], f['w']))
                placable_feats.append(shaft.add_shoulder(pos, f['h'], f['w']))

    def strfmt(fvals, decimals=1):
        def _parse(fval):
            if isinstance(fval, str):
                return fval
            return f'{fval:.{decimals}f}'.rstrip('0').rstrip('.')
        if isinstance(fvals, (tuple, list)):
            return (_parse(val) for val in fvals)
        else:
            return _parse(fvals)

    def _feature_conflict(f, fs):
        return any((
            ff['type'] == f['type']
            for ff in fs if ff['pos'] == f['pos']
        ))
    uicols = st.columns(2)
    with uicols[0]:
        pos = st.select_slider(
            '阶梯位置', options=length_range, value=s_length / 2)
        sz = st.number_input('尺寸', value=1)
        stype = st.radio('尺寸类型', ['高度', '直径'], horizontal=True)
        if st.button('添加阶梯', use_container_width=True):
            new = {'type': 'step', 'pos': pos}
            if stype == '高度':
                new['h'] = sz
            else:
                new['d'] = sz
            if not _feature_conflict(new, features):
                features.append(new)
                new, pf = _feat_step_tuple(new, shaft)
                steps.append(new)
        steps_table = ((*strfmt(s[:2]), s[2]) for s in steps)
        steps_table = pd.DataFrame(steps_table, columns=['位置', '尺寸', '类型'])
        steps_table.index.name = '序号'
        steps_table = st.data_editor(steps_table, key='stepdata')
        deleted = steps_table.index[steps_table.isna().sum(1) > 0]
        for didx in deleted:
            target = steps[didx]
            target_feature_dict = {
                'type': 'step', 'pos': target[0]
            }
            if target[2] == '高度':
                target_feature_dict['h'] = target[1]
            else:
                target_feature_dict['d'] = target[1]
            features.remove(target_feature_dict)
        steps = [s for i, s in enumerate(steps) if i not in deleted]
    with uicols[1]:
        pos = st.select_slider(
            '环位置', options=length_range, value=s_length / 2)
        height = st.number_input('高度', value=1, key='shoulder_height')
        width = st.number_input('宽度', value=1, key='shoulder_width')
        if st.button('添加轴肩', use_container_width=True):
            new = {
                'type': 'shoulder',
                'pos': pos, 'h': height, 'w': width
            }
            if not _feature_conflict(new, features):
                features.append()
                shoulders.append((pos, height, width))
                placable_feats.append(shaft.add_shoulder(
                    pos, height, width))
        shoulders_table = (strfmt(s) for s in shoulders)
        shoulders_table = pd.DataFrame(
            shoulders_table, columns=['位置', '高度', '宽度'])
        shoulders_table.index.name = '序号'
        shoulders_table = st.data_editor(shoulders_table, key='shoulderdata')
        deleted = shoulders_table.index[shoulders_table.isna().sum(1) > 0]
        for didx in deleted:
            target = shoulders[didx]
            target_feature_dict = {
                'type': 'shoulder', 'pos': target[0],
                'h': target[1], 'w': target[2]
            }
            features.remove(target_feature_dict)
        shoulders = [s for i, s in enumerate(shoulders) if i not in deleted]

    f = choose_feature_ui(placable_feats, 'bushings')
    height = st.number_input('高度', value=1, key='bushing_height')
    width = st.number_input('宽度', value=1, key='bushing_width')
    dire = st.selectbox('放置方向', ['after', 'before'])
    if st.button('添加套筒', use_container_width=True):
        new = {
            'type': 'bushing', 'feat': f,
            'h': height, 'w': width,
            'side': dire
        }
        if not _feature_conflict(new, features):
            features.append(new)
            shoulders.append((pos, height, width))
            placable_feats.append(shaft.add_shoulder(
                pos, height, width))
    bushings_table = ((s[0], *strfmt(s[1:3]), s[3]) for s in bushings)
    bushings_table = pd.DataFrame(bushings_table, columns=[
        '紧贴', '高度', '宽度', '方向'])
    st.data_editor(bushings_table, key='bushingsdata')

    session_state['features'] = features
    return placable_feats


def create_gear_ui(session_state, gear_idx, feats, bs, fr, ft, fa, ds):
    """创建单个齿轮的UI组件并返回配置"""
    rely_key = f'gear_{gear_idx}_rely'
    plane_key = f"gear_{gear_idx}_plane"
    putside_key = f"gear_{gear_idx}_putside"
    mirrored_key = f"gear_{gear_idx}_mirrored"

    default_rely = get_default_value(session_state, rely_key, None)
    default_plane = get_default_value(session_state, plane_key, 0)
    default_putside = get_default_value(session_state, putside_key, PutSide.BEFORE)
    default_mirrored = get_default_value(session_state, mirrored_key, False)
    default_putside = '之前' if default_putside == PutSide.BEFORE else '之后'
    default_mirrored = '左' if default_mirrored else '右'

    uicols = st.columns([2, 1])
    with uicols[0]:
        f = choose_feature_ui(feats, '齿轮紧贴在', default=default_rely)
        forward = st.select_slider('轴向力朝向', ['左', '右'], default_mirrored)
    with uicols[1]:
        plane = st.radio('轴向力平面', ['z', 'y'], horizontal=True, index=default_plane)
        putside = st.select_slider('放置位置', ['之前', '之后'], default_putside)

    return {
        'rely': f,
        'force_plane': plane,
        'width': float(bs[gear_idx]),
        'diameter': float(ds[gear_idx]),
        'radial_force': float(fr[gear_idx]),
        'tangential_force': float(ft[gear_idx]),
        'axial_force': float(fa[gear_idx]),
        'putside': PutSide.BEFORE if putside == '之前' else PutSide.AFTER,
        'mirrored': forward == '左'
    }


def create_coupling_ui(session_state, length_range, s_length):
    """创建联轴器UI组件并返回配置"""
    default_pos = get_default_value(
        session_state, 'coupling_pos', 1, 1, s_length-1)
    default_force = session_state.get('coupling_force', 0.0)
    default_dir = get_default_value(session_state, 'coupling_dir', 0)

    uicols = st.columns([4, 1, 2])
    with uicols[0]:
        pos = st.select_slider(
            '联轴器位置', options=length_range, value=default_pos)
    with uicols[1]:
        direction = st.radio('压轴力方向', ['y', 'z'], index=default_dir)
    with uicols[2]:
        force = st.number_input('压轴力 (N)', value=default_force)

    return {
        'position': pos,
        'force': force,
        'direction': direction
    }


def create_bearing_ui(session_state, length_range, s_length):
    """创建轴承UI组件并返回配置"""
    default_b1 = get_default_value(
        session_state, 'bearing1_pos', 1, 1, s_length-1)
    default_b2 = get_default_value(
        session_state, 'bearing2_pos', s_length-1, 1, s_length-1)

    col1, col2 = st.columns(2)
    with col1:
        bear1 = st.select_slider(
            '轴承 1 位置 (mm)', options=length_range, value=default_b1)
    with col2:
        bear2 = st.select_slider(
            '轴承 2 位置 (mm)', options=length_range, value=default_b2)

    return {'bearing1': bear1, 'bearing2': bear2}


def make_shaft_ui(d_init: float, gears: list[int]):
    """创建轴设计UI并返回Shaft对象"""
    session_state = st.session_state.get('sel_axle', {})

    # 初始化参数
    cols = df.columns[gears]
    bs = df.loc['齿宽 (mm)', cols].astype(float).values
    total_width = sum(bs)
    st.write(f'轴上齿轮宽度总和：{total_width} mm')

    s_length = st.number_input(
        '草图长度 (mm)', value=int(get_default_value(
            session_state, 'slen', total_width + 200)))
    length_range = range(1, s_length)

    # 创建轴对象
    shaft = Shaft(d_init)

    # 处理轴外形
    placable_feats = design_shaft_ui(session_state, shaft,
                    length_range, s_length)
    shaft.end_at(s_length)
    # print(session_state['features'])

    # 处理齿轮配置
    gear_params = []
    for gear_idx in gears:
        params = create_gear_ui(session_state, gear_idx,
                                placable_feats, bs,
                                df.loc['径向力 (N)', cols].values,
                                df.loc['切向力 (N)', cols].values,
                                df.loc['轴向力 (N)', cols].values,
                                df.loc['分度圆直径 (mm)', cols].values)
        gear_params.append(params)
        session_state[f"gear_{gear_idx}_rely"] = params['rely']
        session_state[f"gear_{gear_idx}_plane"] = [
            'z', 'y'].index(params['force_plane'])
        session_state[f"gear_{gear_idx}_putside"] = params['putside'].value
        session_state[f"gear_{gear_idx}_mirrored"] = params['mirrored']

    # 添加齿轮到轴
    for params in gear_params:
        if params['rely'] is None:
            continue
        try:
            shaft.add_gear(params['rely'],
                       params['diameter'],
                       params['width'],
                       params['radial_force'],
                       params['tangential_force'],
                       -params['axial_force'] if params['mirrored'] else params['axial_force'],
                       params['force_plane'],
                       params['putside']
                       )
        except TypeError:
            st.rerun()

    # 处理联轴器
    if st.checkbox('是否有联轴器', value=session_state.get('has_coupling', False)):
        coupling_params = create_coupling_ui(
            session_state, length_range, s_length)
        direction = coupling_params['direction']
        force = coupling_params['force']
        shaft.add_coupling(coupling_params['position'],
                        force if direction == 'y' else 0,
                        force if direction == 'z' else 0)
        session_state['coupling_pos'] = coupling_params['position']
        session_state['coupling_force'] = force
        session_state['coupling_dir'] = ['y', 'z'].index(direction)

    # 处理轴承
    bearing_params = create_bearing_ui(session_state, length_range, s_length)
    
    session_state['bearing1_pos'] = bearing_params['bearing1']
    session_state['bearing2_pos'] = bearing_params['bearing2']

    # 保存状态
    st.session_state.sel_axle = session_state

    return shaft


if __name__ == '__main__':
    sel_axle = st.radio('选择轴', ['I', 'II', 'III'])
    if 'axle_data' not in st.session_state:
        st.session_state.axle_data = {}

    st.header(rf'轴 $\text{{{sel_axle}}}$')

    if sel_axle in st.session_state.axle_data:
        st.session_state.sel_axle = st.session_state.axle_data[sel_axle]
    match sel_axle:
        case 'I':
            shaft = make_shaft_ui(diameters[0], [0])
        case 'II':
            shaft = make_shaft_ui(diameters[1], [1, 2])
        case 'III':
            shaft = make_shaft_ui(diameters[2], [3])
        case _:
            raise ValueError('轴选择错误')
    st.session_state.axle_data[sel_axle] = st.session_state.sel_axle

    shaft_plot = shaft.plot()
    st.pyplot(shaft_plot)

    forces = shaft.forces
    bends = shaft.bends
    st.write('xoy 平面受力：', *[f'{f[1]: .2f}，' for f in forces['y']])
    st.write('xoz 平面受力：', *[f'{f[1]: .2f}，' for f in forces['z']])
    st.write('xoy 平面弯矩：', *[f'{f[1]: .2f}，' for f in bends['y']])
    st.write('xoz 平面弯矩：', *[f'{f[1]: .2f}，' for f in bends['z']])

    if st.button('计算受力'):
        passed, byfig, bzfig, tfig, sigma_fig = calc_typeB(
            shaft, input_params['sig_f'])
        st.subheader('xoy 平面弯矩')
        st.pyplot(byfig)
        st.subheader('xoz 平面弯矩')
        st.pyplot(bzfig)
        st.subheader('xoy 平面转矩')
        st.pyplot(tfig)
        st.subheader('应力')
        st.pyplot(sigma_fig)
        # endregion 确定轴尺寸
