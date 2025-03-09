import streamlit as st
from streamlit_cookies_controller import CookieController
from mygraph import m_graph
from modeling import Shaft, PutSide, calc_typeA, calc_typeB
import pandas as pd
import sys
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


def create_gear_ui(session_state, gear_idx, length_range, s_length, bs, fr, ft, fa, ds):
    """创建单个齿轮的UI组件并返回配置"""
    pos_key = f"gear_{gear_idx}_pos"
    plane_key = f"gear_{gear_idx}_plane"

    default_pos = get_default_value(
        session_state, pos_key, round(sum(bs[:gear_idx])), 1, s_length)
    default_plane = get_default_value(session_state, plane_key, 0)

    uicols = st.columns([2, 1])
    with uicols[0]:
        pos = st.select_slider(f'齿轮 {gear_idx + 1} 位置 (mm)',
                               options=length_range,
                               value=default_pos)
    with uicols[1]:
        plane = st.radio(f'轴向力平面',
                         ['z', 'y'],
                         index=default_plane)

    return {
        'position': pos,
        'force_plane': plane,
        'width': float(bs[gear_idx]),
        'diameter': float(ds[gear_idx]),
        'radial_force': float(fr[gear_idx]),
        'tangential_force': float(ft[gear_idx]),
        'axial_force': float(fa[gear_idx])
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
    shaft.end_at(s_length)
    
    # 处理轴外形
    features = get_default_value(session_state, 'features', [])
    steps, shoulders = [], []
    placable_feats = []
    for f in features:
        match f['type']:
            case 'step':
                if 'h' in f:
                    steps.append((f['pos'], f['h'], '高度'))
                    placable_feats.append(shaft.add_step(f['pos'], f['h']))
                else:
                    steps.append((f['pos'], f['d'], '直径'))
                    placable_feats.append(shaft.add_step(f['pos'], diameter=f['d']))
            case 'shoulder':
                shoulders.append((f['pos'], f['h'], f['w']))
                placable_feats.append(shaft.add_shoulder(f['pos'], f['h'], f['w']))
    def strfmt(*v):
        return (str(vv) if vv.is_integer() else f'{vv: .1f}' for vv in v)
    steps_data = zip(*((*strfmt(s[:1]), s[2]) for s in steps))
    shoulders_data = zip(*(strfmt(s) for s in shoulders))
    steps_data = pd.DataFrame(steps_data, columns=['位置', '尺寸', '类型'])
    steps_data.index.name = '序号'
    shoulders_data = pd.DataFrame(shoulders_data, columns=['位置', '高度', '宽度'])
    steps_data.index.name = '序号'
    uicols = st.columns(2)
    with uicols[0]:
        st.data_editor(steps_data, key='stepdata')
        pos = st.select_slider(
            '阶梯位置', options=length_range, value=s_length / 2)
        sz = st.number_input('尺寸', value=1)
        stype = st.radio('尺寸类型', ['高度', '直径'], horizontal=True)
        steps_data.add((pos, sz, stype))
    with uicols[1]:
        st.data_editor(steps_data, key='shoulderdata')
        pos = st.select_slider(
            '环位置', options=length_range, value=s_length / 2)
        height = st.number_input('高度', value=1)
        width = st.number_input('宽度', value=1)
        shoulders_data.add((pos, height, width))
    # 解析steps_data和shoulders_data，填充到features内
    features = []
    for idx, row in steps_data.iterrows():
        pos, size, stype = row['位置'], row['尺寸'], row['类型']
        if stype == '高度':
            features.append({'type': 'step', 'pos': pos, 'h': size})
        else:
            features.append({'type': 'step', 'pos': pos, 'd': size})

    for idx, row in shoulders_data.iterrows():
        pos, height, width = row['位置'], row['高度'], row['宽度']
        features.append({'type': 'shoulder', 'pos': pos, 'h': height, 'w': width})

    session_state['features'] = features

    # 处理齿轮配置
    gear_params = []
    for gear_idx in gears:
        params = create_gear_ui(session_state, gear_idx,
                                length_range, s_length,
                                bs,
                                df.loc['径向力 (N)', cols].values,
                                df.loc['切向力 (N)', cols].values,
                                df.loc['轴向力 (N)', cols].values,
                                df.loc['分度圆直径 (mm)', cols].values)
        gear_params.append(params)
        session_state[f"gear_{gear_idx}_pos"] = params['position']
        session_state[f"gear_{gear_idx}_plane"] = [
            'z', 'y'].index(params['force_plane'])

    # 添加齿轮到轴
    for params in gear_params:
        shaft.add_gear(params['position'],
                       params['width'],
                       params['diameter'],
                       params['radial_force'],
                       params['tangential_force'],
                       params['axial_force'],
                       params['force_plane'])

    # 处理联轴器
    if st.checkbox('是否有联轴器', value=session_state.get('has_coupling', False)):
        coupling_params = create_coupling_ui(
            session_state, length_range, s_length)
        direction = coupling_params['direction']
        force = coupling_params['force']
        shaft.fix_twist(coupling_params['position'],
                        force if direction == 'y' else 0,
                        force if direction == 'z' else 0)
        session_state['coupling_pos'] = coupling_params['position']
        session_state['coupling_force'] = force
        session_state['coupling_dir'] = ['y', 'z'].index(direction)

    # 处理轴承
    bearing_params = create_bearing_ui(session_state, length_range, s_length)
    shaft.fix_bearing(bearing_params['bearing1'], bearing_params['bearing2'])
    session_state['bearing1_pos'] = bearing_params['bearing1']
    session_state['bearing2_pos'] = bearing_params['bearing2']

    # 保存状态
    st.session_state.sel_axle = session_state

    return shaft


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
