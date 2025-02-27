import streamlit as st
from streamlit_cookies_controller import CookieController
from mygraph import m_graph
from modeling import Shaft, calc_typeA, calc_typeB
import pandas as pd
import sys
import numpy as np


st.title("【展开式】轴和轴承计算")

def input_params_ui():
    if 'run_count' not in st.session_state:
        st.session_state.run_count = 0
    st.session_state.run_count += 1
    
    controller = CookieController(key='cookies')
    input_params = controller.get('axle_params')
    INIT_PARAMS = {
        't_T': 100., 'Pi': 2., 'ni': 500.,
        'eta': [1., 1.], 'i': [1., 1.], 'c': [1, 1, 1],
        'P': [2., 2., 2.], 'n': [500., 500., 500.],
    }
    if input_params is None:
        input_params = INIT_PARAMS
    else:
        para_keys = INIT_PARAMS.keys()
        for k in para_keys:
            if k not in input_params:
                input_params[k] = INIT_PARAMS[k]
    print(input_params)
    
    st.header('输入')
    with st.container(border=True):
        st.subheader(body='材料参数')
        input_params['t_T'] = st.number_input(r"$\tau_{T}$", value=input_params['t_T'])
        
        form_type = st.radio('输入格式', ['分组输入', '一次输入'])
        axles = []
        if form_type == '分组输入':
            for i in range(3):
                axle_name = ['I', 'II', 'III'][i]
                axle_name = rf'轴 $\text{{{axle_name}}}$'
                with st.container(border=True):
                    st.subheader(f'轴 {axle_name} 参数')
                    p = st.number_input(r'$P$ (kW)', value=input_params['P'][i], key=f'P{i}')
                    n = st.number_input(r'$n$ (RPM)', value=input_params['n'][i], key=f'n{i}')
                    c = st.number_input(r'键槽数', value=input_params['c'][i], key=f'c{i}')
                    axles.append((p, n, c))
        else:
            with st.container(border=True):
                p_i = st.number_input(r'$P_i$ (kW)', value=input_params['Pi'])
                n_i = st.number_input(r'$n_i$ (RPM)', value=input_params['ni'])
                eta1 = st.number_input(r'$\eta_{\text{I}}$', value=input_params['eta'][0])
                eta2 = st.number_input(r'$\eta_{\text{II}}$', value=input_params['eta'][1])
                i1 = st.number_input(r'$i_{\text{I}}$', value=input_params['i'][0])
                i2 = st.number_input(r'$i_{\text{II}}$', value=input_params['i'][1])
                c1 = st.number_input(r'轴 $\text{I}$ 键槽数', value=input_params['c'][0])
                c2 = st.number_input(r'轴 $\text{II}$ 键槽数', value=input_params['c'][1])
                c3 = st.number_input(r'轴 $\text{III}$ 键槽数', value=input_params['c'][2])
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


#---------------------------------------
# region 初算直径
POWER = input_params['P']
SPEED = input_params['n']
TORQUE = [9550e3 * P / n for P, n in zip(POWER, SPEED)]
show_basics(POWER, SPEED, TORQUE)

st.header('初算直径')
TT = input_params['t_T']
diameters = [calc_typeA(p, n, TT, c) for p, n, c in zip(POWER, SPEED, input_params['c'])]
show_diameters(diameters)

for i in range(3):
    diameters[i] = st.number_input(f'轴 {i+1} 初选直径', value=int(diameters[i] + 0.5))
# endregion 初算直径


#---------------------------------------
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


#---------------------------------------
# region 确定轴尺寸
def make_shaft_ui(d_init: int, gears: list[int]):
    sel_axle = dict()
    if 'sel_axle' in st.session_state:
        sel_axle = st.session_state.sel_axle
        
    cols = df.columns[gears]
    bs = df.loc['齿宽 (mm)', cols].values
    bs = map(float, bs)
    st.write(f'轴上齿轮宽度总和：{sum(bs)} mm')
    if 'slen' in sel_axle:
        s_length_def = sel_axle['slen']
    else:
        s_length_def = sum(bs) + 200
    s_length = st.number_input('草图长度 (mm)', value=s_length_def)
    length_range = range(0, s_length + 2)
    
    bs = df.loc['齿宽 (mm)', :].values
    bs = list(map(float, bs))
    ft = df.loc['切向力 (N)', :].values
    fr = df.loc['径向力 (N)', :].values
    fa = df.loc['轴向力 (N)', :].values
    ds = df.loc['分度圆直径 (mm)', :].values
    s = Shaft(d_init, s_length)
        
    for i in gears:
        if f'fa{i}' in sel_axle:
            gear_pos_default = sel_axle[i]
            gear_pos_default = np.clip(gear_pos_default, 0, s_length)
            fa_plane_def = sel_axle[f'fa{i}']
        else:
            gear_pos_default = round(sum(bs[:i]))
            sel_axle[i] = gear_pos_default
            fa_plane_def = 0
        gear_pos = st.select_slider(f'齿轮 {i + 1} 位置 (mm)',
            length_range, value=gear_pos_default)
        fa_plane = st.radio(f'齿轮 {i + 1} 轴向力平面', ['z', 'y'], index=fa_plane_def)
        s.add_gear(gear_pos,
            float(bs[i]), float(ds[i]),
            float(fr[i]), float(ft[i]), float(fa[i]),
            fa_plane)
        sel_axle[i] = gear_pos
        sel_axle[f'fa{i}'] = ['z', 'y'].index(fa_plane)
    
    is_IO = st.checkbox('是否有联轴器', value=sel_axlev['IO'] if 'IO' in sel_axle else False)
    if is_IO:
        if 'fqdir' in sel_axle:
            coupling_def = sel_axle['c']
            coupling_def = np.clip(coupling_def, 0, s_length)
            fq_def = sel_axle['fq']
            fqdir_def = sel_axle['fqdir']
        else:
            coupling_def = 0
            fq_def = 0
            fqdir_def = 0
        coupling = st.select_slider('联轴器位置', length_range, coupling_def)
        fqdir = st.radio('压轴力方向', ['y', 'z'], index=fqdir_def)
        fq = st.number_input('压轴力 (N)', value=fq_def)
        if fqdir == 'y':
            s.fix_twist(coupling, fq, 0)
        else:
            s.fix_twist(coupling, 0, fq)
        sel_axle['c'] = coupling
        sel_axle['fq'] = fq
        sel_axle['fqdir'] = ['y', 'z'].index(fqdir)
        
    if 'b' in sel_axle:
        b1_def = sel_axle['b'][0]
        b2_def = sel_axle['b'][1]
        b1_def = np.clip(b1_def, 0, s_length)
        b2_def = np.clip(b2_def, 0, s_length)
    else:
        b1_def = 0
        b2_def = s_length
    bear1 = st.select_slider('轴承 1 位置 (mm)', length_range, b1_def)
    bear2 = st.select_slider('轴承 2 位置 (mm)', length_range, b2_def)
    sel_axle['b'] = [bear1, bear2]
    
    st.session_state.sel_axle = sel_axle
    
    s.fix_bearing(bear1, bear2)

    return s

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

passed, byfig, bzfig, tfig, sigma_fig = calc_typeB(shaft, TT)
st.subheader('xoy 平面弯矩')
st.pyplot(byfig)
st.subheader('xoz 平面弯矩')
st.pyplot(bzfig)
st.subheader('xoy 平面转矩')
st.pyplot(tfig)
st.subheader('应力')
st.pyplot(sigma_fig)
# endregion 确定轴尺寸
