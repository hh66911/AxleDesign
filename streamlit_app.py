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
st.table(df)
# endregion 导入齿轮参数


#---------------------------------------
# region 确定轴形状
def make_shaft_ui(d_init: int, gears: list[int], is_IO=False):
    cols = df.columns[gears]
    bs = df.loc['齿宽 (mm)', cols].values
    bs = map(float, bs)
    st.write(f'轴上齿轮宽度总和：{sum(bs)} mm')
    s_length = st.number_input('草图长度 (mm)', value=sum(bs) + 200)
    length_range = range(0, s_length + 2)
    
    bs = df.loc['齿宽 (mm)', :].values
    bs = list(map(float, bs))
    ft = df.loc['切向力 (N)', :].values
    fr = df.loc['径向力 (N)', :].values
    fa = df.loc['轴向力 (N)', :].values
    ds = df.loc['分度圆直径 (mm)', :].values
    s = Shaft(d_init, s_length)
        
    for i in gears:
        gear_pos = st.select_slider(f'齿轮 {i + 1} 位置 (mm)',
            length_range, value=round(sum(bs[:i])))
        dir_inv = st.checkbox(f'齿轮 {i + 1} 方向是否相反')
        if dir_inv:
            fti, fri, fai = -float(ft[i]), -float(fr[i]), -float(fa[i])
        else:
            fti, fri, fai = float(ft[i]), float(fr[i]), float(fa[i])
        s.add_gear(gear_pos,
            float(bs[i]), float(ds[i]),
            fti, fri, fai)
    
    if is_IO:
        coupling = st.select_slider('联轴器位置', length_range, 0)
        s.fix_twist(coupling)
        
    bear1 = st.select_slider('轴承 1 位置 (mm)', length_range, 0)
    bear2 = st.select_slider('轴承 2 位置 (mm)', length_range, s_length)
    s.fix_bearing(bear1, bear2)

    return s

st.header(r'轴 $\text{I}$')
shaft = make_shaft_ui(diameters[0], [0],
                      st.checkbox('是否有联轴器'))
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