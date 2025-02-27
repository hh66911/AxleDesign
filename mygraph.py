import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 18

def m_graph(f, b, l, n=5000):
    forces = list(map(lambda x: (x[0], x[1]), sorted(f, key=lambda x: x[0])))
    bends = list(map(lambda x: (x[0], x[1]), sorted(b, key=lambda x: x[0])))
    
    pos_values = np.linspace(0, l, n)
    M_values = np.zeros(n)
    
    critical_points = dict()
    for fpos, f in forces:
        mask = pos_values > fpos
        if np.any(mask) == False:
            break
        critical_points[fpos] = M_values[mask][0]
        new_val = (pos_values - fpos) * f
        M_values[mask] += new_val[mask]
        
    # 删除起始的 0
    c_pos = list(critical_points.keys())
    for k in c_pos:
        if critical_points[k] == 0:
            del critical_points[k]
        
    for bend in bends:
        mask = pos_values >= bend[0]
                
        if bend[0] not in critical_points:
            critical_points[bend[0]] = M_values[mask][0]
        elif critical_points[bend[0]] * bend[1] >= 0:
            # 保留最大值
            critical_points[bend[0]] += bend[1]
            
        M_values[mask] += bend[1]
        
        for k in critical_points.keys():
            if k > bend[0]:
                critical_points[k] += bend[1]
    
    fig = plt.figure(figsize=(10, 4))
    plt.xlabel('Position [mm]')
    plt.ylabel('Moment [Nmm]')
    plt.plot(pos_values, M_values)
    plt.xlim(0, l)
    M_range = (min(M_values), max(M_values))
    M_range_size = (M_range[1] - M_range[0]) * 0.1
    plt.ylim(M_range[0] - M_range_size, M_range[1] + M_range_size)
    for x, y in critical_points.items():
        plt.text(x, y, f'{y: .4e}', fontsize=14, color='red')
    plt.grid()
    plt.tight_layout()
    
    return (pos_values, M_values), fig

def t_graph(t, l, n=5000):
    pos_values = np.linspace(0, l, n)
    T_values = np.zeros(n)
    
    critical_points = dict()
    for tpos, t in t:
        mask = pos_values > tpos
        if np.any(mask) == False:
            break
        critical_points[tpos] = T_values[mask][0]
        T_values[mask] += t
        
    # 删除起始的 0
    c_pos = list(critical_points.keys())
    for k in c_pos:
        if critical_points[k] == 0:
            del critical_points[k]
        
    fig = plt.figure(figsize=(10, 4))
    plt.xlabel('Position [mm]')
    plt.ylabel('Torque [Nmm]')
    plt.plot(pos_values, T_values)
    plt.xlim(0, l)
    T_range = (min(T_values), max(T_values))
    T_range_size = (T_range[1] - T_range[0]) * 0.1
    plt.ylim(T_range[0] - T_range_size, T_range[1] + T_range_size)
    for x, y in critical_points.items():
        plt.text(x, y, f'{y: .4e}', fontsize=14, color='red')
    plt.grid()
    plt.tight_layout()
    
    return (pos_values, T_values), fig
    