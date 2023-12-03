import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import os
from scipy import fft
import time

# 绘图控制colorbar刻度的
from matplotlib import ticker

# 人工加密方程点
def dense_eqns(x_min,x_max,dx,y_min,y_max,dy):
    data_x = np.arange(x_min,x_max,dx)
    data_y = np.arange(y_min,y_max,dy)
    data_mesh = np.meshgrid(data_x,data_y)
    x_eqns_all = np.array(data_mesh[0]).reshape(-1,1)
    y_eqns_all = np.array(data_mesh[1]).reshape(-1,1)

    return x_eqns_all,y_eqns_all

# 区分壁面边界内外
def io_wall(cor_wall,x,y):
    """
    x,y以单列形式传入shape=(X,1)
    给cor_data里的数据加第三列
       第三轮列值为0(默认值)表示在wall外面或wall上
               为1表示在wall里面"""
    id_col = np.zeros_like(x)
    cor_data = np.hstack([x,y,id_col])

    wall_min = np.min(cor_wall,axis=0)
    wall_max = np.max(cor_wall,axis=0)
    p_num = 1
    for cor in cor_data:
        
        x_outer = cor[0] <= wall_min[0] or cor[0] >= wall_max[0]
        y_outer = cor[1] <= wall_min[1] or cor[1] >= wall_max[1]
        outer = x_outer | y_outer # 满足一项就在外面
        # 如果xy之一小于最小或者大于最大，在外面
        if outer:
            continue
        else:
            # 否则，计算此点引出向左的射线与wall边界的交点
            crossing = 0 # 交点个数
            # 两两遍历wall边界
            for i in range(cor_wall.shape[0]):
                x1,y1 = cor_wall[i-1,0],cor_wall[i-1,1]
                x2,y2 = cor_wall[i,0],cor_wall[i,1]
                if (y1>cor[1]) == (y2>cor[1]): # 必须加括号！
                    continue # 此点y坐标不在此段边界的y范围内，即向左的射线无交点
                else:
                    slope = (y2-y1)/(x2-x1)
                    x = (x2-x1)*(cor[1]-y1)/(y2-y1)+x1
                    # 如果交点的x小于cor[0]，则向左的射线有一个交点
                    if x <=cor[0]:
                        crossing = crossing + 1
            # 交点之和为奇数，则在内部
            if crossing % 2 != 0:
                cor[2] = 1
                # print("第{}个点 {} 在边界内部,crossing是{}个".format(p_num,cor,crossing))
            else:
                continue
                # print("第{}个点 {} 靠近边界但在外面,crossing是{}个".format(p_num,cor,crossing))
        p_num = p_num + 1

    return cor_data

def Non_Dim(x_all, y_all, t_all, p_all, u_all, v_all,L, U, rou, p_ref):
    """
    nondimensionless for input data
    ref长度L, ref速度U, ref时间=L/U, ref密度rou, ref压力rou*U^2
    """
    x_ndm = x_all/L
    y_ndm = y_all/L
    t_ndm = t_all/(L/U)
    p_ndm = (p_all-p_ref)/(rou*U*U)
    u_ndm = u_all/U
    v_ndm = v_all/U

    return x_ndm, y_ndm, t_ndm, p_ndm, u_ndm, v_ndm

def Non_Dim_3D(x_all, y_all, z_all, t_all, p_all, u_all, v_all, w_all, L, U, rou, p_ref):
    """
    nondimensionless for input data
    ref长度L, ref速度U, ref时间=L/U, ref密度rou, ref压力rou*U^2
    """
    x_ndm = x_all/L
    y_ndm = y_all/L
    z_ndm = z_all/L
    t_ndm = t_all/(L/U)
    p_ndm = (p_all-p_ref)/(rou*U*U)
    u_ndm = u_all/U
    v_ndm = v_all/U
    w_ndm = w_all/U

    return x_ndm, y_ndm, z_ndm, t_ndm, p_ndm, u_ndm, v_ndm, w_ndm
    
def Dim(x_ndm, y_ndm, p_ndm, u_ndm, v_ndm,L, U, rou):
    """
    恢复量纲 for input data
    ref长度L, ref速度U, ref时间=L/U, ref密度rou, ref压力rou*U^2
    """
    x_dm = x_ndm*L
    y_dm = y_ndm*L
    # t_dm = t_ndm*(L/U)
    p_dm = p_ndm*(rou*U*U)
    u_dm = u_ndm*U
    v_dm = v_ndm*U

    return x_dm, y_dm, p_dm, u_dm, v_dm

def Dim_3D(x_ndm, y_ndm, z_ndm, t_ndm, p_ndm, u_ndm, v_ndm, w_ndm, L, U, rou):
    """
    恢复量纲 for input data
    ref长度L, ref速度U, ref时间=L/U, ref密度rou, ref压力rou*U^2
    """
    x_dm = x_ndm*L
    y_dm = y_ndm*L
    z_dm = z_ndm*L
    t_dm = t_ndm*(L/U)
    p_dm = p_ndm*(rou*U*U)
    u_dm = u_ndm*U
    v_dm = v_ndm*U
    w_dm = w_ndm*U

    return x_dm, y_dm, z_dm, t_dm, p_dm, u_dm, v_dm, w_dm


def Read_csv_3D(path,var_id, start, dt, Nx, Ny, Nz, Nt, x_jump=0, y_jump=0, z_jump=0, t_jump=0):
    """path of data, num of data, sample interval"""

    x = np.zeros((Ny * Nx * Nz, 1), dtype='float64')
    y = np.zeros((Ny * Nx * Nz, 1), dtype='float64')
    z = np.zeros((Ny * Nx * Nz, 1), dtype='float64')
    
    read_path = path.format(0)
    var_all = np.loadtxt(open(read_path.format(0), "rb"), delimiter=",", skiprows=1)
    x_ns = var_all[:,var_id[0]] # no sort
    y_ns = var_all[:,var_id[1]]
    z_ns = var_all[:,var_id[2]]
    arg_xyz = np.lexsort((z_ns,y_ns,x_ns))
    x = x_ns[arg_xyz]
    y = y_ns[arg_xyz]
    z = z_ns[arg_xyz]

    if t_jump == 0:

        p = np.zeros((Ny * Nx * Nz, Nt), dtype='float64')
        u = np.zeros((Ny * Nx * Nz, Nt), dtype='float64')
        v = np.zeros((Ny * Nx * Nz, Nt), dtype='float64')
        w = np.zeros((Ny * Nx * Nz, Nt), dtype='float64')
        t = np.zeros((Nt, 1), dtype='float64')

        for i in range(Nt):
            t[i] = start + i * dt
            read_path = path.format(i)
            var_all = np.loadtxt(open(read_path.format(0), "rb"), delimiter=",", skiprows=1) # 读取

            p_ns = var_all[:,var_id[3]]
            u_ns = var_all[:,var_id[4]]
            v_ns = var_all[:,var_id[5]]
            w_ns = var_all[:,var_id[6]]
            u[:, i] = u_ns[arg_xyz]
            v[:, i] = v_ns[arg_xyz]
            w[:, i] = w_ns[arg_xyz]
            p[:, i] = p_ns[arg_xyz]
            print("csv {}, Time {:.3f}s is done.".format(i,start + i * dt))
            sys.stdout.flush()
        if (x_jump != 0)or(y_jump != 0)or(z_jump != 0):
            x_re = x.reshape(Nz,Ny,Nx)[0:Nz:z_jump,0:Ny:y_jump,0:Nx:x_jump]
            y_re = y.reshape(Nz,Ny,Nx)[0:Nz:z_jump,0:Ny:y_jump,0:Nx:x_jump]
            z_re = z.reshape(Nz,Ny,Nx)[0:Nz:z_jump,0:Ny:y_jump,0:Nx:x_jump]
            x = x_re.flatten()
            y = y_re.flatten()
            z = z_re.flatten()
            p_re = p.reshape(Nz,Ny,Nx,Nt)[0:Nz:z_jump,0:Ny:y_jump,0:Nx:x_jump,:]
            u_re = u.reshape(Nz,Ny,Nx,Nt)[0:Nz:z_jump,0:Ny:y_jump,0:Nx:x_jump,:]
            v_re = v.reshape(Nz,Ny,Nx,Nt)[0:Nz:z_jump,0:Ny:y_jump,0:Nx:x_jump,:]
            w_re = w.reshape(Nz,Ny,Nx,Nt)[0:Nz:z_jump,0:Ny:y_jump,0:Nx:x_jump,:]
            p = p_re.reshape(-1,Nt)
            u = u_re.reshape(-1,Nt)
            v = v_re.reshape(-1,Nt)
            w = w_re.reshape(-1,Nt)

    else:

        Nt_j = int(Nt/t_jump) # 2
        p = np.zeros((Nz * Ny * Nx, Nt_j), dtype='float64')
        u = np.zeros((Nz * Ny * Nx, Nt_j), dtype='float64')
        v = np.zeros((Nz * Ny * Nx, Nt_j), dtype='float64')
        w = np.zeros((Nz * Ny * Nx, Nt_j), dtype='float64')
        t = np.zeros((Nt_j, 1), dtype='float64')

        for i in range(Nt_j): # 2
            t[i] = start + i * dt * t_jump # 0.122+1*0.195*2
            read_path = path.format(i * t_jump) # 2: 1*2
            var_all = np.loadtxt(open(read_path.format(0), "rb"), delimiter=",", skiprows=1) # 读取
            
            p_ns = var_all[:,var_id[3]]
            u_ns = var_all[:,var_id[4]]
            v_ns = var_all[:,var_id[5]]
            w_ns = var_all[:,var_id[6]]

            u[:, i] = u_ns[arg_xyz]
            v[:, i] = v_ns[arg_xyz]
            w[:, i] = w_ns[arg_xyz]
            p[:, i] = p_ns[arg_xyz]
            print("csv {}, Time {:.3f}s is done.".format(i * t_jump,start + i * dt * t_jump))
            sys.stdout.flush()

        if (x_jump != 0)or(y_jump != 0)or(z_jump != 0):

            x_re = x.reshape(Nz,Ny,Nx)[0:Nz:z_jump,0:Ny:y_jump,0:Nx:x_jump]
            y_re = y.reshape(Nz,Ny,Nx)[0:Nz:z_jump,0:Ny:y_jump,0:Nx:x_jump]
            z_re = z.reshape(Nz,Ny,Nx)[0:Nz:z_jump,0:Ny:y_jump,0:Nx:x_jump]
            x = x_re.flatten()
            y = y_re.flatten()
            z = z_re.flatten()
            p_re = p.reshape(Nz,Ny,Nx,Nt_j)[0:Nz:z_jump,0:Ny:y_jump,0:Nx:x_jump,:]
            u_re = u.reshape(Nz,Ny,Nx,Nt_j)[0:Nz:z_jump,0:Ny:y_jump,0:Nx:x_jump,:]
            v_re = v.reshape(Nz,Ny,Nx,Nt_j)[0:Nz:z_jump,0:Ny:y_jump,0:Nx:x_jump,:]
            w_re = w.reshape(Nz,Ny,Nx,Nt_j)[0:Nz:z_jump,0:Ny:y_jump,0:Nx:x_jump,:]
            p = p_re.reshape(-1,Nt_j)
            u = u_re.reshape(-1,Nt_j)
            v = v_re.reshape(-1,Nt_j)
            w = w_re.reshape(-1,Nt_j)
            
    return x, y, z, t, p, u, v, w

def Read_Fluent(path, t_num, start, dt, Nx, Ny, x_jump=0,y_jump=0):
    """path of data, num of data, sample interval"""

    p = np.zeros((Ny * Nx, t_num), dtype='float32')
    u = np.zeros((Ny * Nx, t_num), dtype='float32')
    v = np.zeros((Ny * Nx, t_num), dtype='float32')
    x = np.zeros((Ny * Nx, 1), dtype='float32')
    y = np.zeros((Ny * Nx, 1), dtype='float32')
    t = np.zeros((t_num, 1), dtype='float32')

    read_path = path.format(start + dt)
    x = np.loadtxt(open(read_path, "rb"), delimiter=" ", skiprows=11, usecols=[1])
    y = np.loadtxt(open(read_path, "rb"), delimiter=" ", skiprows=11, usecols=[2])

    for i in range(t_num):
        t[i] = i * dt
        read_path = path.format(start + (i + 1) * dt)
        u[:, i] = np.loadtxt(open(read_path, "rb"), delimiter=" ", skiprows=11, usecols=[3])
        v[:, i] = np.loadtxt(open(read_path, "rb"), delimiter=" ", skiprows=11, usecols=[4])
        p[:, i] = np.loadtxt(open(read_path, "rb"), delimiter=" ", skiprows=11, usecols=[5])
        print("Snap {}, Time {:.2f}s is done.".format(i+1,start + (i + 1) * dt))
    if x_jump != 0:
        if y_jump !=0:
            x_re = x.reshape(Ny,Nx)[0:Ny:y_jump,0:Nx:x_jump]
            y_re = y.reshape(Ny,Nx)[0:Ny:y_jump,0:Nx:x_jump]
            x = x_re.flatten()
            y = y_re.flatten()
            p_re = p.reshape(Ny,Nx,t_num)[0:Ny:y_jump,0:Nx:x_jump,:]
            u_re = u.reshape(Ny,Nx,t_num)[0:Ny:y_jump,0:Nx:x_jump,:]
            v_re = v.reshape(Ny,Nx,t_num)[0:Ny:y_jump,0:Nx:x_jump,:]
            p = p_re.reshape(-1,t_num)
            u = u_re.reshape(-1,t_num)
            v = v_re.reshape(-1,t_num)
        else:
            x_re = x.reshape(Ny,Nx)[:,0:Nx:x_jump]
            x = x_re.flatten()
            p_re = p.reshape(Ny,Nx,t_num)[:,0:Nx:x_jump,:]
            u_re = u.reshape(Ny,Nx,t_num)[:,0:Nx:x_jump,:]
            v_re = v.reshape(Ny,Nx,t_num)[:,0:Nx:x_jump,:]
            p = p_re.reshape(-1,t_num)
            u = u_re.reshape(-1,t_num)
            v = v_re.reshape(-1,t_num)
    else: 
        if y_jump !=0:
            y_re = y.reshape(Ny,Nx)[0:Ny:y_jump,0:Nx:x_jump]
            y = y_re.flatten()
            p_re = p.reshape(Ny,Nx,t_num)[0:Ny:y_jump,:,:]
            u_re = u.reshape(Ny,Nx,t_num)[0:Ny:y_jump,:,:]
            v_re = v.reshape(Ny,Nx,t_num)[0:Ny:y_jump,:,:]
            p = p_re.reshape(-1,t_num)
            u = u_re.reshape(-1,t_num)
            v = v_re.reshape(-1,t_num)

    return x, y, t, p, u, v

def Read_Fluent_t_jump(path, start, dt, Nx, Ny, Nt, x_jump=0, y_jump=0, t_jump=0):
    """path of data, num of data, sample interval"""

    p = np.zeros((Ny * Nx, Nt), dtype='float32')
    u = np.zeros((Ny * Nx, Nt), dtype='float32')
    v = np.zeros((Ny * Nx, Nt), dtype='float32')
    x = np.zeros((Ny * Nx, 1), dtype='float32')
    y = np.zeros((Ny * Nx, 1), dtype='float32')
    t = np.zeros((Nt, 1), dtype='float32')

    read_path = path.format(start + dt)
    x = np.loadtxt(open(read_path, "rb"), delimiter=" ", skiprows=11, usecols=[1])
    y = np.loadtxt(open(read_path, "rb"), delimiter=" ", skiprows=11, usecols=[2])


    if t_jump == 0:
        for i in range(Nt):
            t[i] = start + (i + 1) * dt
            read_path = path.format(start + (i + 1) * dt)
            u[:, i] = np.loadtxt(open(read_path, "rb"), delimiter=" ", skiprows=11, usecols=[3])
            v[:, i] = np.loadtxt(open(read_path, "rb"), delimiter=" ", skiprows=11, usecols=[4])
            p[:, i] = np.loadtxt(open(read_path, "rb"), delimiter=" ", skiprows=11, usecols=[5])
            print("Snap {}, Time {:.2f}s is done.".format(i+1,start + (i + 1) * dt))
        if x_jump != 0:
            if y_jump !=0:
                x_re = x.reshape(Ny,Nx)[0:Ny:y_jump,0:Nx:x_jump]
                y_re = y.reshape(Ny,Nx)[0:Ny:y_jump,0:Nx:x_jump]
                x = x_re.flatten()
                y = y_re.flatten()
                p_re = p.reshape(Ny,Nx,Nt)[0:Ny:y_jump,0:Nx:x_jump,:]
                u_re = u.reshape(Ny,Nx,Nt)[0:Ny:y_jump,0:Nx:x_jump,:]
                v_re = v.reshape(Ny,Nx,Nt)[0:Ny:y_jump,0:Nx:x_jump,:]
                p = p_re.reshape(-1,Nt)
                u = u_re.reshape(-1,Nt)
                v = v_re.reshape(-1,Nt)
            else:
                x_re = x.reshape(Ny,Nx)[:,0:Nx:x_jump]
                x = x_re.flatten()
                p_re = p.reshape(Ny,Nx,Nt)[:,0:Nx:x_jump,:]
                u_re = u.reshape(Ny,Nx,Nt)[:,0:Nx:x_jump,:]
                v_re = v.reshape(Ny,Nx,Nt)[:,0:Nx:x_jump,:]
                p = p_re.reshape(-1,Nt)
                u = u_re.reshape(-1,Nt)
                v = v_re.reshape(-1,Nt)
        else: 
            if y_jump !=0:
                y_re = y.reshape(Ny,Nx)[0:Ny:y_jump,0:Nx:x_jump]
                y = y_re.flatten()
                p_re = p.reshape(Ny,Nx,Nt)[0:Ny:y_jump,:,:]
                u_re = u.reshape(Ny,Nx,Nt)[0:Ny:y_jump,:,:]
                v_re = v.reshape(Ny,Nx,Nt)[0:Ny:y_jump,:,:]
                p = p_re.reshape(-1,Nt)
                u = u_re.reshape(-1,Nt)
                v = v_re.reshape(-1,Nt)
    else:
        Nt_j = int(Nt/t_jump)
        p = np.zeros((Ny * Nx, Nt_j), dtype='float32')
        u = np.zeros((Ny * Nx, Nt_j), dtype='float32')
        v = np.zeros((Ny * Nx, Nt_j), dtype='float32')
        t = np.zeros((Nt_j, 1), dtype='float32')

        for i in range(Nt_j):
            t[i] = start + dt + i * dt * t_jump
            read_path = path.format(start + dt + i * dt * t_jump)
            u[:, i] = np.loadtxt(open(read_path, "rb"), delimiter=" ", skiprows=11, usecols=[3])
            v[:, i] = np.loadtxt(open(read_path, "rb"), delimiter=" ", skiprows=11, usecols=[4])
            p[:, i] = np.loadtxt(open(read_path, "rb"), delimiter=" ", skiprows=11, usecols=[5])
            print("Snap {}, Time {:.2f}s is done.".format(i+1,start + dt + i * dt * t_jump))

        if x_jump != 0:
            if y_jump !=0:
                x_re = x.reshape(Ny,Nx)[0:Ny:y_jump,0:Nx:x_jump]
                y_re = y.reshape(Ny,Nx)[0:Ny:y_jump,0:Nx:x_jump]
                x = x_re.flatten()
                y = y_re.flatten()
                p_re = p.reshape(Ny,Nx,Nt)[0:Ny:y_jump,0:Nx:x_jump,0:Nt:t_jump]
                u_re = u.reshape(Ny,Nx,Nt)[0:Ny:y_jump,0:Nx:x_jump,0:Nt:t_jump]
                v_re = v.reshape(Ny,Nx,Nt)[0:Ny:y_jump,0:Nx:x_jump,0:Nt:t_jump]
                p = p_re.reshape(-1,Nt_j)
                u = u_re.reshape(-1,Nt_j)
                v = v_re.reshape(-1,Nt_j)
            else:
                x_re = x.reshape(Ny,Nx)[:,0:Nx:x_jump]
                x = x_re.flatten()
                p_re = p.reshape(Ny,Nx,Nt_j)[:,0:Nx:x_jump,0:Nt:t_jump]
                u_re = u.reshape(Ny,Nx,Nt_j)[:,0:Nx:x_jump,0:Nt:t_jump]
                v_re = v.reshape(Ny,Nx,Nt_j)[:,0:Nx:x_jump,0:Nt:t_jump]
                p = p_re.reshape(-1,Nt_j)
                u = u_re.reshape(-1,Nt_j)
                v = v_re.reshape(-1,Nt_j)
        else: 
            if y_jump !=0:
                y_re = y.reshape(Ny,Nx)[0:Ny:y_jump,0:Nx:x_jump]
                y = y_re.flatten()
                p_re = p.reshape(Ny,Nx,Nt)[0:Ny:y_jump,:,0:Nt:t_jump]
                u_re = u.reshape(Ny,Nx,Nt)[0:Ny:y_jump,:,0:Nt:t_jump]
                v_re = v.reshape(Ny,Nx,Nt)[0:Ny:y_jump,:,0:Nt:t_jump]
                p = p_re.reshape(-1,Nt_j)
                u = u_re.reshape(-1,Nt_j)
                v = v_re.reshape(-1,Nt_j)
            
    return x, y, t, p, u, v


def Read_StarCCM(path, t_num, start, dt, Nx, Ny, x_jump=0,y_jump=0):
    """path of data, num of data, sample interval"""

    p = np.zeros((Ny * Nx, t_num), dtype='float32')
    u = np.zeros((Ny * Nx, t_num), dtype='float32')
    v = np.zeros((Ny * Nx, t_num), dtype='float32')
    x = np.zeros((Ny * Nx, 1), dtype='float32')
    y = np.zeros((Ny * Nx, 1), dtype='float32')
    t = np.zeros((t_num, 1), dtype='float32')

    read_path = path.format(start + dt)
    x = np.loadtxt(open(read_path, "rb"), delimiter=",", skiprows=1, usecols=[3])
    y = np.loadtxt(open(read_path, "rb"), delimiter=",", skiprows=1, usecols=[4])

    for i in range(t_num):
        t[i] = start + (i + 1) * dt
        read_path = path.format(start + (i + 1) * dt)
        p[:, i] = np.loadtxt(open(read_path, "rb"), delimiter=",", skiprows=1, usecols=[0])
        u[:, i] = np.loadtxt(open(read_path, "rb"), delimiter=",", skiprows=1, usecols=[1])
        v[:, i] = np.loadtxt(open(read_path, "rb"), delimiter=",", skiprows=1, usecols=[2])
        print("Snap {}, Time {:.2f}s is done.".format(i+1,start + (i + 1) * dt))
    if x_jump != 0:
        if y_jump !=0:
            x_re = x.reshape(Ny,Nx)[0:Ny:y_jump,0:Nx:x_jump]
            y_re = y.reshape(Ny,Nx)[0:Ny:y_jump,0:Nx:x_jump]
            x = x_re.flatten()
            y = y_re.flatten()
            p_re = p.reshape(Ny,Nx,t_num)[0:Ny:y_jump,0:Nx:x_jump,:]
            u_re = u.reshape(Ny,Nx,t_num)[0:Ny:y_jump,0:Nx:x_jump,:]
            v_re = v.reshape(Ny,Nx,t_num)[0:Ny:y_jump,0:Nx:x_jump,:]
            p = p_re.reshape(-1,t_num)
            u = u_re.reshape(-1,t_num)
            v = v_re.reshape(-1,t_num)
        else:
            x_re = x.reshape(Ny,Nx)[:,0:Nx:x_jump]
            x = x_re.flatten()
            p_re = p.reshape(Ny,Nx,t_num)[:,0:Nx:x_jump,:]
            u_re = u.reshape(Ny,Nx,t_num)[:,0:Nx:x_jump,:]
            v_re = v.reshape(Ny,Nx,t_num)[:,0:Nx:x_jump,:]
            p = p_re.reshape(-1,t_num)
            u = u_re.reshape(-1,t_num)
            v = v_re.reshape(-1,t_num)
    else: 
        if y_jump !=0:
            y_re = y.reshape(Ny,Nx)[0:Ny:y_jump,0:Nx:x_jump]
            y = y_re.flatten()
            p_re = p.reshape(Ny,Nx,t_num)[0:Ny:y_jump,:,:]
            u_re = u.reshape(Ny,Nx,t_num)[0:Ny:y_jump,:,:]
            v_re = v.reshape(Ny,Nx,t_num)[0:Ny:y_jump,:,:]
            p = p_re.reshape(-1,t_num)
            u = u_re.reshape(-1,t_num)
            v = v_re.reshape(-1,t_num)

    return x, y, t, p, u, v

def Read_StarCCM_t_jump(path, start, dt, Nx, Ny, Nt, x_jump=0,y_jump=0,t_jump=0):
    """path of data, num of data, sample interval"""

    p = np.zeros((Ny * Nx, Nt), dtype='float32')
    u = np.zeros((Ny * Nx, Nt), dtype='float32')
    v = np.zeros((Ny * Nx, Nt), dtype='float32')
    x = np.zeros((Ny * Nx, 1), dtype='float32')
    y = np.zeros((Ny * Nx, 1), dtype='float32')
    t = np.zeros((Nt, 1), dtype='float32')

    read_path = path.format(start + dt)
    x = np.loadtxt(open(read_path, "rb"), delimiter=",", skiprows=1, usecols=[3])
    y = np.loadtxt(open(read_path, "rb"), delimiter=",", skiprows=1, usecols=[4])
    if t_jump == 0:
        for i in range(Nt):
            t[i] = start + (i + 1) * dt
            read_path = path.format(start + (i + 1) * dt)
            p[:, i] = np.loadtxt(open(read_path, "rb"), delimiter=",", skiprows=1, usecols=[0])
            u[:, i] = np.loadtxt(open(read_path, "rb"), delimiter=",", skiprows=1, usecols=[1])
            v[:, i] = np.loadtxt(open(read_path, "rb"), delimiter=",", skiprows=1, usecols=[2])
            print("Snap {}, Time {:.2f}s is done.".format(i+1,start + (i + 1) * dt))
        if x_jump != 0:
            if y_jump !=0:
                x_re = x.reshape(Ny,Nx)[0:Ny:y_jump,0:Nx:x_jump]
                y_re = y.reshape(Ny,Nx)[0:Ny:y_jump,0:Nx:x_jump]
                x = x_re.flatten()
                y = y_re.flatten()
                p_re = p.reshape(Ny,Nx,Nt)[0:Ny:y_jump,0:Nx:x_jump,:]
                u_re = u.reshape(Ny,Nx,Nt)[0:Ny:y_jump,0:Nx:x_jump,:]
                v_re = v.reshape(Ny,Nx,Nt)[0:Ny:y_jump,0:Nx:x_jump,:]
                p = p_re.reshape(-1,Nt)
                u = u_re.reshape(-1,Nt)
                v = v_re.reshape(-1,Nt)
            else:
                x_re = x.reshape(Ny,Nx)[:,0:Nx:x_jump]
                x = x_re.flatten()
                p_re = p.reshape(Ny,Nx,Nt)[:,0:Nx:x_jump,:]
                u_re = u.reshape(Ny,Nx,Nt)[:,0:Nx:x_jump,:]
                v_re = v.reshape(Ny,Nx,Nt)[:,0:Nx:x_jump,:]
                p = p_re.reshape(-1,Nt)
                u = u_re.reshape(-1,Nt)
                v = v_re.reshape(-1,Nt)
        else: 
            if y_jump !=0:
                y_re = y.reshape(Ny,Nx)[0:Ny:y_jump,0:Nx:x_jump]
                y = y_re.flatten()
                p_re = p.reshape(Ny,Nx,Nt)[0:Ny:y_jump,:,:]
                u_re = u.reshape(Ny,Nx,Nt)[0:Ny:y_jump,:,:]
                v_re = v.reshape(Ny,Nx,Nt)[0:Ny:y_jump,:,:]
                p = p_re.reshape(-1,Nt)
                u = u_re.reshape(-1,Nt)
                v = v_re.reshape(-1,Nt)
    else:
        Nt_j = int(Nt/t_jump)
        t = np.zeros((Nt_j, 1), dtype='float32')
        for i in range(Nt_j):
            t[i] = start + dt + i * dt * t_jump
            read_path = path.format(start + dt + i * dt * t_jump)
            p[:, i] = np.loadtxt(open(read_path, "rb"), delimiter=",", skiprows=1, usecols=[0])
            u[:, i] = np.loadtxt(open(read_path, "rb"), delimiter=",", skiprows=1, usecols=[1])
            v[:, i] = np.loadtxt(open(read_path, "rb"), delimiter=",", skiprows=1, usecols=[2])
            print("Snap {}, Time {:.2f}s is done.".format(i+1,start + dt + i * dt * t_jump))
        if x_jump != 0:
            if y_jump !=0:
                x_re = x.reshape(Ny,Nx)[0:Ny:y_jump,0:Nx:x_jump]
                y_re = y.reshape(Ny,Nx)[0:Ny:y_jump,0:Nx:x_jump]
                x = x_re.flatten()
                y = y_re.flatten()
                p_re = p.reshape(Ny,Nx,Nt)[0:Ny:y_jump,0:Nx:x_jump,0:Nt:t_jump]
                u_re = u.reshape(Ny,Nx,Nt)[0:Ny:y_jump,0:Nx:x_jump,0:Nt:t_jump]
                v_re = v.reshape(Ny,Nx,Nt)[0:Ny:y_jump,0:Nx:x_jump,0:Nt:t_jump]
                p = p_re.reshape(-1,Nt_j)
                u = u_re.reshape(-1,Nt_j)
                v = v_re.reshape(-1,Nt_j)
            else:
                x_re = x.reshape(Ny,Nx)[:,0:Nx:x_jump]
                x = x_re.flatten()
                p_re = p.reshape(Ny,Nx,Nt_j)[:,0:Nx:x_jump,0:Nt:t_jump]
                u_re = u.reshape(Ny,Nx,Nt_j)[:,0:Nx:x_jump,0:Nt:t_jump]
                v_re = v.reshape(Ny,Nx,Nt_j)[:,0:Nx:x_jump,0:Nt:t_jump]
                p = p_re.reshape(-1,Nt_j)
                u = u_re.reshape(-1,Nt_j)
                v = v_re.reshape(-1,Nt_j)
        else: 
            if y_jump !=0:
                y_re = y.reshape(Ny,Nx)[0:Ny:y_jump,0:Nx:x_jump]
                y = y_re.flatten()
                p_re = p.reshape(Ny,Nx,Nt_j)[0:Ny:y_jump,:,0:Nt:t_jump]
                u_re = u.reshape(Ny,Nx,Nt_j)[0:Ny:y_jump,:,0:Nt:t_jump]
                v_re = v.reshape(Ny,Nx,Nt_j)[0:Ny:y_jump,:,0:Nt:t_jump]
                p = p_re.reshape(-1,Nt_j)
                u = u_re.reshape(-1,Nt_j)
                v = v_re.reshape(-1,Nt_j)
    return x, y, t, p, u, v

def Read_csv(path,var_id, start, dt, Nx, Ny, Nt, x_jump=0, y_jump=0, t_jump=0):
    """path of data, num of data, sample interval"""

    # uvpt用到Nt的，放到判断t_jump之后去定义
    x = np.zeros((Ny * Nx, 1), dtype='float64')
    y = np.zeros((Ny * Nx, 1), dtype='float64')

    read_path = path.format(0)
    var_all = np.loadtxt(open(read_path.format(0), "rb"), delimiter=",", skiprows=1)
    x_ns = var_all[:,var_id[0]]
    y_ns = var_all[:,var_id[1]]
    arg_xy = np.lexsort((y_ns,x_ns))
    x = x_ns[arg_xy]
    y = y_ns[arg_xy]

    if t_jump == 0:
        p = np.zeros((Ny * Nx, Nt), dtype='float64')
        u = np.zeros((Ny * Nx, Nt), dtype='float64')
        v = np.zeros((Ny * Nx, Nt), dtype='float64')
        t = np.zeros((Nt, 1), dtype='float64')

        read_path = path.format(0)
        var_all = np.loadtxt(open(read_path.format(0), "rb"), delimiter=",", skiprows=1)
        x_ns = var_all[:,var_id[0]]
        y_ns = var_all[:,var_id[1]]
        arg_xy = np.lexsort((y_ns,x_ns))
        x = x_ns[arg_xy]
        y = y_ns[arg_xy]

        for i in range(Nt):
            t[i] = start + i * dt
            read_path = path.format(i)
            var_all = np.loadtxt(open(read_path.format(0), "rb"), delimiter=",", skiprows=1) # 读取
            x_ns = var_all[:,var_id[0]]
            y_ns = var_all[:,var_id[1]]
            p_ns = var_all[:,var_id[2]]
            u_ns = var_all[:,var_id[3]]
            v_ns = var_all[:,var_id[4]]
            
            arg_xy = np.lexsort((y_ns,x_ns))
            u[:, i] = u_ns[arg_xy]
            v[:, i] = v_ns[arg_xy]
            p[:, i] = p_ns[arg_xy]
            print("csv {}, Time {:.3f}s is done.".format(i,start + i * dt))
        if (x_jump != 0)or(y_jump != 0):
            x_re = x.reshape(Ny,Nx)[0:Ny:y_jump,0:Nx:x_jump]
            y_re = y.reshape(Ny,Nx)[0:Ny:y_jump,0:Nx:x_jump]
            x = x_re.flatten()
            y = y_re.flatten()
            p_re = p.reshape(Ny,Nx,Nt)[0:Ny:y_jump,0:Nx:x_jump,:]
            u_re = u.reshape(Ny,Nx,Nt)[0:Ny:y_jump,0:Nx:x_jump,:]
            v_re = v.reshape(Ny,Nx,Nt)[0:Ny:y_jump,0:Nx:x_jump,:]
            p = p_re.reshape(-1,Nt)
            u = u_re.reshape(-1,Nt)
            v = v_re.reshape(-1,Nt)

    else:
        Nt_j = int(Nt/t_jump) # 2
        p = np.zeros((Ny * Nx, Nt_j), dtype='float32')
        u = np.zeros((Ny * Nx, Nt_j), dtype='float32')
        v = np.zeros((Ny * Nx, Nt_j), dtype='float32')
        t = np.zeros((Nt_j, 1), dtype='float32')

        for i in range(Nt_j): # 2
            t[i] = start + i * dt * t_jump # 0.122+1*0.195*2
            read_path = path.format(i * t_jump) # 2: 1*2
            var_all = np.loadtxt(open(read_path.format(0), "rb"), delimiter=",", skiprows=1) # 读取
            x_ns = var_all[:,var_id[0]]
            y_ns = var_all[:,var_id[1]]
            p_ns = var_all[:,var_id[2]]
            u_ns = var_all[:,var_id[3]]
            v_ns = var_all[:,var_id[4]]
            arg_xy = np.lexsort((y_ns,x_ns))
            u[:, i] = u_ns[arg_xy]
            v[:, i] = v_ns[arg_xy]
            p[:, i] = p_ns[arg_xy]
            print("csv {}, Time {:.3f}s is done.".format(i * t_jump,start + i * dt * t_jump))

        if (x_jump != 0)or(y_jump != 0):
            x_re = x.reshape(Ny,Nx)[0:Ny:y_jump,0:Nx:x_jump]
            y_re = y.reshape(Ny,Nx)[0:Ny:y_jump,0:Nx:x_jump]
            x = x_re.flatten()
            y = y_re.flatten()
            p_re = p.reshape(Ny,Nx,Nt_j)[0:Ny:y_jump,0:Nx:x_jump,:] # 0:Nt_j还是:
            u_re = u.reshape(Ny,Nx,Nt_j)[0:Ny:y_jump,0:Nx:x_jump,:]
            v_re = v.reshape(Ny,Nx,Nt_j)[0:Ny:y_jump,0:Nx:x_jump,:]
            p = p_re.reshape(-1,Nt_j)
            u = u_re.reshape(-1,Nt_j)
            v = v_re.reshape(-1,Nt_j)
            
    return x, y, t, p, u, v

def Read_csv_3D(path,var_id, start, dt, Nx, Ny, Nz, Nt, x_jump=0, y_jump=0, z_jump=0, t_jump=0):
    """path of data, num of data, sample interval"""

    x = np.zeros((Ny * Nx * Nz, 1), dtype='float64')
    y = np.zeros((Ny * Nx * Nz, 1), dtype='float64')
    z = np.zeros((Ny * Nx * Nz, 1), dtype='float64')
    
    read_path = path.format(0)
    var_all = np.loadtxt(open(read_path.format(0), "rb"), delimiter=",", skiprows=1)
    x_ns = var_all[:,var_id[0]] # no sort
    y_ns = var_all[:,var_id[1]]
    z_ns = var_all[:,var_id[2]]
    arg_xyz = np.lexsort((z_ns,y_ns,x_ns))
    x = x_ns[arg_xyz]
    y = y_ns[arg_xyz]
    z = z_ns[arg_xyz]

    if t_jump == 0:

        p = np.zeros((Ny * Nx * Nz, Nt), dtype='float64')
        u = np.zeros((Ny * Nx * Nz, Nt), dtype='float64')
        v = np.zeros((Ny * Nx * Nz, Nt), dtype='float64')
        w = np.zeros((Ny * Nx * Nz, Nt), dtype='float64')
        t = np.zeros((Nt, 1), dtype='float64')

        for i in range(Nt):
            t[i] = start + i * dt
            read_path = path.format(i)
            var_all = np.loadtxt(open(read_path.format(0), "rb"), delimiter=",", skiprows=1) # 读取

            p_ns = var_all[:,var_id[3]]
            u_ns = var_all[:,var_id[4]]
            v_ns = var_all[:,var_id[5]]
            w_ns = var_all[:,var_id[6]]
            p[:, i] = p_ns[arg_xyz]
            u[:, i] = u_ns[arg_xyz]
            v[:, i] = v_ns[arg_xyz]
            w[:, i] = w_ns[arg_xyz]
            print("csv {}, Time {:.3f}s is done.".format(i,start + i * dt))
        if (x_jump != 0)or(y_jump != 0)or(z_jump != 0):
            x_re = x.reshape(Nz,Ny,Nx)[0:Nz:z_jump,0:Ny:y_jump,0:Nx:x_jump]
            y_re = y.reshape(Nz,Ny,Nx)[0:Nz:z_jump,0:Ny:y_jump,0:Nx:x_jump]
            z_re = z.reshape(Nz,Ny,Nx)[0:Nz:z_jump,0:Ny:y_jump,0:Nx:x_jump]
            x = x_re.flatten()
            y = y_re.flatten()
            z = z_re.flatten()
            p_re = p.reshape(Nz,Ny,Nx,Nt)[0:Nz:z_jump,0:Ny:y_jump,0:Nx:x_jump,:]
            u_re = u.reshape(Nz,Ny,Nx,Nt)[0:Nz:z_jump,0:Ny:y_jump,0:Nx:x_jump,:]
            v_re = v.reshape(Nz,Ny,Nx,Nt)[0:Nz:z_jump,0:Ny:y_jump,0:Nx:x_jump,:]
            w_re = w.reshape(Nz,Ny,Nx,Nt)[0:Nz:z_jump,0:Ny:y_jump,0:Nx:x_jump,:]
            p = p_re.reshape(-1,Nt)
            u = u_re.reshape(-1,Nt)
            v = v_re.reshape(-1,Nt)
            w = w_re.reshape(-1,Nt)

    else:

        Nt_j = int(Nt/t_jump) # 2
        p = np.zeros((Nz * Ny * Nx, Nt_j), dtype='float64')
        u = np.zeros((Nz * Ny * Nx, Nt_j), dtype='float64')
        v = np.zeros((Nz * Ny * Nx, Nt_j), dtype='float64')
        w = np.zeros((Nz * Ny * Nx, Nt_j), dtype='float64')
        t = np.zeros((Nt_j, 1), dtype='float64')

        for i in range(Nt_j): # 2
            t[i] = start + i * dt * t_jump # 0.122+1*0.195*2
            read_path = path.format(i * t_jump) # 2: 1*2
            var_all = np.loadtxt(open(read_path.format(0), "rb"), delimiter=",", skiprows=1) # 读取
            
            p_ns = var_all[:,var_id[3]]
            u_ns = var_all[:,var_id[4]]
            v_ns = var_all[:,var_id[5]]
            w_ns = var_all[:,var_id[6]]

            u[:, i] = u_ns[arg_xyz]
            v[:, i] = v_ns[arg_xyz]
            w[:, i] = w_ns[arg_xyz]
            p[:, i] = p_ns[arg_xyz]
            print("csv {}, Time {:.3f}s is done.".format(i * t_jump,start + i * dt * t_jump))

        if (x_jump != 0)or(y_jump != 0)or(z_jump != 0):

            x_re = x.reshape(Nz,Ny,Nx)[0:Nz:z_jump,0:Ny:y_jump,0:Nx:x_jump]
            y_re = y.reshape(Nz,Ny,Nx)[0:Nz:z_jump,0:Ny:y_jump,0:Nx:x_jump]
            z_re = z.reshape(Nz,Ny,Nx)[0:Nz:z_jump,0:Ny:y_jump,0:Nx:x_jump]
            x = x_re.flatten()
            y = y_re.flatten()
            z = z_re.flatten()
            p_re = p.reshape(Nz,Ny,Nx,Nt_j)[0:Nz:z_jump,0:Ny:y_jump,0:Nx:x_jump,:]
            u_re = u.reshape(Nz,Ny,Nx,Nt_j)[0:Nz:z_jump,0:Ny:y_jump,0:Nx:x_jump,:]
            v_re = v.reshape(Nz,Ny,Nx,Nt_j)[0:Nz:z_jump,0:Ny:y_jump,0:Nx:x_jump,:]
            w_re = w.reshape(Nz,Ny,Nx,Nt_j)[0:Nz:z_jump,0:Ny:y_jump,0:Nx:x_jump,:]
            p = p_re.reshape(-1,Nt_j)
            u = u_re.reshape(-1,Nt_j)
            v = v_re.reshape(-1,Nt_j)
            w = w_re.reshape(-1,Nt_j)
            
    return x, y, z, t, p, u, v, w

# POD分解，返回特征值、特征向量、模态的基函数和时间系数
def POD(UV_all):
    """2D Proper Orthogonal Decomposition
    Input UV stack matrix
    Return Eigenvalues,Eigenvectors,phi,ak_t """
    start_time = time.time()
    # mean-subtracted data
    UV = UV_all - np.mean(UV_all,axis=1).reshape(-1, 1)
    # 时间关联矩阵
    C = np.dot(UV.T, UV)
    print("时间关联矩阵的行列式的值是{}。C=0 means 可逆，非奇异矩阵".format(np.linalg.det(C)))  # C=0,可逆，非奇异矩阵
    # 求特征值
    Eigenvalues, Eigenvectors = np.linalg.eig(C)
    Eigenvalues = Eigenvalues.real
    Eigenvectors = Eigenvectors.real
    Eigenvalues = Eigenvalues.astype(np.float32)
    Eigenvectors = Eigenvectors.astype(np.float32)
    # 模态基函数
    phi = np.dot(UV, Eigenvectors) / np.sqrt(Eigenvalues)
    phi = phi.astype(np.float32)
    # 时间模态系数
    ak_t = np.dot(phi.T, UV)
    ak_t = ak_t.astype(np.float32)
    end_time = time.time()
    print("POD done, time cost:{}s.".format(end_time-start_time))
    return Eigenvalues, Eigenvectors, phi, ak_t

def POD_SVD(UV_all):
    """2D Proper Orthogonal Decomposition
    Input UV stack matrix
    Return Eigenvalues,Eigenvectors,phi,ak_t """
    start_time = time.time()
    # mean-subtracted data
    UV = UV_all - np.mean(UV_all,axis=1)
    # 时间关联矩阵
    # C = np.dot(UV.T, UV)
    # print("时间关联矩阵的行列式的值是{}。C=0 means 可逆，非奇异矩阵".format(np.linalg.det(C)))  # C=0,可逆，非奇异矩阵
    U, S, V = np.linalg.svd(UV, full_matrices=False)
    # 求特征值
    # Eigenvalues, Eigenvectors = np.linalg.eig(C)
    # Eigenvalues = Eigenvalues.real
    # Eigenvectors = Eigenvectors.real
    # Eigenvalues = Eigenvalues.astype(np.float32)
    # Eigenvectors = Eigenvectors.astype(np.float32)
    # 模态基函数
    # phi = np.dot(UV, Eigenvectors) / np.sqrt(Eigenvalues)
    phi = np.dot(np.dot(UV, V), S)
    phi = phi.astype(np.float32)
    # 时间模态系数
    ak_t = np.dot(phi.T, UV)
    ak_t = ak_t.astype(np.float32)
    end_time = time.time()
    print("POD done, time cost:{}s.".format(end_time-start_time))
    return phi, ak_t


# 计算模态
def Calc_Mode(u_test, v_test, Nx, Ny, phi, ak_t, n, t=0):
    """Calculate POD Modes
    Input Nx,Ny,phi,ak_t,n,t=0. 
    Nx/Ny is numbers of x/y points. 
    n is the orders will be calculated.
    t is the snaps be chosen, which is default 0.
    Return list Mode_u,Mode_v"""
    Mode_u = list()
    Mode_v = list()
    Mode_u.append(u_test[:, t].reshape(Ny, Nx))
    Mode_v.append(v_test[:, t].reshape(Ny, Nx))
    #     mode = np.zeros_like(UV)
    for i in range(n):
        # phi的列乘以ak_t的行
        mode = np.matmul(np.expand_dims(phi[:, i], axis=1), np.expand_dims(ak_t[i, :], axis=0))
        mode_u = mode[:Ny * Nx, t].reshape(Ny, Nx)
        mode_v = mode[Ny * Nx:, t].reshape(Ny, Nx)
        Mode_u.append(mode_u)
        Mode_v.append(mode_v)
    print("Snap{}, order{} is done.".format(t,n))
    return Mode_u, Mode_v


# 模态可视化（无colorbar的要求）
def Plot_Mode(x_test, y_test, Nx, Ny, row, col, Mode_u, Mode_v,fig_num=0,Output=False,path="./Fig"):
    """Plot POD modes or origin UV field
    Input row,column,Mode
    Mode format is [u,v,Mode_u[0],Mode_v[0],Mode_u[1],Mode_v[1],...]
    """
    x_mesh = x_test.reshape(Ny, Nx)
    y_mesh = y_test.reshape(Ny, Nx)
    fig = plt.figure(figsize=(12 * col, 6 * row))

    Mode = list()
    for i in range(len(Mode_u)):
        Mode.append(Mode_u[i])
        Mode.append(Mode_v[i])

    for i in range(1, row * col + 1):
        ax = fig.add_subplot(row, col, i)
        ax = ax.contourf(x_mesh, y_mesh, Mode[i - 1], cmap='jet')

        cb = fig.colorbar(ax)
        cb.ax.tick_params(labelsize=24)  # 设置colorbar字号
        plt.tick_params(axis='both', labelsize=36)  # 设置坐标轴字号
    if Output == True:
        if os.path.exists(path) == False:
            os.makedirs(path)
        plt.savefig(path+"/Mode_t{}.png".format(fig_num))
        print("Mode_t{}.png is saved".format(fig_num))

# 模态可视化（每一阶的模态colorbar全局一致）
def Plot_Mode_global(x_all, y_all, Nx, Ny, u_all, v_all,order,t=0,Output=False,path="./Fig"):
    """
    Mode = {'Mode_u1':Mode_u
            }

    """
    # normal/bold 正常/粗体
    # Times New Roman/Arial/SimHei（中文）
    font_title = {'family': 'Arial', 'weight': 'bold', 'size': 44 }
    font_label = {'family': 'Arial', 'weight': 'bold', 'size': 30 }

    x_mesh = x_all.reshape(Ny, Nx)
    y_mesh = y_all.reshape(Ny, Nx)

    # POD
    Nt = u_all.shape[1]
    UV_all = np.vstack([u_all, v_all])
    eig, eigvect, phi, ak_t = POD(UV_all)
    mode_dict = {}
    for i in range(order):
        mode = np.matmul(np.expand_dims(phi[:, i], axis=1), np.expand_dims(ak_t[i, :], axis=0))
        mode_u = mode[:Ny * Nx,:].reshape(Ny, Nx, Nt) # (Ny, Nx, Nt)
        mode_v = mode[Ny * Nx:,:].reshape(Ny, Nx, Nt)
        mode_dict['mode_u{}'.format(i)] = mode_u
        mode_dict['mode_v{}'.format(i)] = mode_v
        mode_dict['mode_u{}_max'.format(i)] = np.max(mode_u)
        mode_dict['mode_u{}_min'.format(i)] = np.min(mode_u)
        mode_dict['mode_v{}_max'.format(i)] = np.max(mode_v)
        mode_dict['mode_v{}_min'.format(i)] = np.min(mode_v)

    # 绘图
    fig = plt.figure(dpi=150, figsize=(12 * 2, 6 * order))
    plt.rc('font', family='Arial')
    plt.rcParams["font.weight"] = "bold"  #
    plt.rcParams["axes.labelweight"] = "bold"

    for i in range(order):
        # u 子图
        print('mode_u{}'.format(i))
        ax11 = plt.subplot(order, 2, 2*i+1)
        min_u, max_u = mode_dict['mode_u{}_min'.format(i)], mode_dict['mode_u{}_max'.format(i)]
        du = np.abs(max_u - min_u)
        cb_lv_u = np.linspace(min_u, max_u, num=1000) # colobar level
        f11 = plt.contourf(x_mesh, y_mesh,mode_dict['mode_u{}'.format(i)][:,:,t],
                           cmap='jet', levels=cb_lv_u, extend='both')  # 绘图===================
        plt.tick_params(axis='both', labelsize=30)  # 坐标轴字号
        # colorbar
        cb11 = fig.colorbar(f11, pad=0.02)  # pad偏离坐标轴距离，anchor锚点
        cb11.locator = ticker.MaxNLocator(nbins=6)  # 颜色条最多刻度数
        cb11.ax.tick_params(labelsize=30)  # 颜色条字号
        #         cb11.ax.set_title('U-velocity',font_label, y=1.05, loc='left') # 颜色条的title
        for cb_tick in cb11.ax.get_yticklabels():  # list
            cb_tick.set_horizontalalignment('left')
            cb_tick.set_x(1)
        cb11.update_ticks()  # 更新颜色条刻度

        # v
        print('mode_v{}'.format(i))
        ax12 = plt.subplot(order, 2, 2 * i + 2)
        min_v, max_v = mode_dict['mode_v{}_min'.format(i)], mode_dict['mode_v{}_max'.format(i)]
        dv = np.abs(max_v - min_v)
        cb_lv_v = np.linspace(min_v, max_v, num=1000)
        f12 = plt.contourf(x_mesh, y_mesh, mode_dict['mode_v{}'.format(i)][:, :, t],
                           cmap='jet', levels=cb_lv_v, extend='both')
        plt.tick_params(axis='both', labelsize=30)  # 坐标轴字号
        # colorbar
        cb12 = fig.colorbar(f12, pad=0.02)  # pad偏离坐标轴距离，anchor锚点
        cb12.locator = ticker.MaxNLocator(nbins=6)
        cb12.ax.tick_params(labelsize=30)
        for cb_tick in cb12.ax.get_yticklabels():  # list
            cb_tick.set_horizontalalignment('left')
            cb_tick.set_x(1)
        cb12.update_ticks()

    plt.tight_layout()


# 对比预测模态
def compare_mode(x, y, Nx, Ny, order, u, u_pred, t,mode_u, mode_u_pred,Output=False,path="./Fig"):
    """对比
    """
    row = order + 1
    col = 3
    x_mesh = x.reshape(Ny, Nx)
    y_mesh = y.reshape(Ny, Nx)
    fig = plt.figure(figsize=(12 * col, 6 * row))
#     u_t = u[:,t:t+1].reshape(Ny, Nx)
#     u_pred_t = u_pred[:,t:t+1].reshape(Ny, Nx)
#     u_error = u_t - u_pred_t
#     Mode = [u_t, u_pred_t, u_error]
    Mode = []
    for i in range(len(mode_u)):
        mode_error = mode_u[i] - mode_u_pred[i]
        Mode.append(mode_u[i])
        Mode.append(mode_u_pred[i])
        Mode.append(mode_error)

    for i in range(0, row * col, 3):
        ax1 = fig.add_subplot(row, col, i+1)
        min_1 = np.min(Mode[i])
        max_1 = np.max(Mode[i])
        cb1_level = np.arange(min_1,max_1,0.01*np.abs(max_1-min_1))
        ax1 = ax1.contourf(x_mesh, y_mesh, Mode[i], cmap='jet',levels = cb1_level,extend = 'both')
        cb1 = fig.colorbar(ax1)
        cb1.ax.tick_params(labelsize=24)  # 设置colorbar字号
        plt.tick_params(axis='both', labelsize=36)  # 设置坐标轴字号
        
        # pred
        ax2 = fig.add_subplot(row, col, i+2)
        ax2 = ax2.contourf(x_mesh, y_mesh, Mode[i+1], cmap='jet',levels = cb1_level,extend = 'both')
        cb2 = fig.colorbar(ax2)
        cb2.ax.tick_params(labelsize=24)  # 设置colorbar字号
        plt.tick_params(axis='both', labelsize=36)  # 设置坐标轴字号
        
        # error
        ax3 = fig.add_subplot(row, col, i+3)
        min_3 = np.min(Mode[i+2])
        max_3 = np.max(Mode[i+2])
        cb3_level = np.arange(min_3,max_3,0.01*np.abs(max_3-min_3))
        ax3 = ax3.contourf(x_mesh, y_mesh, Mode[i+2], cmap='jet',levels = cb3_level,extend = 'both')
        cb3 = fig.colorbar(ax3)
        cb3.ax.tick_params(labelsize=24)  # 设置colorbar字号
        plt.tick_params(axis='both', labelsize=36)  # 设置坐标轴字号
        
    if Output == True:
        if os.path.exists(path) == False:
            os.makedirs(path)
        plt.savefig(path+"/Mode_t{}.png".format(t))
        print("Mode_t{}.png is saved".format(t))

def Out_plt(x_test, y_test, Nx, Ny, Mode_u, Mode_v):
    """输出模态为plt格式，一个plt文件包含一阶的u和v"""
    if os.path.exists("./POD_Mode") == False:
        os.makedirs("./POD_Mode")
    for i in range(len(Mode_u)):
        mode_plt = np.stack((x_test, y_test, Mode_u[i].flatten(), Mode_v[i].flatten()), axis=-1)
        header = "variables=x,y,u,v \nzone I={} J={}".format(Nx, Ny)
        np.savetxt("./POD_Mode/POD_mode{}.plt".format(i), mode_plt, delimiter=" ", header=header, comments='')

def Read_plt(path,Nx,Ny,Nt,dt,L,U):
    """读无量纲化的预测结果"""
    t_star = L/U
    t = np.arange(0,Nt*dt,dt, dtype='float32')/t_star
    p = np.zeros((Ny * Nx, Nt), dtype='float32')
    u = np.zeros((Ny * Nx, Nt), dtype='float32')
    v = np.zeros((Ny * Nx, Nt), dtype='float32')
    x = np.zeros((Ny * Nx, 1), dtype='float32')
    y = np.zeros((Ny * Nx, 1), dtype='float32')
    
    
    read_path = path.format(0)
    x = np.loadtxt(open(read_path, "rb"), delimiter=" ", skiprows=2, usecols=[0])
    y = np.loadtxt(open(read_path, "rb"), delimiter=" ", skiprows=2, usecols=[1])
    
    for i in range(Nt):
        read_path = path.format(i)
        u[:, i] = np.loadtxt(open(read_path, "rb"), delimiter=" ", skiprows=2, usecols=[2])
        v[:, i] = np.loadtxt(open(read_path, "rb"), delimiter=" ", skiprows=2, usecols=[3])
        p[:, i] = np.loadtxt(open(read_path, "rb"), delimiter=" ", skiprows=2, usecols=[4])
        print("Snap {} is done.".format(i))
    return x,y,t,p,u,v

# 单个流场的快速预览
def plot_single(x,y,Nx,Ny,scalar, dpi=40, figsize=(12, 6)):
    font_title = {'family': 'Arial', 'weight': 'bold', 'size': 44, }
    font_label = {'family': 'Arial', 'weight': 'bold', 'size': 30, }
    fig = plt.figure(dpi=dpi, figsize=figsize)
    # 设置字体格式
    plt.rc('font', family='Arial')
    plt.rcParams["font.weight"] = "bold"  #
    plt.rcParams["axes.labelweight"] = "bold"
    # 绘图坐标
    x_mesh = x.reshape(Ny, Nx)
    y_mesh = y.reshape(Ny, Nx)
    min_l = np.min(scalar)
    max_l = np.max(scalar)
    dl = np.abs(max_l - min_l)
    cb_level = np.arange(min_l, max_l, 0.001 * dl)

    # 绘图
    ax11 = plt.subplot(1, 1, 1)
    # ax11.set_title(plt_squ[i], font_title, y=1.05)  # 标题
    ax11.set_xlabel('x coordinate', font_label)
    ax11.set_ylabel('y coordinate', font_label)
    f11 = plt.contourf(x_mesh, y_mesh,
                       scalar.reshape(Ny, Nx),
                       cmap='jet', levels=cb_level, extend='both')
    plt.tick_params(axis='both', labelsize=30)  # 坐标轴字号

    # colorbar
    cb11 = fig.colorbar(f11, pad=0.02)  # pad偏离坐标轴距离，anchor锚点
    cb11.locator = ticker.MaxNLocator(nbins=6)  # 颜色条最多刻度数
    cb11.ax.tick_params(labelsize=30)  # 颜色条字号
    #         cb11.ax.set_title('U-velocity',font_label, y=1.05, loc='left') # 颜色条的title

    for cb_tick in cb11.ax.get_yticklabels():  # list
        cb_tick.set_horizontalalignment('right')  # ticks右对齐
        cb_tick.set_x(7)  # ticks右移

    cb11.update_ticks()  # 更新颜色条刻度
    plt.show()

# 在绘制单个流场的基础上，画出测点的散点图
def plot_smp(x, y, Nx, Ny, scalar, x_smp, y_smp, dpi=40, figsize=(12, 6)):
    font_title = {'family': 'Arial', 'weight': 'bold', 'size': 44, }
    font_label = {'family': 'Arial', 'weight': 'bold', 'size': 30, }
    fig = plt.figure(dpi=dpi, figsize=figsize)
    # 设置字体格式
    plt.rc('font', family='Arial')
    plt.rcParams["font.weight"] = "bold"  #
    plt.rcParams["axes.labelweight"] = "bold"
    # 绘图坐标
    x_mesh = x.reshape(Ny, Nx)
    y_mesh = y.reshape(Ny, Nx)
    min_l = np.min(scalar)
    max_l = np.max(scalar)
    dl = np.abs(max_l - min_l)
    cb_level = np.arange(min_l, max_l, 0.001 * dl)

    # 绘图
    ax11 = plt.subplot(1, 1, 1)
    # ax11.set_title(plt_squ[i], font_title, y=1.05)  # 标题
    ax11.set_xlabel('x coordinate', font_label)
    ax11.set_ylabel('y coordinate', font_label)
    f11 = plt.contourf(x_mesh, y_mesh,
                       scalar.reshape(Ny, Nx),
                       cmap='jet', levels=cb_level, extend='both')
    plt.tick_params(axis='both', labelsize=30)  # 坐标轴字号

    plt.scatter(x_smp.flatten(), y_smp.flatten(), c='black', marker='x')

    # colorbar
    cb11 = fig.colorbar(f11, pad=0.02)  # pad偏离坐标轴距离，anchor锚点
    cb11.locator = ticker.MaxNLocator(nbins=6)  # 颜色条最多刻度数
    cb11.ax.tick_params(labelsize=30)  # 颜色条字号
    #         cb11.ax.set_title('U-velocity',font_label, y=1.05, loc='left') # 颜色条的title

    for cb_tick in cb11.ax.get_yticklabels():  # list
        cb_tick.set_horizontalalignment('right')  # ticks右对齐
        cb_tick.set_x(7)  # ticks右移

    cb11.update_ticks()  # 更新颜色条刻度
    plt.show()

# 预测结果的绘图与对比单个Snap的3x3图
def plot_3x3_dm(Nx,Ny,t,**scalar):
    """
    绘制第t个时刻的uvp的pinn回归与cfd计算及其误差
    传入原始的xyuvp:
    xy:(Nx*Ny,1)
    uvp:(Nx*Ny,Nt)

    **scalar以字典形式传入
    dict={'x':x,
          'y':y,
          'u_cfd','v_cfd','p_cfd',
          'u_pinn','v_pinn','p_pinn',
          'wall':True,'cor_wall',
          'u_error','v_error','p_error'
    """

    # normal/bold 正常/粗体 
    # Times New Roman/Arial/SimHei（中文）
    font_title = {'family' : 'Arial',
             'weight' : 'bold',
             'size'   : 44,
             }

    font_label = {'family' : 'Arial',
             'weight' : 'bold',
             'size'   : 30,
             }
    row = 3
    col = 3
    fig = plt.figure(dpi=150,figsize=(12 * col, 6 * row))
    # 设置字体格式
    plt.rc('font', family='Arial')
    plt.rcParams["font.weight"] = "bold" # 
    plt.rcParams["axes.labelweight"] = "bold"
    
    # 绘图坐标
    x,y = scalar['x'],scalar['y']
    x_mesh = x.reshape(Ny, Nx)
    y_mesh = y.reshape(Ny, Nx)
    scalar['u_error'] = scalar['u_pinn']-scalar['u_cfd']
    scalar['v_error'] = scalar['v_pinn']-scalar['v_cfd']
    scalar['p_error'] = scalar['p_pinn']-scalar['p_cfd']
    

    # 绘制变量的顺序
    plt_squ = ['u_cfd','u_pinn','u_error',
               'v_cfd','v_pinn','v_error',
               'p_cfd','p_pinn','p_error']

    if len(plt_squ) == 3:
        min_l = np.min(scalar[plt_squ[0]][:,t])
        max_l = np.max(scalar[plt_squ[0]][:,t])
        dl = np.abs(max_l-min_l)
        cb1_level = np.arange(min_l,max_l,0.01*dl)
    elif len(plt_squ) == 6:
        min_l1 = np.min(scalar[plt_squ[0]][:,t])
        max_l1 = np.max(scalar[plt_squ[0]][:,t])
        dl1 = np.abs(max_l1-min_l1)
        cb1_level = np.arange(min_l1,max_l1,0.01*dl1)
        
        min_l2 = np.min(scalar[plt_squ[3]][:,t])
        max_l2 = np.max(scalar[plt_squ[3]][:,t])
        dl2 = np.abs(max_l2-min_l2)
        cb2_level = np.arange(min_l2,max_l2,0.01*dl2)

    elif len(plt_squ) == 9:
        min_l1 = np.min(scalar[plt_squ[0]][:,t])
        max_l1 = np.max(scalar[plt_squ[0]][:,t])
        dl1 = np.abs(max_l1-min_l1)
        cb1_level = np.arange(min_l1,max_l1,0.01*dl1)
        
        min_l2 = np.min(scalar[plt_squ[3]][:,t])
        max_l2 = np.max(scalar[plt_squ[3]][:,t])
        dl2 = np.abs(max_l2-min_l2)
        cb2_level = np.arange(min_l2,max_l2,0.01*dl2)
        
        min_l3 = np.min(scalar[plt_squ[6]][:,t])
        max_l3 = np.max(scalar[plt_squ[6]][:,t])
        dl3 = np.abs(max_l3-min_l3)
        cb3_level = np.arange(min_l3,max_l3,0.01*dl3)

    else:
        print("len(plt_squ) is not 3/6/9")
    
    for i in range(col*row):
        # 子图
        plt_var = scalar[plt_squ[i]][:,t] # 从字典获取要绘制的scalar
        ax11 = plt.subplot(row,col,i+1)
        ax11.set_title(plt_squ[i],font_title, y=1.05) # 标题
        ax11.set_xlabel('x coordinate',font_label)
        ax11.set_ylabel('y coordinate',font_label)
        print("第{}个图{}".format(i+1,plt_squ[i]),end='')
        # 标量
        if((i+1 == 1) or (i+1 == 2)):
            cb_level = cb1_level
            print("使用level_u")
        elif((i+1 == 4) or (i+1 == 5)):
            cb_level = cb2_level
            print("使用level_v")
        elif((i+1 == 7) or (i+1 == 8)):
            cb_level = cb3_level
            print("使用level_p")
        else:
            min_var = np.min(plt_var)
            max_var = np.max(plt_var)
            dl_var = np.abs(max_var-min_var)
            cb_level = np.arange(min_var,max_var,0.01*dl_var)
            print("使用{}的(min,max)".format(plt_squ[i]))
        f11 = plt.contourf(x_mesh, y_mesh, 
                           plt_var.reshape(Ny, Nx), 
                           cmap='jet',levels=cb_level,extend='both')
        plt.tick_params(axis='both', labelsize=30)  # 坐标轴字号
        
        # colorbar
        cb11 = fig.colorbar(f11, pad=0.02) # pad偏离坐标轴距离，anchor锚点
        cb11.locator = ticker.MaxNLocator(nbins=6) # 颜色条最多刻度数
        cb11.ax.tick_params(labelsize=30) # 颜色条字号
#         cb11.ax.set_title('U-velocity',font_label, y=1.05, loc='left') # 颜色条的title
        
        for cb_tick in cb11.ax.get_yticklabels(): # list
            cb_tick.set_horizontalalignment('right') # ticks右对齐
            cb_tick.set_x(7) # ticks右移
        
        cb11.update_ticks() # 更新颜色条刻度
        
        # 填充壁面
        if scalar['wall']==True:
            cor_wall = scalar['cor_wall']
            plt.fill(cor_wall[:,0],cor_wall[:,1], facecolor='white',alpha=1) # fill wall BC 
    plt.tight_layout()
    # plt.show()
    if scalar['save']:
        plt.savefig(scalar['save_path'].format(t))

# 预测结果的绘图与对比单个Snap的3x3图
def plot_3x3_dm_global(Nx, Ny, t, **scalar):
    """
    绘制第t个时刻的uvp的pinn回归与cfd计算及其误差,colorbar的数值全局一致

    **scalar以字典形式传入
    dict={ 'x':x_all,
           'y':y_all,
           'u_cfd':u_avg_tile,
           'v_cfd':v_avg_tile,
           'p_cfd':p_avg_tile,
           'u_pinn':u_pred_avg_tile,
           'v_pinn':v_pred_avg_tile,
           'p_pinn':p_pred_avg_tile,
           'wall':False,
           'save':False,
           'save_path':'Z:/Re2NB{}.png',
           'plt_squ':plt_squ}
    plt_squ = [ 'p_cfd','p_pinn','p_error',
            'u_cfd','u_pinn','u_error',
            'v_cfd','v_pinn','v_error']
    """

    # normal/bold 正常/粗体
    # Times New Roman/Arial/SimHei（中文）
    font_title = {'family': 'Arial',
                  'weight': 'bold',
                  'size': 44,
                  }

    font_label = {'family': 'Arial',
                  'weight': 'bold',
                  'size': 30,
                  }
    # 绘制变量的顺序
    plt_squ = scalar['plt_squ']

    row = int(len(plt_squ) / 3)
    col = 3
    fig = plt.figure(dpi=150, figsize=(12 * col, 6 * row))
    # 设置字体格式
    plt.rc('font', family='Arial')
    plt.rcParams["font.weight"] = "bold"  #
    plt.rcParams["axes.labelweight"] = "bold"

    # 绘图坐标
    x, y = scalar['x'], scalar['y']
    x_mesh = x.reshape(Ny, Nx)
    y_mesh = y.reshape(Ny, Nx)
    scalar['u_error'] = scalar['u_pinn'] - scalar['u_cfd']
    scalar['v_error'] = scalar['v_pinn'] - scalar['v_cfd']
    scalar['p_error'] = scalar['p_pinn'] - scalar['p_cfd']

    if len(plt_squ) == 3:
        min_l = np.min(scalar[plt_squ[0]])
        max_l = np.max(scalar[plt_squ[0]])
        dl = np.abs(max_l - min_l)
        cb1_level = np.around(np.arange(min_l, max_l, 0.0005 * dl), 5)  # u_cfd和u_pinn的colorbar

        min_e1 = np.min(scalar[plt_squ[2]])
        max_e1 = np.max(scalar[plt_squ[2]])
        de1 = np.abs(max_e1 - min_e1)
        cb_e1_level = np.around(np.arange(min_e1, max_e1, 0.0005 * de1), 5)

    elif len(plt_squ) == 6:
        min_l1 = np.min(scalar[plt_squ[0]])
        max_l1 = np.max(scalar[plt_squ[0]])
        dl1 = np.abs(max_l1 - min_l1)
        cb1_level = np.around(np.arange(min_l1, max_l1, 0.0005 * dl1), 5)

        min_e1 = np.min(scalar[plt_squ[2]])
        max_e1 = np.max(scalar[plt_squ[2]])
        de1 = np.abs(max_e1 - min_e1)
        cb_e1_level = np.around(np.arange(min_e1, max_e1, 0.0005 * de1), 5)

        min_l2 = np.min(scalar[plt_squ[3]])
        max_l2 = np.max(scalar[plt_squ[3]])
        dl2 = np.abs(max_l2 - min_l2)
        cb2_level = np.around(np.arange(min_l2, max_l2, 0.0005 * dl2), 5)

        min_e2 = np.min(scalar[plt_squ[5]])
        max_e2 = np.max(scalar[plt_squ[5]])
        de2 = np.abs(max_e2 - min_e2)
        cb_e2_level = np.around(np.arange(min_e2, max_e2, 0.0005 * de2), 5)

    elif len(plt_squ) == 9:
        min_l1 = np.min(scalar[plt_squ[0]])
        max_l1 = np.max(scalar[plt_squ[0]])
        dl1 = np.abs(max_l1 - min_l1)
        cb1_level = np.around(np.arange(min_l1, max_l1, 0.0005 * dl1), 5)

        min_e1 = np.min(scalar[plt_squ[2]])
        max_e1 = np.max(scalar[plt_squ[2]])
        de1 = np.abs(max_e1 - min_e1)
        cb_e1_level = np.around(np.arange(min_e1, max_e1, 0.0005 * de1), 5)

        min_l2 = np.min(scalar[plt_squ[3]])
        max_l2 = np.max(scalar[plt_squ[3]])
        dl2 = np.abs(max_l2 - min_l2)
        cb2_level = np.around(np.arange(min_l2, max_l2, 0.0005 * dl2), 5)

        min_e2 = np.min(scalar[plt_squ[5]])
        max_e2 = np.max(scalar[plt_squ[5]])
        de2 = np.abs(max_e2 - min_e2)
        cb_e2_level = np.around(np.arange(min_e2, max_e2, 0.0005 * de2), 5)

        min_l3 = np.min(scalar[plt_squ[6]][:, t])
        max_l3 = np.max(scalar[plt_squ[6]][:, t])
        dl3 = np.abs(max_l3 - min_l3)
        cb3_level = np.around(np.arange(min_l3, max_l3, 0.0005 * dl3), 5)

        min_e3 = np.min(scalar[plt_squ[5]])
        max_e3 = np.max(scalar[plt_squ[5]])
        de3 = np.abs(max_e3 - min_e3)
        cb_e3_level = np.around(np.arange(min_e3, max_e3, 0.0005 * de3), 5)

    else:
        print("len(plt_squ) is not 3/6/9")

    for i in range(col * row):
        # 子图
        print(plt_squ[i])
        plt_var = scalar[plt_squ[i]][:, t]  # 从字典获取要绘制的scalar
        ax11 = plt.subplot(row, col, i + 1)
        ax11.set_title(plt_squ[i], font_title, y=1.05)  # 标题
        ax11.set_xlabel('x', font_label)
        ax11.set_ylabel('y', font_label)
        print("第{}个图{}".format(i + 1, plt_squ[i]), end='')
        # 标量
        if ((i + 1 == 1) or (i + 1 == 2)):  # u_cfd和u_pinn的colorbar
            cb_level = cb1_level
            print("使用level_u_global")
        elif ((i + 1 == 3)):  # u_error的colorbar
            cb_level = cb_e1_level
            print("使用level_e1_global")
        elif ((i + 1 == 4) or (i + 1 == 5)):  # v_cfd和v_pinn的colorbar
            cb_level = cb2_level
            print("使用level_v_global")
        elif ((i + 1 == 6)):  # v_error的colorbar
            cb_level = cb_e2_level
            print("使用level_e1_global")
        elif ((i + 1 == 7) or (i + 1 == 8)):
            cb_level = cb3_level
            print("使用level_p_global")
        elif ((i + 1 == 9)):  # p_error的colorbar
            cb_level = cb_e3_level
            print("使用level_e3_global")
        f11 = plt.contourf(x_mesh, y_mesh,
                           plt_var.reshape(Ny, Nx),
                           cmap='jet', levels=cb_level, extend='both')  # 绘图===================
        plt.tick_params(axis='both', labelsize=30)  # 坐标轴字号

        # colorbar
        cb11 = fig.colorbar(f11, pad=0.02)  # pad偏离坐标轴距离，anchor锚点
        cb11.locator = ticker.MaxNLocator(nbins=6)  # 颜色条最多刻度数
        cb11.ax.tick_params(labelsize=30)  # 颜色条字号
        #         cb11.ax.set_title('U-velocity',font_label, y=1.05, loc='left') # 颜色条的title
        # add 20220816 =========================
        for cb_tick in cb11.ax.get_yticklabels():  # list
            cb_tick.set_horizontalalignment('right')
            cb_tick.set_x(7)
        # add 20220816 =========================
        cb11.update_ticks()  # 更新颜色条刻度

        # 填充壁面
        if scalar['wall'] == True:
            cor_wall = scalar['cor_wall']
            plt.fill(cor_wall[:, 0], cor_wall[:, 1], facecolor='white', alpha=1)  # fill wall BC
    plt.tight_layout()
    # plt.show()
    if scalar['save']:
        plt.savefig(scalar['save_path'].format(t))

def Re_L2_error(u,v,p,u_pred,v_pred,p_pred,**save):
    N = u.shape[0]
    T = u.shape[1]
    u_Re_L2 = np.zeros(T)
    v_Re_L2 = np.zeros(T)
    p_Re_L2 = np.zeros(T)
    for i in range(T):
        u_mean = u[:,i].mean()
        v_mean = v[:,i].mean()
        p_mean = p[:,i].mean()
        
        u_up = np.sum((u_pred[:,i]-u[:,i])**2) # up和down同时除以N，省略
        v_up = np.sum((v_pred[:,i]-v[:,i])**2)
        p_up = np.sum((p_pred[:,i]-p[:,i])**2)
        
        u_down = np.sum((u[:,i]-u_mean)**2)
        v_down = np.sum((v[:,i]-v_mean)**2)
        p_down = np.sum((p[:,i]-p_mean)**2)
        
        u_Re_L2[i] = u_up/u_down
        v_Re_L2[i] = v_up/v_down
        p_Re_L2[i] = p_up/p_down
    if save['save']:
        # 绘图
        plt.rc('font', family='Arial')
        plt.rcParams["font.weight"] = "bold" # 
        plt.rcParams["axes.labelweight"] = "bold"
        fig = plt.figure(figsize=(12, 9))
        line_u, = plt.plot(np.arange(0,T,1),u_Re_L2)
        line_v, = plt.plot(np.arange(0,T,1),v_Re_L2)
        line_p, = plt.plot(np.arange(0,T,1),p_Re_L2)
        plt.legend(handles=[line_u,line_v,line_p],
                labels=["u","v","p"],
                loc='best',fontsize=20,
                frameon=False)
        # plt.legend(handles=[line_p],
        #            labels=["p"],
        #            loc='best',fontsize=20,
        #            frameon=False)
        # plt.xscale("log")
        plt.yscale("log")
        plt.xlabel('snap', fontsize=20)
        plt.ylabel('L2 error', fontsize=20)
        plt.tick_params(labelsize=14)
        plt.savefig(save['save_path'])

    return u_Re_L2,v_Re_L2,p_Re_L2

def Re_L2_error_3D(u,v,w,p,u_pred,v_pred,w_pred,p_pred,**save):
    N = u.shape[0]
    T = u.shape[1]
    u_Re_L2 = np.zeros(T)
    v_Re_L2 = np.zeros(T)
    w_Re_L2 = np.zeros(T)
    p_Re_L2 = np.zeros(T)
    for i in range(T):
        u_mean = u[:,i].mean()
        v_mean = v[:,i].mean()
        w_mean = w[:,i].mean()
        p_mean = p[:,i].mean()
        
        u_up = np.sum((u_pred[:,i]-u[:,i])**2) # up和down同时除以N，省略
        v_up = np.sum((v_pred[:,i]-v[:,i])**2)
        w_up = np.sum((w_pred[:,i]-w[:,i])**2)
        p_up = np.sum((p_pred[:,i]-p[:,i])**2)
        
        u_down = np.sum((u[:,i]-u_mean)**2)
        v_down = np.sum((v[:,i]-v_mean)**2)
        w_down = np.sum((w[:,i]-w_mean)**2)
        p_down = np.sum((p[:,i]-p_mean)**2)
        
        u_Re_L2[i] = u_up/u_down
        v_Re_L2[i] = v_up/v_down
        w_Re_L2[i] = w_up/w_down
        p_Re_L2[i] = p_up/p_down
    if save['save']:
        # 绘图
        plt.rc('font', family='Arial')
        plt.rcParams["font.weight"] = "bold" # 
        plt.rcParams["axes.labelweight"] = "bold"
        fig = plt.figure(figsize=(12, 9))
        line_u, = plt.plot(np.arange(0,T,1),u_Re_L2)
        line_v, = plt.plot(np.arange(0,T,1),v_Re_L2)
        line_w, = plt.plot(np.arange(0,T,1),w_Re_L2)
        line_p, = plt.plot(np.arange(0,T,1),p_Re_L2)
        plt.legend(handles=[line_u,line_v,line_w,line_p],
                labels=["u","v","w","p"],
                loc='best',fontsize=20,
                frameon=False)
        # plt.legend(handles=[line_p],
        #            labels=["p"],
        #            loc='best',fontsize=20,
        #            frameon=False)
        # plt.xscale("log")
        plt.yscale("log")
        plt.xlabel('snap', fontsize=20)
        plt.ylabel('L2 error', fontsize=20)
        plt.tick_params(labelsize=14)
        plt.savefig(save['save_path'])

    return u_Re_L2,v_Re_L2,w_Re_L2,p_Re_L2


# 模态可视化（colorbar按等级给值）
def Plot_Mode_samelevel(x_mesh, y_mesh, row, col, Mode, u_level, v_level):
    """Plot POD modes or origin UV field
    Input row,column,Mode,u_level,v_level
    Mode format is [u,v,Mode_u[i],Mode_v[i]]
    u/v_level is np.arange(min,max,delat)
    """
    #     u_delta =  (np.max(Mode[0])-np.min(Mode[0]))/10
    #     v_delta =  (np.max(Mode[1])-np.min(Mode[1]))/10
    #     u_level = np.arange(np.min(Mode[0])-2*u_delta,np.max(Mode[0])+2*u_delta,u_delta)
    #     v_level = np.arange(np.min(Mode[1])-2*v_delta,np.max(Mode[1])+2*v_delta,v_delta)
    fig = plt.figure(figsize=(12 * col, 6 * row))
    #     norm1 = plc.Normalize(vmin=-4,vmax=16)
    #     norm2 = plc.Normalize(vmin=-8,vmax=8)
    #     u_level = np.arange(-4,18,2)
    #     v_level = np.arange(-8,10,2)
    for i in range(1, row * col + 1):
        ax = fig.add_subplot(row, col, i)
        #         ax = ax.contourf(x_mesh,y_mesh,Mode[i-1],10,cmap='jet') # 调colorbar之前的
        if i % 2 == 0:
            # ax = ax.contourf(x_mesh,y_mesh,Mode[i-1],cmap=plt.cm.coolwarm,levels = level2)
            ax = ax.contourf(x_mesh, y_mesh, Mode[i - 1], cmap='jet', levels=v_level)  # 统一colorbar
            # ax = ax.contourf(x_mesh,y_mesh,Mode[i-1],cmap='jet')
        else:
            ax = ax.contourf(x_mesh, y_mesh, Mode[i - 1], cmap='jet', levels=u_level)  # 统一colorbar
            # ax = ax.contourf(x_mesh,y_mesh,Mode[i-1],cmap='jet')

        cb = fig.colorbar(ax)
        cb.ax.tick_params(labelsize=24)
        plt.tick_params(axis='both', labelsize=36)  # 设置坐标轴格式


    # plt.savefig(".Fig/Mode_t{%2d}".format())



# 傅里叶变换的绘图
def FFT_plot(row, col, point_num, delta_t, y, L, U):
    fig = plt.figure(dpi=100,figsize=(10 * col, 6 * row))
    fft_x_o = fft.rfftfreq(n=point_num, d=delta_t)
    for i in range(1, row * col + 1):
        fft_y_o = np.abs(fft.rfft(y[i - 1, :]))
        if fft_y_o.argmax() == 0:
            fft_x = fft_x_o[1:]
            fft_y = fft_y_o[1:]
            ax = fig.add_subplot(row, col, i)
            ax = ax.plot(fft_x, fft_y,linewidth=3)
            f_max = fft_x[fft_y.argmax()]
            plt.title("Cut f=0, f_max={:.3f}Hz, St=Lf/U={:.3f}".format(f_max, L * f_max / U),fontsize=20)
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel('frequency', fontsize=20)
            plt.ylabel('amplitude', fontsize=20)
            plt.tick_params(axis='both', labelsize=20)  # 设置坐标轴字号
        else:
            ax = fig.add_subplot(row, col, i)
            ax = ax.plot(fft_x_o, fft_y_o,linewidth=3)
            f_max = fft_x_o[fft_y_o.argmax()]
            plt.title("No Cut f, f_max={:.3f}Hz, St=Lf/U={:.3f}".format(f_max, L * f_max / U),fontsize=20)
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel('frequency', fontsize=20)
            plt.ylabel('amplitude', fontsize=20)
            plt.tick_params(axis='both', labelsize=20)  # 设置坐标轴字号
    plt.tight_layout()
    plt.show()

def FFT_plot_compare(row, col, point_num, delta_t, y,y_pred, L, U):
    fig = plt.figure(dpi=100,figsize=(10 * col, 6 * row))
    fft_x_o = fft.rfftfreq(n=point_num, d=delta_t)
    for i in range(1, row * col + 1):
        fft_y_o = np.abs(fft.rfft(y[i - 1, :]))
        fft_y_o_pred = np.abs(fft.rfft(y_pred[i - 1, :]))
        if fft_y_o.argmax() == 0:
            fft_x = fft_x_o[1:]
            fft_y = fft_y_o[1:]
            fft_y_pred = fft_y_o_pred[1:]
            ax = fig.add_subplot(row, col, i)
            line_r, = ax.plot(fft_x, fft_y,linewidth=3)
            line_p, = ax.plot(fft_x, fft_y_pred,linewidth=3)
            plt.legend(handles=[line_r,line_p],
                       labels=["POD","PINN-POD"],
                       loc='best',fontsize=20,
                       frameon=False)
            f_max = fft_x[fft_y.argmax()]
            f_max_pred = fft_x[fft_y_pred.argmax()]
            plt.title("f_max={:.3f}Hz, St=Lf/U={:.3f},\n f_max_pred={:.3f}Hz, St_pred=Lf/U={:.3f}".format(
                        f_max, L * f_max / U,f_max_pred, L * f_max_pred / U),fontsize=20)
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel('frequency', fontsize=20)
            plt.ylabel('amplitude', fontsize=20)
            plt.tick_params(axis='both', labelsize=20)  # 设置坐标轴字号
        else:
            ax = fig.add_subplot(row, col, i)
            line_r, = ax.plot(fft_x_o, fft_y_o,linewidth=3)
            line_p, = ax.plot(fft_x_o, fft_y_o_pred,linewidth=3)
            plt.legend(handles=[line_r,line_p],
                       labels=["POD","PINN-POD"],
                       loc='best',fontsize=20,
                       frameon=False)
            f_max = fft_x_o[fft_y_o.argmax()]
            f_max_pred = fft_x_o[fft_y_o_pred.argmax()]
            plt.title("No Cut f, f_max={:.3f}Hz, St=Lf/U={:.3f},\n f_max_pred={:.3f}Hz, St_pred=Lf/U={:.3f}".format(
                        f_max, L * f_max / U, f_max_pred, L * f_max_pred / U),fontsize=20)
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel('frequency', fontsize=20)
            plt.ylabel('amplitude', fontsize=20)
            plt.tick_params(axis='both', labelsize=20)  # 设置坐标轴字号
    plt.tight_layout()
    plt.show()

# 傅里叶变换的输出
def FFT_Out(point_num, delta_t, y, L, U):
    """point_num is time-step,delta_t,y
    """
    fft_y_o = np.abs(fft.rfft(y))
    fft_x_o = fft.rfftfreq(n=point_num, d=delta_t)  # 返回FFT频率
    if fft_y_o.argmax() == 0:
        fft_x = fft_x_o[1:]
        fft_y = fft_y_o[1:]
        print("Abandon frequency=0")
        print("MAX fft_amp is {}".format(fft_y.max()))
        print("Num of max fft_amp is {}".format(fft_y.argmax()))
        print("Main frequency is {}".format(fft_x[fft_y.argmax()]))
        print("St = Lf/U = {}".format(L * fft_x[fft_y.argmax()] / U))
    else:
        St = L * fft_x_o[fft_y_o.argmax()] / U
        print("MAX fft_amp is {}".format(fft_y_o.max()))
        print("Num of max fft_amp is {}".format(fft_y_o.argmax()))
        print("Main frequency is {}".format(fft_x_o[fft_y_o.argmax()]))
        print("St = Lf/U = {}".format(St))
    return 

# 根据mat数据生成csv数据
def mat_to_csv_2D(csv_path,Nx,Ny,dt,is_pred = True,**data):
    if is_pred == True:
        x_all, y_all, t_all, p_all, u_all, v_all = data['x_pred'],data['y_pred'],data['t_pred'],data['p_pred'],data['u_pred'],data['v_pred']
        out_path = csv_path+'/pred-{}.csv'
    else:
        x_all, y_all, t_all, p_all, u_all, v_all = data['x'],data['y'],data['t'],data['p'],data['u'],data['v']
        out_path = csv_path + '/cfd-{}.csv'
    if os.path.exists(csv_path) == False:
        os.makedirs(csv_path)
    for i in range(len(t_all)):
    # for i in range(10): # 调试输出10个
        csv_data = np.hstack([x_all,y_all,p_all[:,i:i+1],u_all[:,i:i+1],v_all[:,i:i+1]])
        header = 'x,y,p,u,v'
        t = i*dt
#         print(csv_path+'/pred-{:0>5.2f}.csv'.format(t))
#         print(csv_path+'/pred-{}.csv'.format(i))
        np.savetxt(out_path.format(i), csv_data, delimiter=",", header=header, comments='')
        print("Snap {} Done".format(i))




