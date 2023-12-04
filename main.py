"""
@author: Chang, responsible for DOI:https://doi.org/10.1063/5.0138287
"""

import subprocess
import numpy as np
import PyPOD
import time
import h5py

# gpu_process/cpu_process depends on your HPC properties.
def gpu_process(args_tpi, tp_idx):
    with open("run_tp{}.sh".format(tp_idx), "w", newline='\n') as file:
        file.write("#!/bin/bash\n")
        file.write("#SBATCH -J tp{}\n".format(tp_idx))
        file.write("#SBATCH -N 1\n")
        file.write("#SBATCH --gres=dcu:1\n")
        file.write("#SBATCH -p wzhdnormal\n") # GPU Nodes
        file.write("#SBATCH --exclusive\n")
        file.write("source ~/.bashrc\n")
        file.write("module unload compiler/dtk/21.10\n")
        file.write("module load compiler/dtk/22.10\n")
        file.write("conda activate tensorflow-py37\n")
        file.write("python -u sub_train.py " + args_tpi)
    subprocess.call(['chmod', '+x', './run_tp{}.sh'.format(tp_idx)])
    subprocess.call(['sbatch', './run_tp{}.sh'.format(tp_idx)])

def cpu_process(script_name, other_args, log_file):
    with open("run_tp{}.sh".format(tp_idx), "w", newline='\n') as file:
        file.write("#!/bin/bash\n")
        file.write("#SBATCH -J tp{}\n".format(tp_idx))
        file.write("#SBATCH -N 1\n")
        file.write("#SBATCH -p wzhcnormal\n") # CPU Nodes
        file.write("#SBATCH --exclusive\n")
        file.write("source ~/.bashrc\n")
        file.write("module unload compiler/dtk/21.10\n")
        file.write("module load compiler/dtk/22.10\n")
        file.write("conda activate tensorflow-py37\n")
        file.write("python -u sub_train.py " + args_tpi)
    subprocess.call(['chmod', '+x', './run_tp{}.sh'.format(tp_idx)])
    subprocess.call(['sbatch', './run_tp{}.sh'.format(tp_idx)])

def Calc_Cni(file_paths, order = 6, N=3):
    try:
        u_data = []
        v_data = []
        phi = []
        eps = 0
        for path in file_paths:
            with h5py.File(path, 'r') as file:
                u_data.append(np.array(file['u_pred']))
                v_data.append(np.array(file['v_pred']))
        u_pred = np.hstack(u_data)
        v_pred = np.hstack(v_data)
        print("********* Compulting POD *********")
        snaps = u_pred.shape[1]
        print("Total snaps is ",snaps)
        for snap in range(100, snaps+1, 100):
            UV_pred = np.vstack((u_pred[:,0:snap],v_pred[:,0:snap]))
            Eigenvalues_pred,Eigenvectors_pred,phi_pred,ak_t_pred = PyPOD.POD(UV_pred)
            phi.append(phi_pred[:,0:order+1])
            del Eigenvalues_pred,Eigenvectors_pred,ak_t_pred
        phi_base = phi[-1] # [x,order]

        for n in range(N):
            print("Calculate scalar product of snap{} and snap{} ".format(
                snaps,snaps+100*(n-N-1)))

            for i in range(order): # 3?
                scalar_product = np.abs(np.dot(phi[n-N-1][:,i],phi_base[:,i])) # n-N: -4,-3,-2
                print("{} order's scalar product of phi_base in snap{} and snap{} is {}".format(
                        i,snaps,snaps+100*(n-N-1),scalar_product))
                eps += scalar_product
        Cni = 1 - eps/(N*order)
        return Cni
    except FileNotFoundError:
        for path in file_paths:
            print(path," may not exist.")
        return "wait"

def Calc_Loss(file_paths):
    try:
        Loss = []
        for path in file_paths: # signale_Loss_0_T0
            with open(path, "r") as f:
                Loss.append(float(f.read()))
        print("********* Compulting average Loss *********")
        Loss_avg = np.mean(np.array(Loss))
        return Loss_avg
    except FileNotFoundError:
        print("No Loss File Until Now!!!!!")
        return "wait"

def send_signal_Cni(signal_Cni, index, T_i):
    with open(f'signal_Cni_{index}_T{T_i}.txt', 'w') as f:
        f.write(signal_Cni)
if __name__ == "__main__":
    Cni_converge = 1e-2
    Loss_converge = 1e-3
    UseGPU = True
    mainDebug = False
    subDebug = 0 # 0=False,1=True
    if UseGPU:
        sub_process = gpu_process
    else:
        sub_process = cpu_process
    # 定义每个脚本的参数
    numSubNN = 10
    for tp_idx in range(numSubNN):
        args_tpi = "--debug={} --index={} --tp1={} --tp2={}".format(subDebug,
                                                                     tp_idx,
                                                                     tp_idx*100,
                                                                     (tp_idx+1)*100+20)
        sub_process(args_tpi, tp_idx)

    tm = 2
    dn_max = 6
    if subDebug==1:
        ep = 10
    else:
        ep = 1000
    np_ep = np.zeros(dn_max) + ep
    for i in range(1, dn_max):
        ep =  tm * ep
        np_ep[i] = int(ep)
    Loss = np.zeros(dn_max)
    for T_i in range(np_ep.shape[0]):
        print("********** Period {} monitoring **********".format(T_i))
        nextPeriod = 0
        file_paths = []
        if mainDebug:
            for i in range(0,numSubNN,1):
                file_paths.append("./tp{}-{}_dt0.1s_X-x29p_t{}-{}_test".format(i*100+10, ((i+1)*100+20)-10, i*100, ((i+1)*100+20))
                                  + "/Pred_ep{}.h5".format(int(np_ep[T_i])))
        else:
            for i in range(0,numSubNN,1):
                file_paths.append("./tp{}-{}_dt0.1s_X-x29p_t{}-{}".format(i*100+10, ((i+1)*100+20)-10, i*100, ((i+1)*100+20))
                                  + "/Pred_ep{}.h5".format(int(np_ep[T_i])))
        loss_path = []
        for inx in range(numSubNN):
            loss_path.append('signal_Loss_{}_T{}.txt'.format(inx,T_i))

        while nextPeriod==0:
            Cni = Calc_Cni(file_paths, order = 6, N=3)
            loss_Ti = Calc_Loss(loss_path)
            print("=======Refresh: Cni is ",Cni,", loss_T",T_i," is ",loss_Ti)

            if(str(loss_Ti) != "wait"):
                # Case 0 T0已完成, 更新Loss数值
                print("Loss T", T_i, " is ", loss_Ti)
                Loss[T_i] = float(loss_Ti)
                # Loss.append(float(loss_Ti))
            
            if T_i == 0: # First Peroid, continue 第一个周期，直接给继续的信号，但是要等待计算完Cni和Loss才进入下一周期
                if (str(Cni) != "wait") and (str(loss_Ti) != "wait"):
                    # Case 0-1 T0已完成, 更新下一周期的Cni文件
                    print("*******Get Cni and loss_Ti*******")
                    print("Case 0-1 T", T_i, ": ", "pass. so start next peroid Cni_T",T_i+1,".")
                    for inx in range(numSubNN):
                        send_signal_Cni('T{}_continue'.format(T_i+1), inx, T_i+1)
                    nextPeriod = 1
                else:
                    print("Case 0-2 T", T_i, ": ", "waiting. Cni_T",T_i," is continue")
                    # Case 0-1 T0还没完成, 保持T0周期的Cni文件
                    for inx in range(numSubNN):
                        send_signal_Cni('T{}_continue'.format(T_i), inx, T_i)
                
            else: # The rest peroids, continue 剩余的周期，判断
                if (str(Cni) == "wait") or (str(loss_Ti) == "wait"):
                    # Case 1 至少一个收敛判据在等待计算, 保持当前周期的Cni文件
                    print("Case 1, T ", T_i, ": ", "waiting. Cni or delta Loss are waiting for computing.")
                    for inx in range(numSubNN):
                        send_signal_Cni('T{}_continue'.format(T_i), inx, T_i)
                elif (float(Cni) <= Cni_converge) and (np.abs(Loss[T_i]-Loss[T_i-1]) <= Loss_converge):
                    # Case 2 两个收敛判据同时满足, 更新下一周期的Cni文件为stop
                    print("Case 2, T ", T_i, ": ", "stop. Cni and Loss are satisfied, so stop.\n",
                              "    Cni is ", Cni, ", delta Loss is ", np.abs(Loss[T_i]-Loss[T_i-1]))
                    for inx in range(numSubNN):
                        send_signal_Cni('T{}_stop'.format(T_i + 1), inx, T_i + 1)
                    nextPeriod = 1
                    break
                else:
                    # Case 3 两个收敛判据还没有同时满足, 更新下一周期的Cni文件
                    print("Case 3, T ", T_i, ": ", "pass. Cni and Loss are not both satisfied, so start next peroid.\n",
                              "    Cni is ", Cni, ", delta Loss is ", np.abs(Loss[-1]-Loss[-2]))
                    for inx in range(numSubNN):
                        send_signal_Cni('T{}_continue'.format(T_i + 1), inx, T_i + 1)
                    nextPeriod = 1
            time.sleep(60)