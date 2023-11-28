import subprocess
import numpy as np
import PyPOD
import time

UseGPU = False
mainDebug = True

def gpu_process(script_name, gpu_id, other_arg):
    return subprocess.Popen(["python", script_name, "--gpu_id", str(gpu_id), "--other_arg", other_arg])

def cpu_process(script_name, args):
    cmd = ["python", script_name] + args
    return subprocess.Popen(cmd)

def Calc_Cni(file_paths, order = 6, N=3, snaps=1000):
    try:
        u_data = []
        v_data = []
        phi = []
        eps = 0
        for path in file_paths:
            with open(path, "r") as file:
                u_data.append(np.array(file['u_pred']))
                v_data.append(np.array(file['v_pred']))
        u_pred = np.hstack(u_data)
        v_pred = np.hstack(v_data)
        # UV_pred = np.vstack((u_pred,v_pred))
        for snap in range(100, snaps+1, 100):
            UV_pred = np.vstack((u_pred[:,0:snap],v_pred[:,0:snap]))
            Eigenvalues_pred,Eigenvectors_pred,phi_pred,ak_t_pred = PyPOD.POD(UV_pred)
            phi.append(phi_pred[:,order])
            del Eigenvalues_pred,Eigenvectors_pred,ak_t_pred
        phi_base = phi[-1]
        for n in range(N):
            for i in range(order):
                eps += np.abs(np.dot(phi[n-N-1][:,i],phi_base[:,i])) # n-N: -4,-3,-2
        Cni = 1 - eps/(N*order)
        return Cni
    except FileNotFoundError:
        return "wait"

def Calc_Loss(file_paths):
    try:
        Loss = []
        for path in file_paths:
            with open(path, "r") as f:
                Loss.append(f.read())
        Loss_avg = np.mean(Loss)
        return Loss_avg
    except FileNotFoundError:
        return "wait"

def send_signal_Cni(signal_Cni, index, T_i):
    with open(f'signal_Cni_{index}_T{T_i}.txt', 'w') as f:
        f.write(signal_Cni)
if __name__ == "__main__":
    Cni_converge = 1e-2
    Loss_converge = 1e-3
    # 定义每个脚本的参数
    numSubNN = 4
    args_tp1 = ["--debug=True", "--index=0", "--tp1=0", "--tp2=120"]
    args_tp2 = ["--debug=True", "--index=1", "--tp1=100", "--tp2=220"]
    args_tp3 = ["--debug=True", "--index=2", "--tp1=200", "--tp2=320"]
    args_tp4 = ["--debug=True", "--index=3", "--tp1=300", "--tp2=420"]

    if UseGPU:
        sub_process = gpu_process
        processes = {
        "tp1": sub_process("sub_train.py", 0, args_tp1), 
        "tp2": sub_process("sub_train.py", 1, args_tp2),
        "tp3": sub_process("sub_train.py", 2, args_tp3),
        "tp4": sub_process("sub_train.py", 3, args_tp4),
        }
    else:
        sub_process = cpu_process
        processes = {
        "tp1": sub_process("sub_train.py", args_tp1), 
        "tp2": sub_process("sub_train.py", args_tp2),
        "tp3": sub_process("sub_train.py", args_tp3),
        "tp4": sub_process("sub_train.py", args_tp4),
        }
    tm = 2
    dn_max = 6
    ep = 10
    np_ep = np.zeros(dn_max) + ep
    for i in range(1, dn_max):
        ep =  tm * ep
        np_ep[i] = ep
    Loss = []
    for T_i in range(np_ep.shape[0]):
        nextPeriod = 0
        ep = np_ep[i]
        file_paths = []
        if mainDebug:
            for i in range(0,4,1):
                file_paths.append("./tp{}-{}_dt0.1s_X-x29p_t{}-{}_test".format(i*100+10, ((i+1)*100+20)-10, i*100, ((i+1)*100+20))
                                  + "/Pred_ep{}.mat".format(ep))
        else:
            for i in range(0,4,1):
                file_paths.append("./tp{}-{}_dt0.1s_X-x29p_t{}-{}".format(i*100+10, ((i+1)*100+20)-10, i*100, ((i+1)*100+20))
                                  + "/Pred_ep{}.mat".format(ep))
        loss_path = []
        for inx in range(numSubNN):
            loss_path.append('Loss_{}_T{}.txt'.format(inx,T_i))

        while nextPeriod==0:
            Cni = Calc_Cni(file_paths, order = 6, N=3, snaps=1000)
            loss_Ti = Calc_Loss(loss_path)

            if(str(loss_Ti) != "wait"):
                if T_i == 0:
                    Loss.append(1e10)
                else:
                    Loss.append(loss_Ti)
            
            if T_i == 0: # First Peroid
                if (str(Cni) != "wait") and (str(loss_Ti) != "wait"):
                    for inx in range(numSubNN):
                        send_signal_Cni('T{}_continue'.format(T_i), inx, T_i)
                    nextPeriod==1
                else:
                    for inx in range(numSubNN):
                        send_signal_Cni('T{}_continue'.format(T_i), inx, T_i)
                
            else:
                if (str(Cni) == "wait") or (str(loss_Ti) == "wait"):
                    for inx in range(numSubNN):
                        send_signal_Cni('T{}_continue'.format(T_i), inx, T_i)
                elif (float(Cni) <= Cni_converge) and (np.abs(Loss[-1]-Loss[-2]) <= Loss_converge):
                    for inx in range(numSubNN):
                        send_signal_Cni('T{}_stop'.format(T_i), inx, T_i)
                    nextPeriod==1
                else:
                    for inx in range(numSubNN):
                        send_signal_Cni('T{}_continue'.format(i), inx, i)
                    nextPeriod==1
            time.sleep(10)

# with open(f'signal_{index}_T{T_i}.txt', 'w') as f:
#        f.write(signal)
# def Calc_Loss(file_paths):
#     try:
#         Loss = []
#         for path in file_paths:
#             with open(path, "r") as f:
#                 Loss.append(f.read())
#         Loss_avg = np.mean(Loss)
#         return Loss_avg
#     except FileNotFoundError:
#         return "wait"