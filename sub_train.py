"""
@author: Chang, responsible for DOI:https://doi.org/10.1063/5.0138287
"""
import os
import tensorflow as tf # TF1.14
import numpy as np
import scipy.io as io
import time
import argparse
import h5py

from utilities_ndm import data_process, HFM

# 随机种子用于复现实验结果 Random seed is used for repeating experiments.
np.random.seed(3407)
tf.set_random_seed(3407)

def check_signal_Cni(index,T_i):
    try:
        with open(f'signal_Cni_{index}_T{T_i}.txt', 'r') as f:
            return f.read()
    except FileNotFoundError:
        return 'wait'

def send_signal_Loss(Loss, index, T_i):
    with open(f'signal_Loss_{index}_T{T_i}.txt', 'w') as f:
        f.write(Loss)

if __name__ == "__main__":
    time0 = time.time()
    # =====================================================================
    # ||              0.Hyper-parameter defination 定义超参数              ||
    # =====================================================================

    # 创建解析器接收参数 Create a parser to receive some input parameters
    parser = argparse.ArgumentParser(description='Receive some input parameters.')
    parser.add_argument('--debug', type=int, choices=[0, 1], help='Debug or not')
    parser.add_argument('--index', type=int, default=0, help='sub_process')
    parser.add_argument('--tp1', type=int, default=0, help='Time block start')
    parser.add_argument('--tp2', type=int, default=120, help='Time block end')

    # 解析参数 Parsing parameters
    args = parser.parse_args()

    tp1, tp2 = args.tp1, args.tp2

    if args.debug==1:
        log_path = "./tp{}-{}_dt0.1s_X-x29p_t{}-{}_test".format(args.tp1+10, args.tp2-10, args.tp1, args.tp2)
        # batch_size 注意要能被eqns_Nx * eqns_Ny * eqns_Nt整除 40000
        # batch_size should be an integer multiple of (eqns_Nx * eqns_Ny * eqns_Nt)
        batch_size = 6000
        layers = [3] + 2 * [2] + [3]
        # 学习率衰减 Learning rate decay.
        lr = 0.001 # 初始学习率 Initial lr.
        mm = 1.0 # 学习率峰值衰减率 The decaying ratio of the peak lr.
        ep = 10 # 第一个周期的训练代数 Epoch of the first period.
        tm = 2 # 周期增长率 The growth rate of periods.
        dn = 6 # 最大衰减周期数 The max periods.
        # Nt_true是用于训练的快照数，间隔n*dt，Nt_true=Nt/n，Nt是总的快照数
        # Nt_true is the number of snapshots for training, Nt_true=Nt/n, Nt is the total number of snapshots
        Nx, Ny, Nt_true = 400, 200, 1100
        eqns_Nx, eqns_Ny, eqns_Nt = 10, 5, tp2 - tp1
    else:
        log_path = "./tp{}-{}_dt0.1s_X-x29p_t{}-{}".format(args.tp1+10, args.tp2-10, args.tp1, args.tp2)
        batch_size = 200000
        layers = [3] + 10 * [50] + [3]
        lr = 0.001
        mm = 1.0
        ep = 1000
        tm = 2
        dn = 6
        Nx, Ny, Nt_true = 400, 200, 1100
        eqns_Nx, eqns_Ny, eqns_Nt = 100, 50, tp2 - tp1
    
    index = args.index
    print("************************ SubNN ", index, " start, debug=", args.debug,
          ", tp", args.tp1,"-tp", args.tp2, " ************************")
    if os.path.exists(log_path) == False:
        os.makedirs(log_path)
    loss_name = log_path + "/loss-epoch.dat"

    loss_header = 'iter loss loss_data loss_eqns loss_e1 loss_e2 loss_e3\n'
    with open(loss_name, 'a') as f:
        f.write(loss_header)

    log_name = log_path + "/PINN_log.txt"
    pred_path = log_path + "/Pred_ep{}.h5" # .h5 or .mat, corresponds to Part3.Post process 
    NN_path = log_path + '/Trained_HFM_ep{}/tp_NN.ckpt'

    Rey = 100

    # mat_path = '../cylinder_Re100_lam_X1-9_400x200_5500_ndm.mat' # para
    mat_path = 'D:/AI/cylinder_post/cylinder_Re100_lam_X1-9_400x200_5500_ndm.mat'  # local

    # =====================================================================
    # ||                     1.前处理  preprocess                         ||
    # =====================================================================

    DATA = data_process(mat_path, Nx, Ny, Nt_true)

    # =====================Time block=====================
    t_tp, u_tp, v_tp, p_tp = DATA.tp(tp1, tp2)

    # =====================数据点布置方式一：根据xy方向上点数均匀布置数据点===================
    # =====================Sensor placement type 1：Rectangle=====================
    # data_Nx, data_Ny, data_Nt = 6, 5, tp2 - tp1  # 仅限数据点采用均匀布置(Only use in this placemrnttype)
    # x_data, y_data, t_data, u_data, v_data = DATA.data_uniform(t_tp, u_tp, v_tp, p_tp,
    #                                                            data_Nx, data_Ny, data_Nt)
    # =====================数据点布置方式一：结束 Sensor placement type 1: end ===========

    # =====================数据点布置方式二：根据点id抽取任意布置的数据点=====================
    # =====================Sensor placement type 2：Extract by ID =====================
    # 给id_ex, 参考最下面的备份 The various types of extraction are listed at the end.
    # Type diamond29 
    id_ex = np.array([[1, 1], [1, 100], [1, 200],
                      [50, 25], [50, 75], [50, 125], [50, 175],
                      [100, 50], [100, 150],
                      [150, 25], [150, 75], [150, 125], [150, 175],
                      [200, 1], [200, 100], [200, 200],
                      [250, 25], [250, 75], [250, 125], [250, 175],
                      [300, 50], [300, 150],
                      [350, 25], [350, 75], [350, 125], [350, 175],
                      [400, 1], [400, 100], [400, 200]
                      ])-1
    x_data, y_data, t_data, u_data, v_data = DATA.data_ext(t_tp, u_tp, v_tp, p_tp,id_ex)
    # ================ 数据点布置方式二：结束 Sensor placement type 2:end==================

    print("x_data shape is", x_data.shape)
    print("t_data shape is", t_data.shape)
    print("u_data shape is", u_data.shape)

    # =========================== 方程点 Equation points ===============================
    x_eqns, y_eqns, t_eqns = DATA.eqns(t_tp, eqns_Nx, eqns_Ny, eqns_Nt)
    print("x_eqns shape is", x_eqns.shape)
    print("t_eqns shape is", t_eqns.shape)

    # =====================预留用于重构的点 Inputs of reconstruction=======================
    # x_pred, y_pred, t_pred = DATA.x_all, DATA.y_all, DATA.t_all[DATA.tp1:DATA.tp2]
    x_pred_100, y_pred_100, t_pred_100, N_100, T_100 = DATA.data_pred(t1=10, t2=110,t_jump=1)  # Tensor
    # x_pred_500, y_pred_500, t_pred_500, N_500, T_500 = DATA.data_pred(t1=50, t2=550, t_jump=1)
    print("x_pred_100 shape is", x_pred_100.shape)
    # print("t_pred_500 shape is", t_pred_500.shape)

    # DATA.plot_data()
    # DATA.plot_eqns()

    del DATA

    time1 = time.time()
    print("Data prepare done, costs {:.3e}s".format(time1 - time0))
    with open(log_name, "a") as f:
        f.write('Data prepare done, costs {:.3e}s'.format(time1 - time0) + '\n')



    # =====================================================================
    # ||                        2.训练网络 NN training                     ||
    # =====================================================================
    # time0 = time.time()
    # Training
    print("==============================Load HFM=============================")
    
    np_lr = np.zeros(dn) + lr
    np_ep = np.zeros(dn) + ep
    for i in range(1, dn):
        lr, ep = mm * lr, tm * ep
        np_lr[i], np_ep[i] = lr, ep
    print("========Max lr======== is ", np_lr)
    print("========Each epochs======== is ", np_ep)
    
    print("==============================Training start=============================")
    for i in range(dn):
        lr = np_lr[i].astype('float32')
        epochs = np_ep[i].astype('int32')
        model = HFM(t_data, x_data, y_data, u_data, v_data,
                    t_eqns, x_eqns, y_eqns,
                    layers, batch_size,
                    Rey,
                    lr, epochs, tm, mm)
        if i>=1:
            model.saver.restore(model.sess, NN_path.format(int(np_ep[i-1])))
        
        model.train(epochs)
        # print(tf.contrib.framework.get_variables_to_restore())
        model.saver.save(model.sess, NN_path.format(epochs)) # 会自动创建文件夹
    
        time1 = time.time()
        print("=====================SubNN",index," training T",i," done!=============================")
        print("Training cost {:.3e}s".format(time1 - time0) + '\n')
        with open(log_name, "a") as f:
            f.write('Training cost {:.3e}s'.format(time1 - time0) + '\n')
    
        # =====================================================================
        # ||                         3.Post process                       ||
        # =====================================================================
        time0 = time.time()
        # ================================预测=================================
        U_pred_dict = model.predict(x_pred_100,y_pred_100,t_pred_100,N_100,T_100)
        # io.savemat(pred_path.format(epochs), U_pred_dict) # save as .mat
        # 创建一个新的.h5文件
        with h5py.File(pred_path.format(epochs), 'w') as hf:
            for key, value in U_pred_dict.items():
                hf.create_dataset(key, data=value)
    
        # ============================ 输出损失 Write loss ==============================
        a_ep = np.array(model.a_ep)
        a_loss = np.array(model.a_loss)
        a_loss_data = np.array(model.a_loss_data)
        a_loss_eqns = np.array(model.a_loss_eqns)
        a_loss_e1 = np.array(model.a_loss_e1)
        a_loss_e2 = np.array(model.a_loss_e2)
        a_loss_e3 = np.array(model.a_loss_e3)
        # a_inertial_x = np.array(model.a_inertial_x)
        # a_inertial_y = np.array(model.a_inertial_y)
        # a_p_gradient_x = np.array(model.a_p_gradient_x)
        # a_p_gradient_y = np.array(model.a_p_gradient_y)
        # a_viscous_x = np.array(model.a_viscous_x)
        # a_viscous_y = np.array(model.a_viscous_y)
        a_lr = np.array(model.a_lr)
        l_e = np.concatenate((a_ep.reshape(-1,1), a_loss.reshape(-1,1), a_loss_data.reshape(-1,1),
                              a_loss_eqns.reshape(-1,1),
                              a_loss_e1.reshape(-1,1),
                              a_loss_e2.reshape(-1,1),
                              a_loss_e3.reshape(-1,1),
                              # a_inertial_x.reshape(-1,1),
                              # a_inertial_y.reshape(-1,1),
                              # a_p_gradient_x.reshape(-1,1),
                              # a_p_gradient_y.reshape(-1,1),
                              # a_viscous_x.reshape(-1,1),
                              # a_viscous_y.reshape(-1,1),
                              a_lr.reshape(-1,1),
                              ),axis=1)
        loss_header = 'iter loss loss_data loss_eqns loss_e1 loss_e2 loss_e3'
        with open(loss_name,'ab') as f:
            np.savetxt(f,l_e,delimiter=" ")
    
    
        time1 = time.time()
        with open(log_name,"a") as f:
            f.write('Output cost {}s'.format(time1-time0) + '\n')
    
        # def send_signal_Loss(Loss, index, T_i):
        #   with open(f'Loss_{index}_T{T_i}.txt', 'w') as f:
        #       f.write(Loss)

        send_signal_Loss(str(a_loss[-1]), index, i)
        nextPeriod = 0
        while nextPeriod == 0:
            # 'signal_Cni_{index}_{T_i}.txt'
            signal = check_signal_Cni(index,i)
            print("=======SubRefresh: SubNN", index," received signal ", signal)
            if signal == 'T{}_stop'.format(i):
                print(f"Process {index} stopping as per the signal.")
                break
            elif signal == 'T{}_continue'.format(i):
                print(f"Process {index} continuing as per the signal.")
                nextPeriod = 1
            else:
                print(f"Process {index} received unknown signal: {signal}")
            time.sleep(30)

        tf.reset_default_graph()
        del model

    # =====================================================================
    # ||                             4.加载模型预测                            ||
    # =====================================================================

    # 加载模型
    # ============================================查看checkpoint中变量名称================================================
    # from tensorflow.python import pywrap_tensorflow
    # model_dir = "Saved_model"
    # checkpoint_path = os.path.join(model_dir, "model.ckpt")
    # checkpoint_path = './A_TF1test_TrainedModel/training/HFM_trained.ckpt'
    # reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    # var_to_shape_map = reader.get_variable_to_shape_map()
    # for key in var_to_shape_map:
    #     print("tensor_name: ", key, end=' ')
    #     print(reader.get_tensor(key))
    # ============================================查看checkpoint中变量名称================================================


    # model_load_done = HFM( t_data, x_data, y_data, u_data, v_data,
    #                        t_eqns, x_eqns, y_eqns,
    #                        layers, batch_size,
    #                        Rey,
    #                        lr,ep,tm,mm)
    # print(NN_path)
    # model_load_done.saver.restore(model_load_done.sess, NN_path)

    # U_pred_dict_done = model_load_done.predict(x_pred_500, y_pred_500, t_pred_500, N_500, T_500)
    # print(pred_path)
    # io.savemat(pred_path + '_dt0.02.mat', U_pred_dict_done)
    # del model_load_done

    # 点坐标备份
    # 大X套小x型
    # id_ex = np.array([[1, 1], [1, 100], [1, 200],
    #                   [50, 25], [50, 75], [50, 125], [50, 175],
    #                   [100, 50], [100, 150],
    #                   [150, 25], [150, 75], [150, 125], [150, 175],
    #                   [200, 0], [200, 100], [200, 200],
    #                   [250, 25], [250, 75], [250, 125], [250, 175],
    #                   [300, 50], [300, 150],
    #                   [350, 25], [350, 75], [350, 125], [350, 175],
    #                   [400, 1], [400, 100], [400, 200]
    #                   ])-1

    # 一个大X
    # id_ex = np.array([[0, 0], [0, 196],
    #                   [28, 14], [28, 182],
    #                   [56, 28], [56, 168],
    #                   [84, 42], [84, 154],
    #                   [112, 56], [112, 140],
    #                   [140, 70], [140, 126],
    #                   [168, 84], [168, 112],
    #                   [196, 98],
    #                   [224, 112], [224, 84],
    #                   [252, 126], [252, 70],
    #                   [280, 140], [280, 56],
    #                   [308, 154], [308, 42],
    #                   [336, 168], [336, 28],
    #                   [364, 182], [364, 14],
    #                   [399, 196], [399, 0],
    #                   ])

    # O型
    # id_ex = np.array([[49, 24],
    #                   [34, 9], [64, 9],
    #                   [9, 19], [89, 19],
    #                   [9, 29], [89, 29],
    #                   [34, 39], [64, 39]])
