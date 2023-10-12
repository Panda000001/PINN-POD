"""
@author: Maziar Raissi
"""
import sys
import time
import scipy.io as io
import tensorflow as tf
import numpy as np
import PyPOD


def tf_session():
    # tf session
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=True)
    config.gpu_options.force_gpu_compatible = True
    sess = tf.Session(config=config)
    
    # init
    init = tf.global_variables_initializer()
    sess.run(init)
    
    return sess

def relative_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.sqrt(np.mean(np.square(pred - exact))/np.mean(np.square(exact - np.mean(exact))))
    return tf.sqrt(tf.reduce_mean(tf.square(pred - exact))/tf.reduce_mean(tf.square(exact - tf.reduce_mean(exact))))

def mean_squared_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.mean(np.square(pred - exact)) # 
    return tf.reduce_mean(tf.square(pred - exact))

def fwd_gradients(Y, x):
    dummy = tf.ones_like(Y)
    G = tf.gradients(Y, x, grad_ys=dummy, colocate_gradients_with_ops=True)[0]
    Y_x = tf.gradients(G, dummy, colocate_gradients_with_ops=True)[0]
    return Y_x


class neural_net_potential(object):
    def __init__(self, *inputs, layers):
        
        layers[-1] = layers[-1] - 1
        self.layers = layers
        self.num_layers = len(self.layers)
        
        if len(inputs) == 0:
            in_dim = self.layers[0]
            self.X_mean = np.zeros([1, in_dim])
            self.X_std = np.ones([1, in_dim])
        else:
            X = np.concatenate(inputs, 1)
            self.X_mean = X.mean(0, keepdims=True)
            self.X_std = X.std(0, keepdims=True)
        
        self.weights = []
        self.biases = []
        self.gammas = []
        
        for l in range(0,self.num_layers-1):
            in_dim = self.layers[l]
            out_dim = self.layers[l+1]
            W = np.random.normal(size=[in_dim, out_dim])
            b = np.zeros([1, out_dim])
            g = np.ones([1, out_dim])
            # tensorflow variables
            self.weights.append(tf.Variable(W, dtype=tf.float32, trainable=True))
            self.biases.append(tf.Variable(b, dtype=tf.float32, trainable=True))
            self.gammas.append(tf.Variable(g, dtype=tf.float32, trainable=True))
        
    def __call__(self, *inputs):
                
        H = (tf.concat(inputs, 1) - self.X_mean)/self.X_std
    
        for l in range(0, self.num_layers-1):
            W = self.weights[l]
            b = self.biases[l]
            g = self.gammas[l]
            # weight normalization
            V = W/tf.norm(W, axis = 0, keepdims=True)
            # matrix multiplication
            H = tf.matmul(H, V)
            # add bias
            H = g*H + b
            # activation
            if l < self.num_layers-2:
                H = H*tf.sigmoid(H) # swish(x) = x * sigmoid(x) 
                
        # Y = tf.split(H, num_or_size_splits=H.shape[1], axis=1)
        psi_and_p = tf.split(H, num_or_size_splits=H.shape[1], axis=1)
        psi = psi_and_p[0]
        p = psi_and_p[1]
        x = inputs[0]
        y = inputs[1]
        u = tf.gradients(psi, y)[0]
        v = -tf.gradients(psi, x)[0]
        return u, v, p

class neural_net_save(object):
    def __init__(self, *inputs, layers):
        
        self.layers = layers
        self.num_layers = len(self.layers)
        
        if len(inputs) == 0:
            in_dim = self.layers[0]
            self.X_mean = np.zeros([1, in_dim])
            self.X_std = np.ones([1, in_dim])
        else:
            X = np.concatenate(inputs, 1)
            self.X_mean = X.mean(0, keepdims=True) # 均值
            self.X_std = X.std(0, keepdims=True) # 标准差
        
        self.weights = []
        self.biases = []
        self.gammas = []

        for l in range(0,self.num_layers-1):
            in_dim = self.layers[l]
            out_dim = self.layers[l+1]
            W = np.random.normal(size=[in_dim, out_dim])
            b = np.zeros([1, out_dim])
            g = np.ones([1, out_dim])
            # tensorflow variables
            self.weights.append(tf.Variable(W, dtype=tf.float32, trainable=True,name='w{}'.format(l)))
            self.biases.append(tf.Variable(b, dtype=tf.float32, trainable=True,name='b{}'.format(l)))
            self.gammas.append(tf.Variable(g, dtype=tf.float32, trainable=True,name='g{}'.format(l)))
            
    def __call__(self, *inputs):
                
        H = (tf.concat(inputs, 1) - self.X_mean)/self.X_std  # 归一化

    
        for l in range(0, self.num_layers-1):
            W = self.weights[l]
            b = self.biases[l]
            g = self.gammas[l]
            # weight normalization
            V = W/tf.norm(W, axis = 0, keepdims=True)
            # matrix multiplication
            H = tf.matmul(H, V)
            # add bias
            H = g*H + b
            # activation
            if l < self.num_layers-2:
                H = H*tf.sigmoid(H)
                
        Y = tf.split(H, num_or_size_splits=H.shape[1], axis=1)
    
        return Y


class neural_net(object):
    def __init__(self, *inputs, layers):
        
        self.layers = layers
        self.num_layers = len(self.layers)
        
        if len(inputs) == 0:
            in_dim = self.layers[0]
            self.X_mean = np.zeros([1, in_dim])
            self.X_std = np.ones([1, in_dim])
        else:
            X = np.concatenate(inputs, 1)
            self.X_mean = X.mean(0, keepdims=True)
            self.X_std = X.std(0, keepdims=True)
        
        self.weights = []
        self.biases = []
        self.gammas = []

        # self.saver = tf.train.Saver()
        
        for l in range(0,self.num_layers-1):
            in_dim = self.layers[l]
            out_dim = self.layers[l+1]
            W = np.random.normal(size=[in_dim, out_dim])
            b = np.zeros([1, out_dim])
            g = np.ones([1, out_dim])
            # tensorflow variables
            self.weights.append(tf.Variable(W, dtype=tf.float32, trainable=True,name='W{}'.format(l)))
            self.biases.append(tf.Variable(b, dtype=tf.float32, trainable=True,name='B{}'.format(l)))
            self.gammas.append(tf.Variable(g, dtype=tf.float32, trainable=True,name='G{}'.format(l)))
            
    def __call__(self, *inputs):
                
        H = (tf.concat(inputs, 1) - self.X_mean)/self.X_std
    
        for l in range(0, self.num_layers-1):
            W = self.weights[l]
            b = self.biases[l]
            g = self.gammas[l]
            # weight normalization
            V = W/tf.norm(W, axis = 0, keepdims=True)
            # matrix multiplication
            H = tf.matmul(H, V)
            # add bias
            H = g*H + b
            # activation
            if l < self.num_layers-2:
                H = H*tf.sigmoid(H)
                
        Y = tf.split(H, num_or_size_splits=H.shape[1], axis=1)
    
        return Y


def Navier_Stokes_2D_Re_mv(u, v, p, t, x, y, Rey):
    

    Y = tf.concat([u, v, p], 1)
    
    Y_t = fwd_gradients(Y, t)
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    Y_xx = fwd_gradients(Y_x, x)
    Y_yy = fwd_gradients(Y_y, y)
    
    u = Y[:,0:1]
    v = Y[:,1:2]
    p = Y[:,2:3]
    # c = Y[:,3:4]
    
    u_t = Y_t[:,0:1]
    v_t = Y_t[:,1:2]
    # p_t = Y_t[:,2:3]
    
    u_x = Y_x[:,0:1]
    v_x = Y_x[:,1:2]
    p_x = Y_x[:,2:3]
    # c_x = Y_x[:,3:4]
    
    u_y = Y_y[:,0:1]
    v_y = Y_y[:,1:2]
    p_y = Y_y[:,2:3]
    # c_y = Y_y[:,3:4]
    
    u_xx = Y_xx[:,0:1]
    v_xx = Y_xx[:,1:2]
    # c_xx = Y_xx[:,2:3]
    
    u_yy = Y_yy[:,0:1]
    v_yy = Y_yy[:,1:2]
    # c_yy = Y_yy[:,2:3]
    
    # e1 = c_t + (u*c_x + v*c_y) - (1.0/Pec)*(c_xx + c_yy)
    # e2 = u_t + (u*u_x + v*u_y) + p_x - (1.0/Rey)*(u_xx + u_yy) 
    # e3 = v_t + (u*v_x + v*v_y) + p_y - (1.0/Rey)*(v_xx + v_yy)
    # e4 = u_x + v_y
    res_h = u_x + v_y
    res_f = Rey*(u_t + u*u_x + v*u_y + p_x) - (u_xx + u_yy)
    res_g = Rey*(v_t + u*v_x + v*v_y + p_y) - (v_xx + v_yy)

    return res_h, res_f, res_g


def Navier_Stokes_2D_psi(psi, p, t, x, y, Rey, eqns=True):
    if eqns:
        u = fwd_gradients(psi, y) # u=∂ψ/∂y
        v = -fwd_gradients(psi, x) # v=-∂ψ/∂x

        Y = tf.concat([u, v, p], 1)
        p = Y[:,2:3]

        Y_t = fwd_gradients(Y, t)
        Y_x = fwd_gradients(Y, x)
        Y_y = fwd_gradients(Y, y)
        Y_xx = fwd_gradients(Y_x, x)
        Y_yy = fwd_gradients(Y_y, y)
            
        u_t = Y_t[:,0:1]
        v_t = Y_t[:,1:2]

        u_x = Y_x[:,0:1]
        v_x = Y_x[:,1:2]
        p_x = Y_x[:,2:3]

        u_y = Y_y[:,0:1]
        v_y = Y_y[:,1:2]
        p_y = Y_y[:,2:3]
        
        u_xx = Y_xx[:,0:1]
        v_xx = Y_xx[:,1:2]
        
        u_yy = Y_yy[:,0:1]
        v_yy = Y_yy[:,1:2]
        
        res_f = Rey*(u_t + u*u_x + v*u_y + p_x) - (u_xx + u_yy)
        res_g = Rey*(v_t + u*v_x + v*v_y + p_y) - (v_xx + v_yy)

        inertial_x = u_t + u*u_x + v*u_y
        inertial_y = v_t + u*v_x + v*v_y
        p_gradient_x = p_x
        p_gradient_y = p_y
        viscous_x = - (1.0/Rey)*(u_xx + u_yy)
        viscous_y = - (1.0/Rey)*(v_xx + v_yy)

        NS_dict = {'u':u,'v':v,'p':p,
                'res_f':res_f, 'res_g':res_g,
                'inertial_x':inertial_x, 'inertial_y':inertial_y,
                'p_gradient_x':p_gradient_x, 'p_gradient_y':p_gradient_y,
                'viscous_x':viscous_x, 'viscous_y':viscous_y}
    else:
        u = fwd_gradients(psi, y) # u=∂ψ/∂y
        v = -fwd_gradients(psi, x) # v=-∂ψ/∂x
        NS_dict = {'u':u,'v':v}

    return NS_dict 




def Navier_Stokes_2D(u, v, p, t, x, y, Rey):
    
    Y = tf.concat([u, v, p], 1)
    
    Y_t = fwd_gradients(Y, t)
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    Y_xx = fwd_gradients(Y_x, x)
    Y_yy = fwd_gradients(Y_y, y)
    
    u = Y[:,0:1]
    v = Y[:,1:2]
    p = Y[:,2:3]
    # c = Y[:,3:4]
    
    u_t = Y_t[:,0:1]
    v_t = Y_t[:,1:2]
    # p_t = Y_t[:,2:3]
    
    u_x = Y_x[:,0:1]
    v_x = Y_x[:,1:2]
    p_x = Y_x[:,2:3]
    # c_x = Y_x[:,3:4]
    
    u_y = Y_y[:,0:1]
    v_y = Y_y[:,1:2]
    p_y = Y_y[:,2:3]
    # c_y = Y_y[:,3:4]
    
    u_xx = Y_xx[:,0:1]
    v_xx = Y_xx[:,1:2]
    # c_xx = Y_xx[:,2:3]
    
    u_yy = Y_yy[:,0:1]
    v_yy = Y_yy[:,1:2]
    # c_yy = Y_yy[:,2:3]
    
    # e1 = c_t + (u*c_x + v*c_y) - (1.0/Pec)*(c_xx + c_yy)
    # e2 = u_t + (u*u_x + v*u_y) + p_x - (1.0/Rey)*(u_xx + u_yy) 
    # e3 = v_t + (u*v_x + v*v_y) + p_y - (1.0/Rey)*(v_xx + v_yy)
    # e4 = u_x + v_y
    res_h = u_x + v_y
    res_f = u_t + u*u_x + v*u_y + p_x - (1.0/Rey)*(u_xx + u_yy)
    res_g = v_t + u*v_x + v*v_y + p_y - (1.0/Rey)*(v_xx + v_yy)
    inertial_x = u_t + u*u_x + v*u_y
    inertial_y = v_t + u*v_x + v*v_y
    p_gradient_x = p_x
    p_gradient_y = p_y
    viscous_x = - (1.0/Rey)*(u_xx + u_yy)
    viscous_y = - (1.0/Rey)*(v_xx + v_yy)
    NS_dict = {'res_h':res_h, 'res_f':res_f, 'res_g':res_g,
               'inertial_x':inertial_x, 'inertial_y':inertial_y,
               'p_gradient_x':p_gradient_x, 'p_gradient_y':p_gradient_y,
               'viscous_x':viscous_x, 'viscous_y':viscous_y}

    return NS_dict 

def Gradient_Velocity_2D(u, v, x, y):
    
    Y = tf.concat([u, v], 1)
    
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    
    u_x = Y_x[:,0:1]
    v_x = Y_x[:,1:2]
    
    u_y = Y_y[:,0:1]
    v_y = Y_y[:,1:2]
    
    return [u_x, v_x, u_y, v_y]

def Strain_Rate_2D(u, v, x, y):
    
    [u_x, v_x, u_y, v_y] = Gradient_Velocity_2D(u, v, x, y)
    
    eps11dot = u_x
    eps12dot = 0.5*(v_x + u_y)
    eps22dot = v_y
    
    return [eps11dot, eps12dot, eps22dot]

def Navier_Stokes_3D(u, v, w, p, t, x, y, z, Rey):
    
    Y = tf.concat([u, v, w, p], 1)
    
    Y_t = fwd_gradients(Y, t)
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    Y_z = fwd_gradients(Y, z)
    Y_xx = fwd_gradients(Y_x, x)
    Y_yy = fwd_gradients(Y_y, y)
    Y_zz = fwd_gradients(Y_z, z)
    
    u = Y[:,0:1]
    v = Y[:,1:2]
    w = Y[:,2:3]
    p = Y[:,3:4]

    
    u_t = Y_t[:,0:1]
    v_t = Y_t[:,1:2]
    w_t = Y_t[:,2:3]
    
    u_x = Y_x[:,0:1]
    v_x = Y_x[:,1:2]
    w_x = Y_x[:,2:3]
    p_x = Y_x[:,3:4]
    
    u_y = Y_y[:,0:1]
    v_y = Y_y[:,1:2]
    w_y = Y_y[:,2:3]
    p_y = Y_y[:,3:4]
       
    u_z = Y_z[:,0:1]
    v_z = Y_z[:,1:2]
    w_z = Y_z[:,2:3]
    p_z = Y_z[:,3:4]
    
    u_xx = Y_xx[:,0:1]
    v_xx = Y_xx[:,1:2]
    w_xx = Y_xx[:,2:3]
    
    u_yy = Y_yy[:,0:1]
    v_yy = Y_yy[:,1:2]
    w_yy = Y_yy[:,2:3]
       
    u_zz = Y_zz[:,0:1]
    v_zz = Y_zz[:,1:2]
    w_zz = Y_zz[:,2:3]

    e1 = u_x + v_y + w_z
    e2 = u_t + (u*u_x + v*u_y + w*u_z) + p_x - (1.0/Rey)*(u_xx + u_yy + u_zz)
    e3 = v_t + (u*v_x + v*v_y + w*v_z) + p_y - (1.0/Rey)*(v_xx + v_yy + v_zz)
    e4 = w_t + (u*w_x + v*w_y + w*w_z) + p_z - (1.0/Rey)*(w_xx + w_yy + w_zz)
    
    inertial_x = u_t + (u*u_x + v*u_y + w*u_z)
    inertial_y = v_t + (u*v_x + v*v_y + w*v_z)
    inertial_z = w_t + (u*w_x + v*w_y + w*w_z)
    p_gradient_x = p_x
    p_gradient_y = p_y
    p_gradient_z = p_z
    viscous_x = - (1.0/Rey)*(u_xx + u_yy + u_zz)
    viscous_y = - (1.0/Rey)*(v_xx + v_yy + v_zz)
    viscous_z = - (1.0/Rey)*(w_xx + w_yy + w_zz)

    NS_dict = {'e1':e1, 'e2':e2, 'e3':e3, 'e4':e4,
               'inertial_x':inertial_x, 'inertial_y':inertial_y, 'inertial_z':inertial_z,
               'p_gradient_x':p_gradient_x, 'p_gradient_y':p_gradient_y, 'p_gradient_z':p_gradient_z,
               'viscous_x':viscous_x, 'viscous_y':viscous_y, 'viscous_z':viscous_z}

    
    return NS_dict


def Gradient_Velocity_3D(u, v, w, x, y, z):
    
    Y = tf.concat([u, v, w], 1)
    
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    Y_z = fwd_gradients(Y, z)
    
    u_x = Y_x[:,0:1]
    v_x = Y_x[:,1:2]
    w_x = Y_x[:,2:3]
    
    u_y = Y_y[:,0:1]
    v_y = Y_y[:,1:2]
    w_y = Y_y[:,2:3]
    
    u_z = Y_z[:,0:1]
    v_z = Y_z[:,1:2]
    w_z = Y_z[:,2:3]
    
    return [u_x, v_x, w_x, u_y, v_y, w_y, u_z, v_z, w_z]

def Shear_Stress_3D(u, v, w, x, y, z, nx, ny, nz, Rey):
        
    [u_x, v_x, w_x, u_y, v_y, w_y, u_z, v_z, w_z] = Gradient_Velocity_3D(u, v, w, x, y, z)

    uu = u_x + u_x
    uv = u_y + v_x
    uw = u_z + w_x
    vv = v_y + v_y
    vw = v_z + w_y
    ww = w_z + w_z
    
    sx = (uu*nx + uv*ny + uw*nz)/Rey
    sy = (uv*nx + vv*ny + vw*nz)/Rey
    sz = (uw*nx + vw*ny + ww*nz)/Rey
    
    return sx, sy, sz


class HFM(object):
    # notational conventions 符号约定
    # _tf: placeholders for input/output data and points used to regress the equations
    # _pred: output of neural network
    # _eqns: points used to regress the equations
    # _data: input-output data
    # _inlet: input-output data at the inlet
    # _star: preditions

    def __init__(self, t_data, x_data, y_data, u_data, v_data,
                 t_eqns, x_eqns, y_eqns,
                 layers, batch_size,
                 Rey,
                 lr,t0,tm,mm):
        self.Rey = Rey

        # specs
        self.layers = layers
        self.batch_size = batch_size

        # flow properties
        # self.Pec = Pec
        # self.Rey = Rey

        # data
        [self.t_data, self.x_data, self.y_data, self.u_data, self.v_data] = [t_data, x_data, y_data, u_data, v_data]
        [self.t_eqns, self.x_eqns, self.y_eqns] = [t_eqns, x_eqns, y_eqns]
        # [self.t_inlet, self.x_inlet, self.y_inlet, self.u_inlet, self.v_inlet] = [t_inlet, x_inlet, y_inlet, u_inlet, v_inlet]

        # placeholders
        [self.t_data_tf, self.x_data_tf, self.y_data_tf, self.u_data_tf, self.v_data_tf] = [
            tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(5)]
        [self.t_eqns_tf, self.x_eqns_tf, self.y_eqns_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in
                                                            range(3)]
        # [self.t_inlet_tf, self.x_inlet_tf, self.y_inlet_tf, self.u_inlet_tf, self.v_inlet_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(5)]

        # physics "uninformed" neural networks
        self.net_uvp = neural_net(self.t_data, self.x_data, self.y_data, layers=self.layers)

        # 被网络计算出的观测点的uvp: _data_pred
        [self.u_data_pred,
         self.v_data_pred,
         self.p_data_pred] = self.net_uvp(self.t_data_tf,
                                          self.x_data_tf,
                                          self.y_data_tf)

        # physics "uninformed" neural networks (data at the inlet)被网络计算出的Inlet上的uv
        # [_,self.u_inlet_pred,self.v_inlet_pred,_] = self.net_cuvp(self.t_inlet_tf,
        #                     self.x_inlet_tf,
        #                     self.y_inlet_tf)

        # physics "informed" neural networks===================================
        # 被网络计算出的非观测方程点的cuvp: _eqns_pred
        [self.u_eqns_pred,
         self.v_eqns_pred,
         self.p_eqns_pred] = self.net_uvp(self.t_eqns_tf,
                                          self.x_eqns_tf,
                                          self.y_eqns_tf)
        # PINN计算出的非观测方程点的残差
        NS_dict_data_pred = Navier_Stokes_2D(self.u_eqns_pred,
                                             self.v_eqns_pred,
                                             self.p_eqns_pred,
                                             self.t_eqns_tf,
                                             self.x_eqns_tf,
                                             self.y_eqns_tf,
                                             self.Rey)
        self.e1_eqns_pred = NS_dict_data_pred['res_h']
        self.e2_eqns_pred = NS_dict_data_pred['res_f']
        self.e3_eqns_pred = NS_dict_data_pred['res_g']
        self.inertial_x_eqns_pred = tf.reduce_sum(NS_dict_data_pred['inertial_x'])
        self.inertial_y_eqns_pred = tf.reduce_sum(NS_dict_data_pred['inertial_y'])
        self.p_gradient_x_eqns_pred = tf.reduce_sum(NS_dict_data_pred['p_gradient_x'])
        self.p_gradient_y_eqns_pred = tf.reduce_sum(NS_dict_data_pred['p_gradient_y'])
        self.viscous_x_eqns_pred = tf.reduce_sum(NS_dict_data_pred['viscous_x'])
        self.viscous_y_eqns_pred = tf.reduce_sum(NS_dict_data_pred['viscous_y'])

        # gradients required for the lift and drag forces
        # [self.u_x_eqns_pred,
        #  self.v_x_eqns_pred,
        #  self.u_y_eqns_pred,
        #  self.v_y_eqns_pred] = Gradient_Velocity_2D(self.u_eqns_pred,
        #                                             self.v_eqns_pred,
        #                                             self.x_eqns_tf,
        #                                             self.y_eqns_tf)

        # loss
        self.loss_data = mean_squared_error(self.u_data_pred, self.u_data_tf) + \
                         mean_squared_error(self.v_data_pred, self.v_data_tf)

        self.loss_e1 = mean_squared_error(self.e1_eqns_pred, 0.0)
        self.loss_e2 = mean_squared_error(self.e2_eqns_pred, 0.0)
        self.loss_e3 = mean_squared_error(self.e3_eqns_pred, 0.0)

        self.loss_eqns = self.loss_e1 + self.loss_e2 + self.loss_e3

        self.loss = self.loss_data + self.loss_eqns

        self.a_ep = []  # 存所有的迭代次数
        self.a_loss = []  # 存所有的loss值
        self.a_loss_data = []  # 存所有的loss_data值
        self.a_loss_e1 = []
        self.a_loss_e2 = []
        self.a_loss_e3 = []
        self.a_loss_eqns = []  # 存所有的loss_eqns值

        # self.a_inertial_x = [] # 存分项的loss值
        # self.a_inertial_y = []
        # self.a_p_gradient_x = []
        # self.a_p_gradient_y = []
        # self.a_viscous_x = []
        # self.a_viscous_y = []
        self.a_lr = []  # 存学习率如果衰减的话

        # optimizers 无衰减
        # self.learning_rate = tf.placeholder(tf.float32, shape=[])
        # self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        # self.train_op = self.optimizer.minimize(self.loss)

        self.global_step = tf.placeholder(tf.int32, shape=[])
        # 指数衰减
        # self.lr_decayed = tf.train.exponential_decay(
        #                                             learning_rate = 0.001,
        #                                             global_step = self.global_step,
        #                                             decay_steps = 40000,
        #                                             decay_rate = 0.8,
        #                                             staircase = True, # 台阶还是连续
        #                                             name = None)

        # 余弦退火衰减率CA1w0.8

        self.lr = lr
        self.t0 = t0
        self.t_m = tm
        self.m_m = mm
        self.lr_min = (1e-8)/self.lr

        self.lr_decayed = tf.train.cosine_decay_restarts(
                            learning_rate=self.lr,
                            global_step=self.global_step,  # 当前全局iter迭代次数，传入一个递增的variable
                            first_decay_steps=self.t0,  # 第一次衰减结束发生在哪一步，设一个epoch含有100个iter，可以设为2*100
                            t_mul=self.t_m,  # 后续的warm restarts衰减周期相较于前一次的倍率（周期越来越长）
                            m_mul=self.m_m,  # 每一次warm restarts学习率峰值的改变
                            alpha=self.lr_min,  # 学习率到最小衰减到lr*alpha，表示为learning_rate的分数
                            name=None)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_decayed)
        self.train_op = self.optimizer.minimize(self.loss)

        self.sess = tf_session()
        self.saver = tf.train.Saver()

    def train(self, epoch, save=True):
        batch_size = self.batch_size
        N_data = self.t_data.shape[0]
        N_eqns = self.t_eqns.shape[0]

        start_time = time.time()
        running_time = 0
        it = 0
        # 为什么以时间控制迭代呢？
        while it < epoch:
            # 以epoch 控制的搞法
            # 数据点比较少直接上完，方程点多分batch
            it_per_epoch = int(N_eqns / batch_size)
            for i in range(0, it_per_epoch):
                (t_data_batch, x_data_batch, y_data_batch,
                 u_data_batch, v_data_batch,) = (self.t_data, self.x_data, self.y_data,
                                                 self.u_data, self.v_data)

                (t_eqns_batch,
                 x_eqns_batch,
                 y_eqns_batch) = (self.t_eqns[i * batch_size:(i + 1) * batch_size, :],
                                  self.x_eqns[i * batch_size:(i + 1) * batch_size, :],
                                  self.y_eqns[i * batch_size:(i + 1) * batch_size, :])

                tf_dict = {self.t_data_tf: t_data_batch, self.x_data_tf: x_data_batch, self.y_data_tf: y_data_batch,
                           self.u_data_tf: u_data_batch, self.v_data_tf: v_data_batch,
                           self.t_eqns_tf: t_eqns_batch, self.x_eqns_tf: x_eqns_batch, self.y_eqns_tf: y_eqns_batch,
                           self.global_step: it}

                self.sess.run([self.train_op], tf_dict)

            [loss_value, loss_data_value, loss_eqns_value,
             loss_e1_value, loss_e2_value, loss_e3_value,
             learning_rate_value] = self.sess.run([self.loss, self.loss_data, self.loss_eqns,
                                                   self.loss_e1, self.loss_e2, self.loss_e3,
                                                   self.lr_decayed], tf_dict)

            self.a_ep.append(it)
            self.a_loss.append(loss_value)
            self.a_loss_data.append(loss_data_value)
            self.a_loss_e1.append(loss_e1_value)
            self.a_loss_e2.append(loss_e2_value)
            self.a_loss_e3.append(loss_e3_value)
            self.a_loss_eqns.append(loss_eqns_value)
            self.a_lr.append(learning_rate_value)

            # 前100步逐步输出, 100-10000步逢10输出, 10000步以后逢百输出
            if (it <= 100) or ((it % 10 == 0) and (100 < it <= 10000)) or ((it % 100 == 0) and (10000 < it <= epoch)):
                # if (it<=10) or ((it % 10 == 0)and(10<it<=100)) or ((it % 100 == 0)and(100<it<=epoch)): # 调试用
                elapsed = time.time() - start_time
                running_time += elapsed / 3600.0
                print('It: %d, Loss: %.3e, Time: %.2fs, Running Time: %.2fh, Learning Rate lr_decayed: %.1e'
                      % (it, loss_value, elapsed, running_time, learning_rate_value))
                sys.stdout.flush()  # 刷新输出
                start_time = time.time()
            it += 1
        # 结束训练 保存模型，在这里保存和结束后用saver保存是一样的效果
        # if save:
        #     self.saver.save(self.sess,
        #                     './A_TF1test2_tp10_X6Y5_t1-50_N5kT10_bch1w_L10N100_CA_1w/Trained_HFM_training/HFM_trained.ckpt')
        # print(tf.contrib.framework.get_variables_to_restore())

    def predict(self, x_pred, y_pred, t_pred, N, T):

        tf_dict = {self.t_data_tf: t_pred,
                   self.x_data_tf: x_pred,
                   self.y_data_tf: y_pred}

        # c_star = self.sess.run(self.c_data_pred, tf_dict)
        u_pred = self.sess.run(self.u_data_pred, tf_dict)
        v_pred = self.sess.run(self.v_data_pred, tf_dict)
        p_pred = self.sess.run(self.p_data_pred, tf_dict)

        x_pred = x_pred.reshape(N, T)[:, 0]
        y_pred = y_pred.reshape(N, T)[:, 0]
        t_pred = t_pred.reshape(N, T)[0, :]
        u_pred = u_pred.reshape(N, T)
        v_pred = v_pred.reshape(N, T)
        p_pred = p_pred.reshape(N, T)

        U_pred_dict = {'x_pred': x_pred,
                       'y_pred': y_pred,
                       't_pred': t_pred,
                       'u_pred': u_pred,
                       'v_pred': v_pred,
                       'p_pred': p_pred}

        return U_pred_dict


class data_process():
    def __init__(self, mat_path, Nx, Ny, Nt_true):
        time1 = time.time()
        self.data = io.loadmat(mat_path)  # local
        self.Nx = Nx
        self.Ny = Ny
        self.Nt = self.data['t'].shape[0]  # mat文件里dt=0.02的时间步数
        self.Nt_true = Nt_true  # 实际用于pinn训练的快照数
        self.t_jump = int(self.Nt / self.Nt_true)

        # self.t_all = self.data['t'][:, 0:self.Nt:self.t_jump]
        self.t_all = self.data['t'][0:self.Nt:self.t_jump,:]
        print(self.t_all.shape)
        self.x_all = self.data['x'].reshape([Nx * Ny, 1])  # 为了让它的shape从(?,)变成(?,1)
        self.y_all = self.data['y'].reshape([Nx * Ny, 1])
        self.u_all = self.data['u'][:, 0:self.Nt:self.t_jump]
        self.v_all = self.data['v'][:, 0:self.Nt:self.t_jump]
        self.p_all = self.data['p'][:, 0:self.Nt:self.t_jump]

        # 记录数据点/方程点，用于出图/验证
        self.x_smp, self.y_smp = self.x_all, self.y_all
        self.x_smp_eq, self.y_smp_eq = self.x_all, self.y_all
        # del self.data
        time2 = time.time()
        print("Raw data load done, costs {}s.".format(time2 - time1))

    def tp(self, tp1, tp2):
        self.tp1, self.tp2 = tp1, tp2
        t_tp = self.t_all[tp1:tp2,:]
        u_tp = self.u_all[:, tp1:tp2]  # data里取出N x T, 再取tp1-tp2时间批次: N x (tp2-tp1)
        v_tp = self.v_all[:, tp1:tp2]  # N x T
        p_tp = self.p_all[:, tp1:tp2]  # N x T
        print("Time split done.")

        return t_tp, u_tp, v_tp, p_tp

    def data_ext(self, t_all, u_all, v_all, p_all, id_ex):
        Nx, Ny = self.Nx, self.Ny
        x_ex = self.x_all.reshape(Ny, Nx)[id_ex[:, 1], id_ex[:, 0]]
        y_ex = self.y_all.reshape(Ny, Nx)[id_ex[:, 1], id_ex[:, 0]]
        u_ex = u_all.reshape(Ny, Nx, u_all.shape[1])[id_ex[:, 1], id_ex[:, 0], :]
        v_ex = v_all.reshape(Ny, Nx, v_all.shape[1])[id_ex[:, 1], id_ex[:, 0], :]
        # p_ex = p_all.reshape(Ny, Nx, p_all.shape[1])[id_ex[:, 1], id_ex[:, 0], :]

        N_ex, Nt_ex = x_ex.shape[0], t_all.shape[0]
        x_data = np.tile(x_ex.reshape(-1, 1), (1, Nt_ex)).flatten()[:, None].astype(np.float32)
        y_data = np.tile(y_ex.reshape(-1, 1), (1, Nt_ex)).flatten()[:, None].astype(np.float32)
        t_data = np.tile(t_all, (1, N_ex)).T.flatten()[:, None].astype(np.float32)

        u_data = u_ex.flatten()[:, None].astype(np.float32)
        v_data = v_ex.flatten()[:, None].astype(np.float32)

        # X_data = tf.stack([x_data, y_data, t_data], axis=1)
        # U_data = tf.stack([u_data, v_data], axis=1)

        self.x_smp = x_ex  # 记录数据点，用于出图/验证
        self.y_smp = y_ex
        print("Arbitrary data extract by id_ex is done.")
        return x_data, y_data, t_data, u_data, v_data

    def data_uniform(self, t_all, u_all, v_all, p_all, data_Nx, data_Ny, data_Nt):
        idx = np.linspace(0, self.Nx - 1, num=data_Nx, endpoint=True, dtype='int')
        idy = np.linspace(0, self.Ny - 1, num=data_Ny, endpoint=True, dtype='int')
        idt = np.linspace(0, t_all.shape[0] - 1, num=data_Nt, endpoint=True, dtype='int')

        Nx_uni, Nt_uni = idx.shape[0] * idy.shape[0], idt.shape[0]
        t_uni = t_all[idt]
        x_uni = self.x_all.reshape(self.Ny, self.Nx)[idy, :][:, idx]
        y_uni = self.y_all.reshape(self.Ny, self.Nx)[idy, :][:, idx]
        u_uni = u_all.reshape(self.Ny, self.Nx, t_all.shape[0])[idy, :, :][:, idx, :][:, :, idt]
        v_uni = v_all.reshape(self.Ny, self.Nx, t_all.shape[0])[idy, :, :][:, idx, :][:, :, idt]

        x_data = np.tile(x_uni.reshape(-1, 1), (1, Nt_uni)).flatten()[:, None].astype(np.float32)
        y_data = np.tile(y_uni.reshape(-1, 1), (1, Nt_uni)).flatten()[:, None].astype(np.float32)
        t_data = np.tile(t_uni, (1, Nx_uni)).T.flatten()[:, None].astype(np.float32)
        u_data = u_uni.flatten()[:, None].astype(np.float32)
        v_data = v_uni.flatten()[:, None].astype(np.float32)

        # X_data = tf.stack([x_data, y_data, t_data], axis=1)
        # U_data = tf.stack([u_data, v_data], axis=1)

        self.x_smp = x_uni  # 记录数据点，用于出图/验证
        self.y_smp = y_uni
        print("Uniform data distribute in x,y is done.")

        return x_data, y_data, t_data, u_data, v_data

    def eqns(self, t_all, eqns_Nx, eqns_Ny, eqns_Nt):
        idx = np.linspace(0, self.Nx - 1, num=eqns_Nx, endpoint=True, dtype='int')
        idy = np.linspace(0, self.Ny - 1, num=eqns_Ny, endpoint=True, dtype='int')
        idt = np.linspace(0, t_all.shape[0] - 1, num=eqns_Nt, endpoint=True, dtype='int')

        t_eq = t_all[idt]
        x_eq = self.x_all.reshape(self.Ny, self.Nx)[idy, :][:, idx]
        y_eq = self.y_all.reshape(self.Ny, self.Nx)[idy, :][:, idx]

        x_eqns = np.tile(x_eq.reshape(-1, 1), (1, eqns_Nt)).flatten()[:, None]
        y_eqns = np.tile(y_eq.reshape(-1, 1), (1, eqns_Nt)).flatten()[:, None]
        t_eqns = np.tile(t_eq, (1, eqns_Nx * eqns_Ny)).T.flatten()[:, None]
        # X_eqns = tf.stack([x_eqns, y_eqns, t_eqns], axis=1)

        self.x_smp_eq = x_eq
        self.y_smp_eq = y_eq

        return x_eqns, y_eqns, t_eqns

    def data_pred(self,t1,t2,t_jump=1):
        N = self.Nx * self.Ny
        T = int((t2 - t1)/t_jump)
        x_pred = np.tile(self.x_all.reshape(-1, 1), (1, T)).flatten()[:, None].astype(np.float32)
        y_pred = np.tile(self.y_all.reshape(-1, 1), (1, T)).flatten()[:, None].astype(np.float32)
        t_pred = np.tile(self.data['t'][t1:t2:t_jump], (1, N)).T.flatten()[:, None].astype(np.float32)
        # X_pred = tf.stack([x_pred, y_pred, t_pred], axis=1)

        return x_pred, y_pred, t_pred, N, T

    def data_pred_interpolate(self,t1,t2,Nt):
        N = self.Nx * self.Ny
        T_in = np.linspace(self.data['t'][t1],self.data['t'][t2],num=Nt,endpoint=False)
        T = len(T_in)
        x_pred = np.tile(self.x_all.reshape(-1, 1), (1, T)).flatten()[:, None].astype(np.float32)
        y_pred = np.tile(self.y_all.reshape(-1, 1), (1, T)).flatten()[:, None].astype(np.float32)
        t_pred = np.tile(T_in, (1, N)).T.flatten()[:, None].astype(np.float32)
        return x_pred, y_pred, t_pred, N, T

    def predict_all(self, model, x_all, y_all, t_all):
        N = x_all.shape[0]
        T = t_all.shape[0]

        x_pred = tf.constant(np.tile(x_all.reshape(-1, 1), (1, T)).flatten(), dtype=tf.float32)
        y_pred = tf.constant(np.tile(y_all.reshape(-1, 1), (1, T)).flatten(), dtype=tf.float32)
        t_pred = tf.constant(np.tile(t_all, (1, N)).T.flatten(), dtype=tf.float32)
        X_pred = tf.stack([x_pred, y_pred, t_pred], axis=1)

        U_pred = model(X_pred).numpy()
        u_pred = U_pred[:, 0].reshape(N, T)
        v_pred = U_pred[:, 1].reshape(N, T)
        p_pred = U_pred[:, 2].reshape(N, T)

        U_pred_dict = {'x_pred': x_all,
                       'y_pred': y_all,
                       't_pred': t_all,
                       'u_pred': u_pred,
                       'v_pred': v_pred,
                       'p_pred': p_pred}
        return U_pred_dict

    # 弃用
    def data_pred_all(self, t_pred):
        """生成预测用的输入"""
        Nt_pred = t_pred.shape[0]
        x_prd = tf.constant(np.tile(self.x_all, (1, Nt_pred)).flatten(), dtype=tf.float32)
        y_prd = tf.constant(np.tile(self.y_all, (1, Nt_pred)).flatten(), dtype=tf.float32)
        t_prd = tf.constant(np.tile(t_pred, (1, self.Nx * self.Ny)).T.flatten(), dtype=tf.float32)

        X_pred = tf.stack([x_prd, y_prd, t_prd], axis=1)
        return X_pred

    # 弃用
    def data_pred_step(self, t_pred, i):
        Nt_pred = t_pred.shape[0]
        x_prd = tf.constant(self.x_all.flatten(), dtype=tf.float32)
        y_prd = tf.constant(self.y_all.flatten(), dtype=tf.float32)
        t_prd = tf.constant(np.tile(t_pred[i], (1, self.Nx * self.Ny)).T.flatten(), dtype=tf.float32)

        X_pred_i = tf.stack([x_prd, y_prd, t_prd], axis=1)
        return X_pred_i

    def plot_data(self):
        """ 在绘制单个流场的基础上，画出测点的散点图 """
        PyPOD.plot_smp(self.x_all, self.y_all, self.Nx, self.Ny,
                       self.u_all[:, 0:1], self.x_smp, self.y_smp,
                       dpi=40, figsize=(12, 6))

        # return

    def plot_eqns(self):
        PyPOD.plot_smp(self.x_all, self.y_all, self.Nx, self.Ny,
                       self.u_all[:, 0:1], self.x_smp_eq, self.y_smp_eq,
                       dpi=40, figsize=(12, 6))
