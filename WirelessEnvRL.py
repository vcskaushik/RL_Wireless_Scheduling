import numpy as np
import scipy.special as sp
import itertools
#from utility import QAM,ML_decoder
import gym
from gym import spaces


def QAM(M,data,normalize=True,return_const=False):
    if M==4:
        qam_const = np.array([-1.0 + 1.0j, -1.0 - 1.0j, 1.0 + 1.0j, 1.0 - 1.0j])
        
    elif M == 16:
        qam_const = np.array([-3.0 + 3.0j, -3.0 + 1.0j, -3.0 - 3.0j, -3.0 - 1.0j
                              ,-1.0 + 3.0j, -1.0 + 1.0j, -1.0 - 3.0j, -1.0 - 1.0j
                              , 3.0 + 3.0j, 3.0 + 1.0j, 3.0 - 3.0j, 3.0 - 1.0j
                              , 1.0 + 3.0j, 1.0 + 1.0j, 1.0 - 3.0j, 1.0 - 1.0j])
    
    else:
        qam_const = np.array([-7.0 + 7.0j, -7.0 + 5.0j, -7.0 + 1.0j, -7.0 + 3.0j
                              , -7.0 - 7.0j, -7.0 - 5.0j, -7.0 - 1.0j, -7.0 - 3.0j
                              , -5.0 + 7.0j, -5.0 + 5.0j, -5.0 + 1.0j, -5.0 + 3.0j
                              , -5.0 - 7.0j, -5.0 - 5.0j, -5.0 - 1.0j, -5.0 - 3.0j
                              , -1.0 + 7.0j, -1.0 + 5.0j, -1.0 + 1.0j, -1.0 + 3.0j
                              , -1.0 - 7.0j, -1.0 - 5.0j, -1.0 - 1.0j, -1.0 - 3.0j
                              , -3.0 + 7.0j, -3.0 + 5.0j, -3.0 + 1.0j, -3.0 + 3.0j
                              , -3.0 - 7.0j, -3.0 - 5.0j, -3.0 - 1.0j, -3.0 - 3.0j
                              , 7.0 + 7.0j, 7.0 + 5.0j, 7.0 + 1.0j, 7.0 + 3.0j
                              , 7.0 - 7.0j, 7.0 - 5.0j, 7.0 - 1.0j, 7.0 - 3.0j
                              , 5.0 + 7.0j, 5.0 + 5.0j, 5.0 + 1.0j, 5.0 + 3.0j
                              , 5.0 - 7.0j, 5.0 - 5.0j, 5.0 - 1.0j, 5.0 - 3.0j
                              , 1.0 + 7.0j, 1.0 + 5.0j, 1.0 + 1.0j, 1.0 + 3.0j
                              , 1.0 - 7.0j, 1.0 - 5.0j, 1.0 - 1.0j, 1.0 - 3.0j
                              , 3.0 + 7.0j, 3.0 + 5.0j, 3.0 + 1.0j, 3.0 + 3.0j
                              , 3.0 - 7.0j, 3.0 - 5.0j, 3.0 - 1.0j, 3.0 - 3.0j])
    b = int(np.log2(M))

    if normalize==True:
        n = int(np.log2(M)/2)
        qam_var = 1/(2**(n-2))*np.sum(np.linspace(1,2**n-1, 2**(n-1))**2)
        qam_const /= np.sqrt(qam_var)
        
    if len(data.shape) == 1:
        data_qam_ind = np.reshape(data,(-1,b))
        data_qam_ind = np.squeeze(np.dot(data_qam_ind,1 << np.arange(b - 1, -1, -1))).astype(int)
        data_qam = qam_const[data_qam_ind]
    elif (len(data.shape)==2):
        if data.shape[1]>1:
            data_qam_ind = np.reshape(data,(data.shape[0],-1,b))
            data_qam_ind = np.squeeze(np.dot(data_qam_ind,1 << np.arange(b - 1, -1, -1))).astype(int)
            #print(data_qam_ind.shape)
            data_qam = qam_const[data_qam_ind]
        else:
            data_qam_ind = np.reshape(data,(-1,b))
            data_qam_ind = np.squeeze(np.dot(data_qam_ind,1 << np.arange(b - 1, -1, -1))).astype(int)
            data_qam = qam_const[data_qam_ind]
            
            
    if return_const==True :
        return data_qam,qam_const
    else:
        return data_qam
        

def ML_decoder(y_deint,Codebook): #,h_deint
    Y_dec = []
    for i in range(y_deint.shape[0]):
        y_err = np.sum(np.abs(Codebook-y_deint[i,:])**2,axis=1) #h_deint[i,:]*
        y_dec = Codebook[np.argmin(y_err),:]
        Y_dec.append(y_dec)
    return np.stack(Y_dec)



class WirelessEnv(gym.Env):
    """
      Custom Environment that follows gym interface.
      This is a simple env where the agent must learn to go always left. 
      """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': [None]}
    # Define constants for clearer code
    P0_M4 = 0
    P0_M16 = 1
    P0_M64 = 2
    
    P0_5_M4 = 3
    P0_5_M16 = 4
    P0_5_M64 = 5
    
    P1_M4 = 6
    P1_M16 = 7
    P1_M64 = 8
    
    P2_M4 = 9
    P2_M16 = 10
    P2_M64 = 11
    
    P5_M4 = 12
    P5_M16 = 13
    P5_M64 = 14
    
    P10_M4 = 15
    P10_M16 = 16
    P10_M64 = 17
    
    

    def __init__(self,name="WirelessEnv"):
        super(WirelessEnv, self).__init__()

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: left and right
        n_actions = 18
        self.action_space = spaces.Discrete(n_actions)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        #self.observation_space = spaces.Box(low=0, high=self.grid_size,shape=(1,), dtype=np.float32)
        self.name = name
        self.nT = 200
        self.R_t_list = self.gen_channel_correlation()
        self.G , self.CodeBooks = self.Encoder_init()
        self.p_on = 10.0
        self.alpha = 0.9 
        
        
        self.V_list = [5,10,30]
        self.V_tp_list = [[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8]]
        self.P_list = [0,0.5,1,2.5,5,10]
        self.M_list = [4,16,64]
        self.buffer_sizes_list = np.array([2,4,6,8,10,12])*150*8
        self.step_no = None
        self.step_max = 25
        self.overflow_cost = 2
        
        self.env_Channel = None
        self.env_Rt = None
        self.env_Vind = None
        self.env_Vtp = None
        self.env_buffer = None
        self.env_bsize = None
        self.env_Power = None
        
        self.agent_M = None
        self.agent_Power = None
        
        

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array) 
        """
        self.step_no = 0
        self.env_Vind = 0
        self.env_Vtp = self.V_tp_list[self.env_Vind]
        self.env_Rt = self.R_t_list[self.env_Vind]
        
        self.env_Channel = self.gen_channel()
        self.env_Power = self.p_on
        
        self.init_buffer()
        self.env_bsize = self.env_buffer.size/self.bsize_init
        
        Channel_state = np.expand_dims(self.env_Channel.view(float),axis=0)
        Power_Buffer_state = np.array([[self.env_Power,self.env_bsize,self.env_Vind]])
        
        
        init_state = [Channel_state,Power_Buffer_state]
        
        return init_state
    
    def gen_channel_correlation(self):
        fc = 5e9
        W = 2e5
        del_t = 1/W
        V = np.array([5,15,30])#100/3.6 # m/s conversion
        T_c = 3e8/(2*V*fc) # Coherence Time
        rate = 0.5
        
        # Time correlation matrix
        R_t_list = []
        for v in V:
            temp_nT = np.expand_dims(np.arange(1,self.nT+1),axis=0) # row vector from 1 to nT
            ik_mat1 = np.dot(temp_nT.T,(np.ones((1,self.nT)))) - np.dot(np.ones((self.nT,1)),temp_nT)
            c = 3e8
            R_t_list += [sp.j0(2*np.pi*(v/c)*fc*del_t*ik_mat1)]
        
        return R_t_list
    
    def gen_channel(self):
        # channel generation
        h = np.random.multivariate_normal(np.zeros(self.nT),self.R_t_list[self.env_Vind])            +1j*np.random.multivariate_normal(np.zeros(self.nT),self.R_t_list[self.env_Vind])
            
        h = np.expand_dims(np.array(h),axis=1)
        
        return h
    
    def init_buffer(self):
        self.bsize_init = np.random.choice(self.buffer_sizes_list)
        self.env_buffer = np.random.randint(0,1,(self.bsize_init))
        
        
        
        
    def Encoder_init(self):
        P = np.ones((4,4))
        P[np.diag_indices(4)] = 0
        P2 = np.array(P)
        G = np.append(np.eye(4),P,axis=1)
        Msgs = np.array(list(itertools.product([0, 1], repeat=4)))
        Msgs_enc = np.mod(Msgs@G,2)

        QAM64_code_ind = np.array(list(itertools.product(list(range(16)), repeat=3)))
        CW_enc = Msgs_enc[QAM64_code_ind].reshape(QAM64_code_ind.shape[0],-1)
 
        CodeBooks = [QAM(4,Msgs_enc),QAM(16,Msgs_enc),QAM(64,CW_enc)]
    
        return G,CodeBooks
    
        
        
    def Transmitter(self):
        n_cw = 8
        k_cw = 4
        bits_per_symb = int(np.log2(self.agent_M))
        ncw_bits = self.nT*bits_per_symb
        nmsg_bits = int(ncw_bits*k_cw/n_cw)
        int_length = int(ncw_bits/n_cw)
        #print(n_cw,k_cw,self.agent_M,ncw_bits,nmsg_bits,int_length)
        
        if nmsg_bits > self.env_buffer.size:
            x_temp = np.random.randint(0,1,(nmsg_bits - self.env_buffer.size))
            x = np.append(self.env_buffer,x_temp)                                     
        else:
            x = self.env_buffer[:nmsg_bits]
        
        x = x.reshape(int_length,k_cw)
        
        #print(x.shape,self.G.shape)
        
        if (self.agent_M==4)or(self.agent_M==16):
            qam_var = 1*n_cw
        if self.agent_M==64:
            qam_var = 3*n_cw
    
        out_var = int(qam_var/bits_per_symb)

        
        x_enc = np.mod(x@self.G,2)
        x_enc_qam = QAM(self.agent_M,np.reshape(x_enc,(-1,qam_var)))
        x_tx = np.reshape(x_enc_qam,(self.nT,1),order='F')

        # QPSK symbols generation
        w = (np.random.randn(self.nT,1)+1j*np.random.randn(self.nT,1))*np.sqrt(0.5*1)
        w = np.array(w)

        # FADING channel
        y = (self.env_Channel*np.sqrt(self.agent_Power)*x_tx + w)*np.exp(-1j*np.angle(self.env_Channel))
        
        return y,x_enc_qam
        
        
        
    def Receiver(self,y,x_enc_qam):
        n_cw = 8
        k_cw = 4
        bits_per_symb = int(np.log2(self.agent_M))
        ncw_bits = self.nT*bits_per_symb
        nmsg_bits = int(ncw_bits*k_cw/n_cw)
        int_length = int(ncw_bits/n_cw)
        
        qam_var = 1*n_cw
        
        if (self.agent_M==4):
            Codebook_QAM = self.CodeBooks[0]
        if (self.agent_M==16):
            Codebook_QAM = self.CodeBooks[1]
        if self.agent_M==64:
            Codebook_QAM = self.CodeBooks[2]
            qam_var = 3*n_cw
    
        out_var = int(qam_var/bits_per_symb)
        
        
        y_deint = np.reshape(y,(-1,out_var),order='F') 
        y_dec = ML_decoder(y_deint,Codebook_QAM)

        err1 = np.sum(np.array(y_dec!=x_enc_qam,dtype='float'),axis=1)

        #error updation
        symerr = np.sum(np.array(err1>0,dtype='float'))
        blockerr = symerr!=0
        
        if blockerr==0:
            self.env_buffer = np.delete(self.env_buffer,np.arange(nmsg_bits))
            
        return symerr,blockerr
        


    def step(self, action):
        
        if action > 17:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))
        
            
        self.agent_M = self.M_list[action%3]
        self.agent_Power = self.P_list[int(np.floor(action/3))]
            
        y,x_enc_qam = self.Transmitter()
        
        symerr,blockerr = self.Receiver(y,x_enc_qam)
        
        self.env_Power = self.env_Power + self.agent_Power/(self.alpha*self.P_list[-1])
        
        next_V = np.random.choice(np.arange(len(self.V_list)),p=self.env_Vtp)
        
        
        if next_V == self.env_Vind:
            self.env_Channel = self.gen_channel()
        else:
            self.env_Vind = next_V 
            self.env_Vtp = self.V_tp_list[self.env_Vind]
            self.env_Rt = self.R_t_list[self.env_Vind]
            self.env_Channel = self.gen_channel()
            
        self.env_bsize = self.env_buffer.size/self.bsize_init
        
        Channel_state = np.expand_dims(self.env_Channel.view(float),axis=0)
        Power_Buffer_state = np.array([[self.env_Power,self.env_bsize,self.env_Vind]])
        
        next_state = [Channel_state,Power_Buffer_state]
        

        # Terminal State is when Buffer is empty
        done = bool(self.env_buffer.size == 0)

        # Null reward everywhere except when reaching the goal (left of the grid)
        power_cost = self.agent_Power/(self.P_list[-1])
        buffer_cost = (self.step_no > self.step_max)*self.overflow_cost + self.env_bsize 
        
        reward = -1*(power_cost+buffer_cost)

        # Optionally we can pass additional info, we are not using that for now
        info = {'Power Cost':power_cost,'Buffer Cost': buffer_cost}

        return next_state, reward, done, info

    def close(self):
        pass
    