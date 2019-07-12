import scipy
import time
import pandas
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import BatchNormalization
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils.training_utils import multi_gpu_model
from keras.layers import Conv1D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from scipy.optimize import least_squares
from tqdm import tqdm
import h5py

class SHO_data:
    
    def __init__(self, file, x, y):
     
        self.file=file
        self.model='network has not been loaded'
        self.raw_data='data has not been loaded'
              
        self.freq_vector='data has not been loaded'
        self.x=x
        self.y=y
        
        self.ls_fit={'phase':'none','amplitude':'none','Q':'none','resonant_frequency':'none'}
        self.DNN_fit={'phase':'none','amplitude':'none','Q':'none','resonant_frequency':'none'}
        self.hybrid_fit={'phase':'none','amplitude':'none','Q':'none','resonant_frequency':'none'}
    
    ###############################################################################################################################
    # TECHNICAL FUNCTIONS
    ###############################################################################################################################
    def SHO_fit_func(self, *parms):
        #SHO Fit function
        
        Amp, w_0, Q, phi = parms
        wvec=self.freq_vector
        w0 = 340E5
        wmax = 360E5
        Amp = 1E-3 * Amp
        w_0 = w0 + w_0*1E8*wmax #Scale it up
        phi = -1*np.pi+2*np.pi*phi #Scale it up
        Q = Q*100 #Scale it up
        wvec = wvec*1E8*wmax + w0
        func = Amp * np.exp(1.j * phi) * w_0 ** 2 / (wvec ** 2 - 1j * wvec * w_0 / Q - w_0 ** 2)
        noise=0.0025*np.random.uniform(0,1)
        func=func+noise*1j*np.random.normal(0,1,len(wvec))+noise*np.random.normal(0,1,len(wvec))

        return func

    def SHO_fit_func_no_noise(self, *parms):
        #SHO Fit function
        wvec=self.freq_vector
        Amp, w_0, Q, phi = parms
        w0 = 340E5
        wmax = 360E5
        Amp = 1E-3 * Amp
        w_0 = w0 + w_0*1E8*wmax #Scale it up
        phi = -1*np.pi+2*np.pi*phi #Scale it up
        Q = Q*100 #Scale it up
        wvec = wvec*1E8*wmax + w0
        func = Amp * np.exp(1.j * phi) * w_0 ** 2 / (wvec ** 2 - 1j * wvec * w_0 / Q - w_0 ** 2)
        noise=0.0025*np.random.uniform(0,1)*0
        func=func+noise*1j*np.random.normal(0,1,len(wvec))+noise*np.random.normal(0,1,len(wvec))

        return func
    

    def return_split_data(self,input_mat):
        #Given a 2d matrix with complex values return a real and imag matrix
        a,b = input_mat.shape
        final_mat = np.zeros(shape=(a,b,2))
        final_mat[:,:,0] = np.real(input_mat)
        final_mat[:,:,1] = np.imag(input_mat)
    

        return final_mat
    
 
    
    ###############################################################################################################################
    # LOADING DATA
    ###############################################################################################################################
   
    def load_raw_data(self):
        self.raw_data = h5py.File(self.file, 'r')
        self.freq_vector = np.linspace(0, 1, len(self.raw_data['Measurement_000']['Channel_000']['Bin_Frequencies']))
        normalization=np.max(np.abs(self.raw_data['Measurement_000']['Channel_000']['Raw_Data']))/0.01
        h5py.File(self.file, 'r').close()
        
    def load_model(self,model_name):
        from keras.models import load_model
        self.model=keras.models.load_model(model_name)
    
    def create_model(self):
        model = Sequential()
        model.add(Conv1D(128, activation = 'relu',
                         input_shape = (len(self.freq_vector),2,), kernel_size = (15)))
        model.add(Conv1D(64, activation = 'relu',
                         input_shape = (len(self.freq_vector),2,), kernel_size = (5)))
        model.add(Dense(512, activation = 'relu'))
        model.add(Dense(512, activation = 'relu'))
        model.add(Dense(128, activation = 'relu'))
        model.add(Dense(128, activation = 'relu'))
        model.add(Flatten())
        model.add(Dense(128, activation = 'relu'))
        model.add(Dense(128, activation = 'relu'))
        model.add(Dense(4, activation='relu'))
        #model = multi_gpu_model(model, gpus = 4)
        model.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])
        self.model=model
        
    def fit_model(self):
        wvec=self.freq_vector 
        def myGenerator(batch_num, max_batch_num, num_curves, wvec): 
            while True:
                if batch_num>=max_batch_num:
                    break

                func_results= np.zeros(shape =(int(num_curves), len(wvec)), dtype = np.complex64)
                func_parms = np.zeros(shape = (int(num_curves), 4))
                A_range = [0, 1]
                Q_range = [0,1]
                w_range = [min(wvec) + 0.1*len(wvec)*(wvec[1] - wvec[0]), max(wvec) - 0.1*len(wvec)*(wvec[1] - wvec[0])]
                phi_range = [0, 1.0]

                for i in range(num_curves):
                    parms = [np.random.uniform(low = A_range[0], high = A_range[1]),
                            np.random.uniform(low = w_range[0], high = w_range[1]),
                            np.random.uniform(low = Q_range[0], high = Q_range[1]),
                            np.random.uniform(low = phi_range[0], high = phi_range[1])]
                    #Get back the function
                    func_results[i,:] = self.SHO_fit_func(*parms)
                    func_parms[i,:] = parms

                yield func_results, func_parms
            batch_num=batch_num+1
                
        data_gen = myGenerator(0,50, 100000, wvec) #initialize the generator object        
        for X,y in data_gen:
            X = self.return_split_data(X)
            X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 32, test_size = 0.20)
            self.model.fit(X_train, y_train, validation_data=(X_test, y_test),verbose=1)
    
        self.model.save_weights(str(file)+'_model_new.h5')
        
    def load_weights(self,weights_file):
        self.model.load_weights(weights_file)   
        
   
    ###############################################################################################################################
    # FITTING DATA
    ###############################################################################################################################        
    
    def do_DNN_fitting(self):
        w_rs=np.zeros(self.x*self.y)
        Qs=np.zeros(self.x*self.y)
        phases=np.zeros(self.x*self.y)
        amplitudes=np.zeros(self.x*self.y)
        normalization=np.max(np.abs(self.raw_data['Measurement_000']['Channel_000']['Raw_Data']))/0.01
        
        for i in tqdm(range(self.x*self.y)):
            ddata=self.raw_data['Measurement_000']['Channel_000']['Raw_Data'][i]/normalization
            sho_ex = ddata
            sho_ex_mat = self.return_split_data(sho_ex[None,:])
            predicted_parms_DNN = self.model.predict(sho_ex_mat)[0]

            amplitudes[i]=predicted_parms_DNN[0]    
            w_rs[i]=predicted_parms_DNN[1]
            Qs[i]=predicted_parms_DNN[2]
            phases[i]=predicted_parms_DNN[3]
            
            self.DNN_fit['amplitude']=np.reshape(amplitudes,(self.x,self.y))
            self.DNN_fit['resonant_frequency']=np.reshape(w_rs,(self.x,self.y))
            self.DNN_fit['Q']=np.reshape(Qs,(self.x,self.y))
            self.DNN_fit['phase']=np.reshape(phases,(self.x,self.y))
        
    def do_LS_fitting(self):
        
        def SHO_fit_flattened(wvec,p):

            Amp, w_0, Q, phi=p[0],p[1],p[2],p[3]

            w0 = 340E5
            wmax = 360E5

            Amp = 1E-3 * Amp
            w_0 = w0 + w_0*1E8*wmax #Scale it up
            phi = -1*np.pi+2*np.pi*phi #Scale it up
            Q = Q*100 #Scale it up
            wvec = wvec*1E8*wmax + w0
            func = Amp * np.exp(1.j * phi) * w_0 ** 2 / (wvec ** 2 - 1j * wvec * w_0 / Q - w_0 ** 2)

            noise=0.0025*np.random.uniform(0,1)*0
            func=func+noise*1j*np.random.normal(0,1,len(wvec))+noise*np.random.normal(0,1,len(wvec))

            return np.hstack([np.real(func),np.imag(func)])
        
        def fun(x,u,y):
            return SHO_fit_flattened(u, x) - y
        
        wvec=self.freq_vector 
        w_rs=np.zeros(self.x*self.y)
        Qs=np.zeros(self.x*self.y)
        phases=np.zeros(self.x*self.y)
        amplitudes=np.zeros(self.x*self.y)
        normalization=np.max(np.abs(self.raw_data['Measurement_000']['Channel_000']['Raw_Data']))/0.01
        
        for i in tqdm(range(self.x*self.y)):
            ddata=self.raw_data['Measurement_000']['Channel_000']['Raw_Data'][i]/normalization
            u=wvec
            y = np.hstack([np.real(ddata),np.imag(ddata)])
            x0 = np.array([0.5,0.5,0.5,0.5])
            predicted_parms_LS=least_squares(fun, x0, args=(u, y), verbose=0).x

            amplitudes[i]=predicted_parms_LS[0]    
            w_rs[i]=predicted_parms_LS[1]
            Qs[i]=predicted_parms_LS[2]
            phases[i]=predicted_parms_LS[3]
            
            self.ls_fit['amplitude']=np.reshape(amplitudes,(self.x,self.y))
            self.ls_fit['resonant_frequency']=np.reshape(w_rs,(self.x,self.y))
            self.ls_fit['Q']=np.reshape(Qs,(self.x,self.y))
            self.ls_fit['phase']=np.reshape(phases,(self.x,self.y))
        
    def do_hybrid_fitting(self):
                
        def SHO_fit_flattened(wvec,p):

            Amp, w_0, Q, phi=p[0],p[1],p[2],p[3]

            w0 = 340E5
            wmax = 360E5

            Amp = 1E-3 * Amp
            w_0 = w0 + w_0*1E8*wmax #Scale it up
            phi = -1*np.pi+2*np.pi*phi #Scale it up
            Q = Q*100 #Scale it up
            wvec = wvec*1E8*wmax + w0
            func = Amp * np.exp(1.j * phi) * w_0 ** 2 / (wvec ** 2 - 1j * wvec * w_0 / Q - w_0 ** 2)

            noise=0.0025*np.random.uniform(0,1)*0
            func=func+noise*1j*np.random.normal(0,1,len(wvec))+noise*np.random.normal(0,1,len(wvec))

            return np.hstack([np.real(func),np.imag(func)])
        
        def fun(x,u,y):
            return SHO_fit_flattened(u, x) - y
        
        wvec=self.freq_vector 
        w_rs=np.zeros(self.x*self.y)
        Qs=np.zeros(self.x*self.y)
        phases=np.zeros(self.x*self.y)
        amplitudes=np.zeros(self.x*self.y)
        normalization=np.max(np.abs(self.raw_data['Measurement_000']['Channel_000']['Raw_Data']))/0.01
        
        for i in tqdm(range(self.x*self.y)):
            ddata=self.raw_data['Measurement_000']['Channel_000']['Raw_Data'][i]/normalization
            sho_ex = ddata
            sho_ex_mat = self.return_split_data(sho_ex[None,:])
            predicted_parms = self.model.predict(sho_ex_mat)[0]
            u=wvec
            y=np.hstack([np.real(sho_ex),np.imag(sho_ex)])
            x0 = np.array([predicted_parms[0],
                           predicted_parms[1],
                           predicted_parms[2],
                           predicted_parms[3]])
           
            if x0[2]==0:
                x0[2]=0.1
            elif x0[2]==1:
                x0[2]=0.9
            else:
                x0[2]=x0[2]
            
            if x0[3]>1:
                x0[3]=1
            elif x0[3]<0:
                x0[3]=0
            else:
                x0[3]=x0[3]


            if x0[0]>1:
                x0[0]=1
            elif x0[0]<0:
                x0[0]=0.1
            else:
                x0[0]=x0[0]   
            
            
            try:
                predicted_parms_hybrid=least_squares(fun, x0, args=(u, y),bounds=(0,1),verbose=0).x
            except ValueError:
                print(x0)
            amplitudes[i]=predicted_parms_hybrid[0]    
            w_rs[i]=predicted_parms_hybrid[1]
            Qs[i]=predicted_parms_hybrid[2]
            phases[i]=predicted_parms_hybrid[3]
            
            self.hybrid_fit['amplitude']=np.reshape(amplitudes,(self.x,self.y))
            self.hybrid_fit['resonant_frequency']=np.reshape(w_rs,(self.x,self.y))
            self.hybrid_fit['Q']=np.reshape(Qs,(self.x,self.y))
            self.hybrid_fit['phase']=np.reshape(phases,(self.x,self.y))
    
    def single_fit(self,index):
          
        def SHO_fit_flattened(wvec,p):

            Amp, w_0, Q, phi=p[0],p[1],p[2],p[3]

            w0 = 340E5
            wmax = 360E5

            Amp = 1E-3 * Amp
            w_0 = w0 + w_0*1E8*wmax #Scale it up
            phi = -1*np.pi+2*np.pi*phi #Scale it up
            Q = Q*100 #Scale it up
            wvec = wvec*1E8*wmax + w0
            func = Amp * np.exp(1.j * phi) * w_0 ** 2 / (wvec ** 2 - 1j * wvec * w_0 / Q - w_0 ** 2)

            noise=0.0025*np.random.uniform(0,1)*0
            func=func+noise*1j*np.random.normal(0,1,len(wvec))+noise*np.random.normal(0,1,len(wvec))

            return np.hstack([np.real(func),np.imag(func)])
        def fun(x,u,y):
            return SHO_fit_flattened(u, x) - y
        
        wvec=self.freq_vector 
        normalization=np.max(np.abs(self.raw_data['Measurement_000']['Channel_000']['Raw_Data']))/0.01
        i=index
        ddata=self.raw_data['Measurement_000']['Channel_000']['Raw_Data'][i]/normalization   
        ####   
        sho_ex = ddata
        sho_ex_mat = self.return_split_data(sho_ex[None,:])
        predicted_parms_DNN = self.model.predict(sho_ex_mat)[0]
        ####
        u=wvec
        y = np.hstack([np.real(ddata),np.imag(ddata)])
        x0 = np.array([0.5,0.5,0.5,0.5])
        predicted_parms_LS=least_squares(fun, x0, args=(u, y), bounds=(0,1),verbose=0).x
        ####
        sho_ex = ddata
        sho_ex_mat = self.return_split_data(sho_ex[None,:])
        predicted_parms = self.model.predict(sho_ex_mat)[0]
        u=wvec
        y=np.hstack([np.real(sho_ex),np.imag(sho_ex)])
        x0 = np.array([predicted_parms[0],
                       predicted_parms[1],
                       predicted_parms[2],
                       predicted_parms[3]])
        if x0[2]==0:
                x0[2]=0.1
        elif x0[2]==1:
                x0[2]=0.9
        else:
                x0[2]=x0[2]
                
        if x0[3]>1:
            x0[3]=1
        elif x0[3]<0:
            x0[3]=0
        else:
            x0[3]=x0[3]
            
            
        if x0[0]>1:
            x0[0]=1
        elif x0[0]<0:
            x0[0]=0.1
        else:
            x0[0]=x0[0]
            
        predicted_parms_hybrid=least_squares(fun, x0, args=(u, y), bounds=(0,1),verbose=0).x
        ####
        
        predicted_sho_DNN = self.SHO_fit_func_no_noise(*predicted_parms_DNN)
        predicted_sho_LS = self.SHO_fit_func_no_noise(*predicted_parms_LS)
        predicted_sho_hybrid = self.SHO_fit_func_no_noise(*predicted_parms_hybrid)

        plt.figure()
        plt.xlabel('Relative frequency')
        plt.ylabel('Real part')
        plt.plot(wvec, np.real(sho_ex), 'k:')
        plt.plot(wvec, np.real(predicted_sho_DNN), 'r')
        plt.plot(wvec, np.real(predicted_sho_LS), 'g')
        plt.plot(wvec, np.real(predicted_sho_hybrid), 'b')
        plt.tight_layout()
        plt.show()


        plt.xlabel('Relative frequency')
        plt.ylabel('Imaginary part')

        plt.plot(wvec, np.imag(sho_ex), 'k:')
        plt.plot(wvec, np.imag(predicted_sho_DNN), 'r')
        plt.plot(wvec, np.imag(predicted_sho_LS), 'g')
        plt.plot(wvec, np.imag(predicted_sho_hybrid), 'b')
        plt.tight_layout()
        plt.show()
        print(predicted_parms_DNN)
        print(predicted_parms_LS)
        print(predicted_parms_hybrid)
    
    def save_results(self,which_ones='all'):

        if which_ones=='LS' or which_ones=='all':
            np.save(file=self.file+'_amplitude_ls',arr=self.ls_fit['amplitude'])
            np.save(file=self.file+'_resonant_ls',arr=self.ls_fit['resonant_frequency'])
            np.save(file=self.file+'_Q_ls',arr=self.ls_fit['Q'])
            np.save(file=self.file+'_phase_ls',arr=self.ls_fit['phase'])
        if which_ones=='hybrid' or which_ones=='all':
            np.save(file=self.file+'_amplitude_h',arr=self.hybrid_fit['amplitude'])
            np.save(file=self.file+'_resonant_h',arr=self.hybrid_fit['resonant_frequency'])
            np.save(file=self.file+'_Q_h',arr=self.hybrid_fit['Q'])
            np.save(file=self.file+'_phase_h',arr=self.hybrid_fit['phase'])
        if which_ones=='DNN' or which_ones=='all':
            np.save(file=self.file+'_amplitude_dnn',arr=self.DNN_fit['amplitude'])
            np.save(file=self.file+'_resonant_dnn',arr=self.DNN_fit['resonant_frequency'])
            np.save(file=self.file+'_Q_dnn',arr=self.DNN_fit['Q'])
            np.save(file=self.file+'_phase_dnn',arr=self.DNN_fit['phase'])

    def load_fit_results(self,which_ones='all'):
        
        if which_ones=='hybrid' or which_ones=='all':
            self.hybrid_fit['amplitude']=np.load(file=self.file+'_amplitude_h',arr=self.hybrid_fit['amplitude'])
            self.hybrid_fit['resonant_frequency']=np.load(file=self.file+'_resonant_h',arr=self.hybrid_fit['resonant_frequency'])
            self.hybrid_fit['Q']=np.load(file=self.file+'_Q_h',arr=self.hybrid_fit['Q'])
            self.hybrid_fit['phase']=np.load(file=self.file+'_phase_h',arr=self.hybrid_fit['phase'])
        if which_ones=='LS' or which_ones=='all':
            self.ls_fit['amplitude']=np.load(file=self.file+'_amplitude_ls',arr=self.ls_fit['amplitude'])
            self.ls_fit['resonant_frequency']=np.load(file=self.file+'_resonant_ls',arr=self.ls_fit['resonant_frequency'])
            self.ls_fit['Q']=np.load(file=self.file+'_Q_ls',arr=self.ls_fit['Q'])
            self.ls_fit['phase']=np.load(file=self.file+'_phase_ls',arr=self.ls_fit['phase'])
        if which_ones=='DNN' or which_ones=='all':
            self.DNN_fit['amplitude']=np.load(file=self.file+'_amplitude_dnn',arr=self.DNN_fit['amplitude'])
            self.DNN_fit['resonant_frequency']=np.load(file=self.file+'_resonant_dnn',arr=self.DNN_fit['resonant_frequency'])
            self.DNN_fit['Q']=np.load(file=self.file+'_Q_dnn',arr=self.DNN_fit['Q'])
            self.DNN_fit['phase']=np.load(file=self.file+'_phase_dnn',arr=self.DNN_fit['phase'])
        
        