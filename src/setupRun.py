
import os
import numpy as np 
from dataclasses import dataclass, field
import src.parser as param
import src.interpolation as intp

@dataclass 
class simulation_setup():

    params : param.PARAMS

    # Initial Value 

    u0 : np.ndarray = field(init=False)

    # Reference Init 

    Xr : np.ndarray = field(init=False)
    ur : np.ndarray = field(init=False) # time local
    Ur : np.ndarray = field(init=False)
    Er : np.ndarray = field(init=False) # Energy Spectrum
    KEr: np.ndarray = field(init=False) # Kinetic Energy

    dx : float      = field(init=False)

    # Parareal Init

    par_u0   : np.ndarray   = field(init=False)

    par_X    : np.ndarray   = field(init=False)
    par_g    : np.ndarray   = field(init=False)
    par_Ug   : np.ndarray   = field(init=False)
    par_g1   : np.ndarray   = field(init=False)
    par_Ug1  : np.ndarray   = field(init=False)
    par_f    : np.ndarray   = field(init=False)
    par_Uf   : np.ndarray   = field(init=False)

    par_Eg   : np.ndarray   = field(init=False)
    par_Eg1  : np.ndarray   = field(init=False)
    par_Ef   : np.ndarray   = field(init=False)

    par_KEg  : np.ndarray   = field(init=False)
    par_KEg1 : np.ndarray   = field(init=False)
    par_KEf  : np.ndarray   = field(init=False)

    par_u    : np.ndarray   = field(init=False)
    par_Uu   : np.ndarray   = field(init=False)
    par_KE   : np.ndarray   = field(init=False) 
    par_E    : np.ndarray   = field(init=False) 

    par_dx   : float  = field(init=False)

    # MMPAR Init

    mmpar_u0c : np.ndarray     = field(init=False)
    mmpar_u0f : np.ndarray     = field(init=False)

    mmpar_Xc    : np.ndarray   = field(init=False)
    mmpar_Xf    : np.ndarray   = field(init=False)

    mmpar_g     : np.ndarray   = field(init=False)
    mmpar_Ug    : np.ndarray   = field(init=False)
    mmpar_g1    : np.ndarray   = field(init=False)
    mmpar_Ug1   : np.ndarray   = field(init=False)
    mmpar_f     : np.ndarray   = field(init=False)
    mmpar_Uf    : np.ndarray   = field(init=False)

    mmpar_Eg    : np.ndarray   = field(init=False)
    mmpar_Eg1   : np.ndarray   = field(init=False)
    mmpar_Ef    : np.ndarray   = field(init=False)

    mmpar_KEg   : np.ndarray   = field(init=False)
    mmpar_KEg1  : np.ndarray   = field(init=False)
    mmpar_KEf   : np.ndarray   = field(init=False)

    mmpar_u     : np.ndarray   = field(init=False)
    mmpar_Uu    : np.ndarray   = field(init=False)
    mmpar_KE    : np.ndarray   = field(init=False) 
    mmpar_E     : np.ndarray   = field(init=False) 

    mmpar_dxc   : float  = field(init=False)
    mmpar_dxf   : float  = field(init=False)

    mmpar_R     : str = field(init=False)

    def __post_init__(self):
        
        # Reference run initialization

        N       = self.params.N
        self.dx = 2.0*np.pi/N
        self.Xr = np.arange(0,2*np.pi,self.dx)

        self.ur = np.zeros((N,self.params.Nt+1))
        self.Er = np.zeros((N,self.params.Nt))
    
        n = int((self.params.Tn-self.params.T0)/(self.params.Nt*self.params.dt*self.params.nn))
        self.KEr = np.zeros((self.params.Nt,n,2))

        if self.params.init:
            if self.params.noise == 2:
                self.u0 = np.zeros(N)
            elif self.params.noise == 1:
                self.u0 = 2.0*np.random.rand(N)-1.0
            else:
                pass
        else:
            path = self.params.path_init
            self.u0 = np.loadtxt(path)
            
        nf = int((self.params.Tn-self.params.T0)/(self.params.Nt*self.params.dt*self.params.nn))

        if self.params.write:
            self._U = np.zeros((self.params.Nt,nf,self.params.Nf))
        else:
                self.U  = np.zeros((self.params.Nt,2,2))

        # init ur
        self.ur[:,0] = np.copy(self.u0)

        # Parareal initialization

        if self.params.run_par:
       
            self.par_dx = 2.0*np.pi/self.params.par_N 
            self.par_X  = np.arange(0,2.0*np.pi,self.par_dx)

            self.par_u  = np.zeros((self.params.par_N,self.params.par_Nt+1))
            self.par_g  = np.zeros((self.params.par_N,self.params.par_Nt+1))
            self.par_g1 = np.zeros((self.params.par_N,self.params.par_Nt+1))
            self.par_f  = np.zeros((self.params.par_N,self.params.par_Nt+1))

            self.par_E  = np.zeros((self.params.par_N,self.params.par_Nt))
            self.par_Eg = np.zeros((self.params.par_N,self.params.par_Nt))
            self.par_Eg1= np.zeros((self.params.par_N,self.params.par_Nt))
            self.par_Ef = np.zeros((self.params.par_N,self.params.par_Nt))

            nf = int((self.params.par_Tn-self.params.par_T0)/(self.params.par_Nt*self.params.par_dtf*self.params.nn))
            nc = int((self.params.par_Tn-self.params.par_T0)/(self.params.par_Nt*self.params.par_dtf*self.params.nn))
    
            self.par_KE     = np.zeros((self.params.par_Nt,nf,2))
            self.par_KEf    = np.zeros((self.params.par_Nt,nf,2))
            self.par_KEg    = np.zeros((self.params.par_Nt,nc,2))
            self.par_KEg1   = np.zeros((self.params.par_Nt,nc,2))
            
            if self.params.write:
                self.par_Uu  = np.zeros((self.params.par_Nt,nf,self.params.par_N))
                self.par_Ug  = np.zeros((self.params.par_Nt,nf,self.params.par_N))
                self.par_Ug1 = np.zeros((self.params.par_Nt,nf,self.params.par_N))
                self.par_Uf  = np.zeros((self.params.par_Nt,nf,self.params.par_N))
            else:
                self.par_Uu  = np.zeros((self.params.par_Nt,2,2))
                self.par_Ug  = np.zeros((self.params.par_Nt,2,2))
                self.par_Ug1 = np.zeros((self.params.par_Nt,2,2))
                self.par_Uf  = np.zeros((self.params.par_Nt,2,2))


            if self.params.init:
                if self.params.noise == 2:
                    self.par_u0 = np.zeros(self.params.par_N)
                elif self.params.noise == 1:
                    self.par_u0 = 2.0*np.random.rand(self.params.par_N)-1.0
                else:
                    pass
            else:
                path = self.params.path_init
                self.par_u0 = np.loadtxt(path)

            self.par_u[:,0]     = np.copy(self.par_u0)
            self.par_g[:,0]     = np.copy(self.par_u0)
            self.par_g1[:,0]    = np.copy(self.par_u0)
            self.par_f[:,0]     = np.copy(self.par_u0)

        else: 
            pass

        # MMPAR initialization
    
        if self.params.run_mmpar:
            
            self.mmpar_dxc    = 2.0*np.pi/self.params.mmpar_Nc
            self.mmpar_Xc     = np.arange(0,2.0*np.pi,self.mmpar_dxc)

            self.mmpar_dxf    = 2.0*np.pi/self.params.mmpar_Nf
            self.mmpar_Xf     = np.arange(0,2.0*np.pi,self.mmpar_dxf)

            self.mmpar_u  = np.zeros((self.params.mmpar_Nf,self.params.mmpar_Nt+1))
            self.mmpar_f  = np.zeros((self.params.mmpar_Nf,self.params.mmpar_Nt+1))

            self.mmpar_g  = np.zeros((self.params.mmpar_Nc,self.params.mmpar_Nt+1))
            self.mmpar_g1 = np.zeros((self.params.mmpar_Nc,self.params.mmpar_Nt+1))

            self.mmpar_E  = np.zeros((self.params.mmpar_Nf,self.params.mmpar_Nt))
            self.mmpar_Ef = np.zeros((self.params.mmpar_Nf,self.params.mmpar_Nt))
            
            self.mmpar_Eg = np.zeros((self.params.mmpar_Nc,self.params.mmpar_Nt))
            self.mmpar_Eg1= np.zeros((self.params.mmpar_Nc,self.params.mmpar_Nt))

            nf = int((self.params.mmpar_Tn-self.params.mmpar_T0)/(self.params.mmpar_Nt*self.params.par_dtf*self.params.nn))
            nc = int((self.params.mmpar_Tn-self.params.mmpar_T0)/(self.params.mmpar_Nt*self.params.par_dtf*self.params.nn))

            if self.params.write:
                self.mmpar_Uu  = np.zeros((self.params.mmpar_Nt,nf,self.params.mmpar_Nf))
                self.mmpar_Ug  = np.zeros((self.params.mmpar_Nt,nf,self.params.mmpar_Nc))
                self.mmpar_Ug1 = np.zeros((self.params.mmpar_Nt,nf,self.params.mmpar_Nc))
                self.mmpar_Uf  = np.zeros((self.params.mmpar_Nt,nf,self.params.mmpar_Nf))
            else:
                self.mmpar_Uu  = np.zeros((self.params.mmpar_Nt,2,2))
                self.mmpar_Ug  = np.zeros((self.params.mmpar_Nt,2,2))
                self.mmpar_Ug1 = np.zeros((self.params.mmpar_Nt,2,2))
                self.mmpar_Uf  = np.zeros((self.params.mmpar_Nt,2,2))

            self.mmpar_KE     = np.zeros((self.params.mmpar_Nt,nf,2))
            self.mmpar_KEf    = np.zeros((self.params.mmpar_Nt,nf,2))
            self.mmpar_KEg    = np.zeros((self.params.mmpar_Nt,nc,2))
            self.mmpar_KEg1   = np.zeros((self.params.mmpar_Nt,nc,2))

            if self.params.init:
                if self.params.noise == 2:
                    self.mmpar_u0f = np.zeros(self.params.mmpar_Nf)
                    self.mmpar_u0c = np.zeros(self.params.mmpar_Nc)
                elif self.params.noise == 1:
                    self.mmpar_u0f = 2.0*np.random.rand(self.params.mmpar_Nf)-1.0
                    self.mmpar_u0c = 2.0*np.random.rand(self.params.mmpar_Nc)-1.0
                else:
                    pass
            else:
                path = self.params.path_init
                self.mmpar_u0f = np.loadtxt(path)
                ## TODO: add interpolation from read-in solution to u0c

            self.mmpar_u[:,0]     = np.copy(self.mmpar_u0f)
            self.mmpar_g[:,0]     = np.copy(self.mmpar_u0c)
            self.mmpar_g1[:,0]    = np.copy(self.mmpar_u0c)
            self.mmpar_f[:,0]     = np.copy(self.mmpar_u0f)

        else: 
            pass

    def save_reference(self,finfo):
       
        E  = np.copy(self.Er[:,0])
        [l,m,n] = np.shape(self.KEr)
        KE = np.zeros((l*m,2))
        KE[0:m,:] = np.copy(self.KEr[0,:,:])

        for i in range(1,self.params.Nt):
            E  = E + self.Er[:,i]
            KE[i*m:(i+1)*m,:] = np.copy(self.KEr[i,:,:])

        E=E/self.params.Nt

        if os.path.exists(self.params.path):

            np.savetxt(self.params.path+'ref_E_N'+str(self.params.N)+'_'+finfo,E)
            np.savetxt(self.params.path+'ref_KE_N'+str(self.params.N)+'_'+finfo,KE)
            np.savetxt(self.params.path+'ref_u_N'+str(self.params.N)+'_'+finfo,self.ur)

        else:
        
            os.makedirs(self.params.path)

            np.savetxt(self.params.path+'ref_E_N'+str(self.params.N)+'_'+finfo,E)
            np.savetxt(self.params.path+'ref_KE_N'+str(self.params.N)+'_'+finfo,KE)
            np.savetxt(self.params.path+'ref_u_N'+str(self.params.N)+'_'+finfo,self.ur)
            

    def save_par_iteration(self,it):

        if os.path.exists(self.params.path):
            pass
        else:
            os.makedirs(self.params.path)

        PATH=self.params.path+'PAR_iteration_'+str(it)+'/'
        
        if os.path.exists(PATH):
            pass
        else:
            os.makedirs(PATH)

        [l,m,n] = np.shape(self.par_KE)
        [L,M,N] = np.shape(self.par_KEg)
        
        KE      = np.zeros((l*m,n))
        KEf     = np.zeros((l*m,n))

        KEg     = np.zeros((L*M,N))
        KEg1    = np.zeros((L*M,N))

        KE[0:m,:] = np.copy(self.par_KE[0,:,:])
        KEf[0:m,:] = np.copy(self.par_KEf[0,:,:])

        KEg[0:M,:] = np.copy(self.par_KEg[0,:,:])
        KEg1[0:M,:] = np.copy(self.par_KEg1[0,:,:])

        E      = np.copy(self.par_E[:,0])
        Ef     = np.copy(self.par_Ef[:,0])
        Eg     = np.copy(self.par_Eg[:,0])
        Eg1    = np.copy(self.par_Eg1[:,0])

        for i in range(1,self.params.par_Nt):
            E  = E + self.par_E[:,i]
            Eg = Eg + self.par_Eg[:,i]
            Eg1= Eg1 + self.par_Eg1[:,i]
            Ef = Ef + self.par_Ef[:,i]

            KE[i*m:(i+1)*m,:] = np.copy(self.par_KE[i,:,:])
            KEf[i*m:(i+1)*m,:] = np.copy(self.par_KEf[i,:,:])

            KEg[i*M:(i+1)*M,:] = np.copy(self.par_KEg[i,:,:])
            KEg1[i*M:(i+1)*M,:] = np.copy(self.par_KEg1[i,:,:])

        E=E/self.params.par_Nt
        Eg=Eg/self.params.par_Nt
        Eg1=Eg1/self.params.par_Nt
        Ef=Ef/self.params.par_Nt


        np.savetxt(PATH+'PAR_U_E',E)
        np.savetxt(PATH+'PAR_U_KE',KE)
        np.savetxt(PATH+'PAR_U_u',self.par_u)

        np.savetxt(PATH+'PAR_G_E',Eg)
        np.savetxt(PATH+'PAR_G_KE',KEg)
        np.savetxt(PATH+'PAR_G_u',self.par_g)

        np.savetxt(PATH+'PAR_F_E',Ef)
        np.savetxt(PATH+'PAR_F_KE',KEf)
        np.savetxt(PATH+'PAR_F_u',self.par_f)

        np.savetxt(PATH+'PAR_G1_E',Eg1)
        np.savetxt(PATH+'PAR_G1_KE',KEg1)
        np.savetxt(PATH+'PAR_G1_u',self.par_g1)

    def save_mmpar_iteration(self,it):

        if os.path.exists(self.params.path):
            pass
        else:
            os.makedirs(self.params.path)

        PATH=self.params.path+'MMPAR_iteration_'+str(it)+'/'
        
        if os.path.exists(PATH):
            pass
        else:
            os.makedirs(PATH)

        [l,m,n] = np.shape(self.mmpar_KE)
        [L,M,N] = np.shape(self.mmpar_KEg)
        
        KE      = np.zeros((l*m,n))
        KEf     = np.zeros((l*m,n))

        KEg     = np.zeros((L*M,N))
        KEg1    = np.zeros((L*M,N))

        KE[0:m,:] = np.copy(self.mmpar_KE[0,:,:])
        KEf[0:m,:] = np.copy(self.mmpar_KEf[0,:,:])

        KEg[0:M,:] = np.copy(self.mmpar_KEg[0,:,:])
        KEg1[0:M,:] = np.copy(self.mmpar_KEg1[0,:,:])

        E      = np.copy(self.mmpar_E[:,0])
        Ef     = np.copy(self.mmpar_Ef[:,0])
        Eg     = np.copy(self.mmpar_Eg[:,0])
        Eg1    = np.copy(self.mmpar_Eg1[:,0])

        for i in range(1,self.params.mmpar_Nt):
            E  = E + self.mmpar_E[:,i]
            Eg = Eg + self.mmpar_Eg[:,i]
            Eg1= Eg1 + self.mmpar_Eg1[:,i]
            Ef = Ef + self.mmpar_Ef[:,i]

            KE[i*m:(i+1)*m,:] = np.copy(self.mmpar_KE[i,:,:])
            KEf[i*m:(i+1)*m,:] = np.copy(self.mmpar_KEf[i,:,:])

            KEg[i*M:(i+1)*M,:] = np.copy(self.mmpar_KEg[i,:,:])
            KEg1[i*M:(i+1)*M,:] = np.copy(self.mmpar_KEg1[i,:,:])

        E=E/self.params.mmpar_Nt
        Eg=Eg/self.params.mmpar_Nt
        Eg1=Eg1/self.params.mmpar_Nt
        Ef=Ef/self.params.mmpar_Nt

        if self.params.write:

            [L,M,N] = np.shape(self.mmpar_Uu)
            [l,m,n] = np.shape(self.mmpar_Ug)

            Uu  = np.zeros((L*M,N))
            Uf  = np.zeros((L*M,N))
            Ug  = np.zeros((l*m,n))
            Ug1 = np.zeros((l*m,n))

            Uu[0:M,:]  = np.copy(self.mmpar_Uu[0,:,:])
            Uf[0:M,:]  = np.copy(self.mmpar_Uf[0,:,:])
            Ug[0:m,:]  = np.copy(self.mmpar_Ug[0,:,:])
            Ug1[0:m,:] = np.copy(self.mmpar_Ug1[0,:,:])

            for i in range(1,self.params.mmpar_Nt):
                Uu[i*M:(i+1)*M,:]  = np.copy(self.mmpar_Uu[i,:,:])
                Uf[i*M:(i+1)*M,:]  = np.copy(self.mmpar_Uf[i,:,:])
                Ug[i*m:(i+1)*m,:]  = np.copy(self.mmpar_Ug[i,:,:])
                Ug1[i*m:(i+1)*m,:] = np.copy(self.mmpar_Ug1[i,:,:])

            np.savetxt(PATH+'MMPAR_U_uu',np.transpose(Uu))
            np.savetxt(PATH+'MMPAR_F_uu',np.transpose(Uf))
            np.savetxt(PATH+'MMPAR_G_uu',np.transpose(Ug))
            np.savetxt(PATH+'MMPAR_G1_uu',np.transpose(Ug1))

        np.savetxt(PATH+'MMPAR_U_E',E)
        np.savetxt(PATH+'MMPAR_U_KE',KE)
        np.savetxt(PATH+'MMPAR_U_u',self.mmpar_u)

        np.savetxt(PATH+'MMPAR_G_E',Eg)
        np.savetxt(PATH+'MMPAR_G_KE',KEg)
        np.savetxt(PATH+'MMPAR_G_u',self.mmpar_g)

        np.savetxt(PATH+'MMPAR_F_E',Ef)
        np.savetxt(PATH+'MMPAR_F_KE',KEf)
        np.savetxt(PATH+'MMPAR_F_u',self.mmpar_f)

        np.savetxt(PATH+'MMPAR_G1_E',Eg1)
        np.savetxt(PATH+'MMPAR_G1_KE',KEg1)
        np.savetxt(PATH+'MMPAR_G1_u',self.mmpar_g1)

