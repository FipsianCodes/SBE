
import src.parser as param

from dataclasses import dataclass, field 
import numpy as np
import sys

@dataclass
class FDM():

    params : param.PARAMS

    prop : str 

    N  : int        = field(init=False)
    M  : int        = field(init=False)

    h  : float      = field(init=False)
    r  : float      = field(init=False)

    dx : float      = field(init=False)
    nu : float      = field(init=False)
    Cs : float      = field(init=False)

    smag : bool     = field(init=False)

    ADV : str       = field(init=False)

    k : np.ndarray  = field(init=False)

    def __post_init__(self) -> None:

        if self.prop == 'REFERENCE':

            self.N          = int(self.params.N)
            self.M          = int(self.N/2)

            self.dx         = float(2.0*np.pi/self.N)
            self.nu         = float(self.params.nu)
            self.Cs         = float(self.params.Cs)

            self.h          = 2.0*np.pi/self.N
            self.r          = self.h/self.dx

            self.k          = np.fft.fftfreq(self.N,d=1/self.N)
            self.k[self.M]  = 0

            self.smag       = self.params.smag
            self.ADV        = self.params.ADV

        elif self.prop == 'PAR_COARSE':
            
            self.N          = int(self.params.par_N)
            self.M          = int(self.N/2)

            self.dx         = float(2.0*np.pi/self.N)
            self.nu         = float(self.params.par_nu)
            self.Cs         = float(self.params.par_Csc)

            self.h          = 2.0*np.pi/self.N
            self.r          = self.h/self.dx

            self.k          = np.fft.fftfreq(self.N,d=1/self.N)
            self.k[self.M]  = 0

            self.smag       = self.params.par_smagc
            self.ADV        = self.params.par_ADVc

        elif self.prop == 'PAR_FINE':
            
            self.N          = int(self.params.par_N)
            self.M          = int(self.N/2)

            self.dx         = float(2.0*np.pi/self.N)
            self.nu         = float(self.params.par_nu)
            self.Cs         = float(self.params.par_Csf)

            self.h          = 2.0*np.pi/self.N
            self.r          = self.h/self.dx

            self.k          = np.fft.fftfreq(self.N,d=1/self.N)
            self.k[self.M]  = 0

            self.smag       = self.params.par_smagf
            self.ADV        = self.params.par_ADVf

        elif self.prop == 'MMPAR_COARSE':
            
            self.N          = int(self.params.mmpar_Nc)
            self.M          = int(self.N/2)

            self.dx         = float(2.0*np.pi/self.N)
            self.nu         = float(self.params.mmpar_nu)
            self.Cs         = float(self.params.mmpar_Csc)

            self.h          = 2.0*np.pi/self.N
            self.r          = self.h/self.dx

            self.k          = np.fft.fftfreq(self.N,d=1/self.N)
            self.k[self.M]  = 0

            self.smag       = self.params.mmpar_smagc
            self.ADV        = self.params.mmpar_ADVc
                
        elif self.prop == 'MMPAR_FINE':
            
            self.N          = int(self.params.mmpar_Nf)
            self.M          = int(self.N/2)

            self.dx         = float(2.0*np.pi/self.N)
            self.nu         = float(self.params.mmpar_nu)
            self.Cs         = float(self.params.mmpar_Csf)

            self.h          = 2.0*np.pi/self.N
            self.r          = self.h/self.dx

            self.k          = np.fft.fftfreq(self.N,d=1/self.N)
            self.k[self.M]  = 0

            self.smag       = self.params.mmpar_smagf
            self.ADV        = self.params.mmpar_ADVf
        
        else:
            sys.exit('Discretization Method unknown! Should be: REFERENCE , PAR_COARSE , PAR_FINE , MMPAR_COARSE , MMPAR_FINE')   


    def dudx(self,u):
        
        return (np.roll(u,-1) - np.roll(u,1))/(2.0*self.dx)

    def du2dx(self,u):
        
        if self.ADV == 'con':
            u1 = self.dudx(u)
            u2 = self.dudx(np.multiply(u,u))
            d2udx = (np.multiply(u,u1) + u2)/3.0
        elif self.ADV == 'adv':
            u1 = self.dudx(u)
            d2udx = np.multiply(u,u1)
        elif self.ADV == 'div':
            d2udx = 0.5*self.dudx(np.multiply(u,u))
        else:
            pass

        return d2udx 


    def d3udx3(self,u):
        pass

    def d2udx2(self,u):

        return (np.roll(u,-1) - 2.0*u + np.roll(u,1))/(self.dx*self.dx)

    def RHS(self,u):
        
        d2udx2 = self.d2udx2(u)
        du2dx  = self.du2dx(u)

        if self.smag:
            
            dudx    = self.dudx(u) 

            gradUp  = np.roll(dudx,-1)
            gradUm  = np.roll(dudx,1)
            gradU2  = (np.multiply(gradUp,np.abs(gradUp)) - np.multiply(gradUm,np.abs(gradUm)))/(2.0*self.dx)

            Cs2     = self.Cs*self.Cs
           
            Su      = Cs2*self.dx*self.dx*gradU2
           
            rhs = self.nu*d2udx2 - du2dx + Su

        else:

            rhs = self.nu*d2udx2 - du2dx 

        return rhs

    def which(self):
        return self.prop
