
import src.parser as param

from dataclasses import dataclass, field 
import numpy as np

@dataclass 
class FFT():

    params : param.PARAMS

    # define propagator: coarse, fine, reference
    prop : str 

    
    N  : int        = field(init=False)
    M  : int        = field(init=False)

    h  : float      = field(init=False)
    r  : float      = field(init=False)
    
    dx : float      = field(init=False)
    nu : float      = field(init=False)
    Cs : float      = field(init=False)

    smag : bool     = field(init=False)

    k  : np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        
        if self.prop == 'REFERENCE':
            self.N      = int(self.params.N)
            self.M      = int(self.N/2)

            self.dx     = float(2.0*np.pi/self.N)
            self.nu     = float(self.params.nu)
            self.Cs     = float(self.params.Cs)

            self.h      = 2.0*np.pi/self.N
            self.r      = self.h/self.dx

            self.k          = np.fft.fftfreq(self.N,d=1/self.N)
            self.k[self.M]  = 0

            self.smag       = self.params.smag

        elif self.prop == 'PAR_COARSE':
            self.N      = int(self.params.par_N)
            self.M      = int(self.N/2)

            self.dx     = float(2.0*np.pi/self.N)
            self.nu     = float(self.params.par_nu)
            self.Cs     = float(self.params.par_Csc)

            self.h      = 2.0*np.pi/self.N
            self.r      = self.h/self.dx

            self.k          = np.fft.fftfreq(self.N,d=1/self.N)
            self.k[self.M]  = 0

            self.smag       = self.params.par_smagc

        elif self.prop == 'PAR_FINE':
            self.N      = int(self.params.par_N)
            self.M      = int(self.N/2)

            self.dx     = float(2.0*np.pi/self.N)
            self.nu     = float(self.params.par_nu)
            self.Cs     = float(self.params.par_Csf)

            self.h      = 2.0*np.pi/self.N
            self.r      = self.h/self.dx

            self.k          = np.fft.fftfreq(self.N,d=1/self.N)
            self.k[self.M]  = 0

            self.smag       = self.params.par_smagf

        elif self.prop == 'MMPAR_COARSE':
            self.N      = int(self.params.mmpar_Nc)
            self.M      = int(self.N/2)

            self.dx     = float(2.0*np.pi/self.N)
            self.nu     = float(self.params.mmpar_nu)
            self.Cs     = float(self.params.mmpar_Csc)

            self.h      = 2.0*np.pi/self.N
            self.r      = self.h/self.dx

            self.k          = np.fft.fftfreq(self.N,d=1/self.N)
            self.k[self.M]  = 0

            self.smag       = self.params.mmpar_smagc

        elif self.prop == 'MMPAR_FINE':
            self.N      = int(self.params.mmpar_Nf)
            self.M      = int(self.N/2)

            self.dx     = float(2.0*np.pi/self.N)
            self.nu     = float(self.params.mmpar_nu)
            self.Cs     = float(self.params.mmpar_Csf)

            self.h      = 2.0*np.pi/self.N
            self.r      = self.h/self.dx

            self.k          = np.fft.fftfreq(self.N,d=1/self.N)
            self.k[self.M]  = 0

            self.smag       = self.params.mmpar_smagf

        else:
            pass



    def dudx(self,fftu):
       
        return self.r*np.real(np.fft.ifft(np.emath.sqrt(-1)*self.k*fftu))

    def du2dx(self,fftu):

        padZero = np.zeros(self.N)
        fupad   = np.insert(fftu,self.M,padZero)
        upad    = np.real(np.fft.ifft(fupad))
        u2      = np.multiply(upad,upad)
        fftu2pad= np.fft.fft(u2)
        fftu2   = fftu2pad[0:self.M]
        fftu2   = np.append(fftu2,fftu2pad[self.N+self.M:])
        
        return self.r*np.real(np.fft.ifft(np.emath.sqrt(-1)*self.k*fftu2))

    def d3udx3(self,u):
        pass

    def d2udx2(self,fftu):

        return self.r**2*np.real(np.fft.ifft(-self.k*self.k*fftu))


    def dealias_square(self,u2):
        
        # 3/2 rule
        fu2   = np.fft.fft(u2)
        Fu2   = np.concatenate((fu2[0:self.M+1],fu2[2*self.M+1:self.M+self.N]))
        # set Nyquist wavenumber to zero
        Fu2[self.M] = 0

        return (3/2)*np.real(np.fft.ifft(Fu2))

    def dealias_pad(self,u):

        # compute fft then de-alias
        fu  = np.fft.fft(u)
        fup = np.concatenate((fu[0:self.M+1],np.zeros(self.M),fu[self.M+1:self.N]))

        # return from spectral space
        return np.real(np.fft.ifft(fup))

    def RHS(self,u):

        fftu = np.fft.fft(u)

        if self.smag:
            # Eddy Viscosity model with constant Cs
            dudx=self.dudx(fftu)

            Cs2         = self.Cs*self.Cs
            gradUabs    = self.dealias_pad(np.abs(dudx))
            gradU       = self.dealias_pad(dudx)
            gradU2      = self.dealias_square(np.multiply(gradUabs,gradU))
            res         = Cs2*(self.dx*self.dx)*gradU2

            Su          = self.dudx(np.fft.fft(res))
            
            rhs = self.nu*self.d2udx2(fftu) - self.du2dx(fftu) + Su
        else:
            # DNS/UDNS case
            rhs = self.nu*self.d2udx2(fftu) - self.du2dx(fftu)

        return rhs

    def which(self):
        return self.prop





